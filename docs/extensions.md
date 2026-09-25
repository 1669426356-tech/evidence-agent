# 扩展工具与模型

## 工具契约

`ToolSpec` 定义名称、说明、输入模型、处理函数及以下声明：

| 字段 | 用途 |
| --- | --- |
| `effect` | `read_only`、`local_compute` 或 `external_write`；只有后者要求显式授权 |
| `version` | 工具行为版本，参与参数指纹 |
| `evidence_key` | 成功结果保存为哪一项证据 |
| `requires` | 执行前必须存在的证据键 |
| `requires_passed` | 执行前必须存在且 `evaluation_passed=True` 的证据键 |
| `invalidates` | 已获许可的调用尝试会清除的证据键 |

新工具必须拒绝额外参数。权限信息不能从诸如 `approved=true` 的模型参数注入。工具名称不能重复。

`requires` 只检查证据是否存在。需要评价通过时，使用 `requires_passed`；缺少证据、评价失败或评价状态为 `None` 均会阻止执行，而且不会消耗外部写入授权。工具评价器负责设置 `evaluation_passed`。更具体的数值门槛及工况匹配仍应由应用明确校验，不要只依赖模型理解。

更新工具行为时同时更新 `version`。指纹由校验和默认值填充后的参数、工具名称、版本及上下文计算；相同默认参数的不同写法可得到相同指纹。

指纹读取全部实际字段值，包括 `Field(exclude=True)` 字段及嵌套模型，不经过可能隐藏或改写值的序列化器。输入字段最终需为有限 JSON 值；不支持依赖私有模型属性或不透明 Python 对象的输入模型。这类资源应由应用通过公开标识符在工具内部解析。

处理函数异常会变为失败结果，框架不自动重试。`external_action_started=True` 表示已经进入外部写入处理函数，不能证明远端成功、失败或完全没有副作用。外部服务应另外提供幂等键、结果查询和恢复机制。

## 证据链

例如，`measure` 产生 `sample`，`analyze` 需要 `sample` 并产生 `analysis`，`evaluate` 需要 `analysis` 并产生 `evaluation`。再次尝试 `measure` 会清除旧 `sample` 及其下游证据，即使此次测量失败，也不会保留旧的通过结论。

工具依赖由注册表声明，`requires` 和 `requires_passed` 都参与下游证据的传递失效。框架不会推断没有声明的依赖关系；完整声明是应用作者的职责。同一模型响应的多个调用逐个通过门控和观察阶段，后一个调用能看到前一个调用导致的失效。

## 自定义模型

```python
from agent_framework.state import Decision


class MyModel:
    def decide(self, state, tools):
        # state 和 tools 是副本。模型只提出调用或文字回答。
        return Decision(answer="No further action is needed.")
```

模型不可通过返回字段改变证据、许可或评价结果。对于不支持 LangChain 工具调用的模型，直接实现此小接口即可。模型错误使本次运行结束并记录经过清理的警告，不输出原始异常消息。

## 自定义反思与经验

反思器实现 `reflect(state, result) -> str`，只返回经验文本，去除首尾空白后最多 4096 个字符。状态、调用状态、评价结果、上下文及标识符由代码提供。反思器失败或文本超限后采用确定性文本，经验库失败不会重新执行已经完成的工具。

经验库实现 `record`、`retrieve`、`close`。`retrieval_limit` 接受 0 到 100 的整数，0 表示关闭检索。内置检索按目标和经验文本中的词项重合度排序，并只检索相同上下文；它没有向量索引，也不声称自动学到可靠策略。反思文本是不可信的历史建议，不能替代当前证据或授权。
