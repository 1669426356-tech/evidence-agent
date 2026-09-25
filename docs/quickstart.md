# 快速开始

## 环境与演示

在项目根目录创建 Python 3.11+ 虚拟环境，然后执行：

```bash
python -m pip install -e ".[test]"
python -m agent_framework.demo
python -m pytest -q
```

安装依赖后，演示与测试均可离线运行。演示使用 `ScriptedModel`，无需设置服务地址或密钥。

复用维护者本次 Windows / Python 3.12 验证的依赖版本，可用 `python -m pip install -c constraints-tested.txt -e ".[test]"`。约束表记录一次测试环境，不是跨平台兼容性保证；常规安装使用项目声明的版本范围。

## 创建模型与工具

模型实现 `decide(state, tools) -> Decision`。返回 `Decision(calls=[...])` 提出工具调用，返回不含调用的 `Decision(answer="...")` 结束本次任务。每次 `AgentRunner.run` 创建新的任务状态。

工具输入必须是设置 `ConfigDict(extra="forbid")` 的 Pydantic 模型。处理函数接收通过校验的输入，返回 `ToolOutput`。证据由执行结果产生；把文件路径放进 `artifacts` 仅记录位置，不会读取、验证或上传文件。

```python
from agent_framework.memory import SQLiteExperienceStore
from agent_framework.graph import AgentRunner

# model 与 tools 由应用构建，参见 README 的最小例子。
store = SQLiteExperienceStore("experiences.sqlite")
try:
    runner = AgentRunner(model, tools, experience_store=store, max_steps=6)
    state = runner.run("Inspect the current document", context_id="project-a")
    print(state.stop_reason)
    print(state.final_answer)
finally:
    store.close()
```

经验库可在关闭后重新打开。不同 `context_id` 的经验不会互相检索；存储文件本身仍需由宿主应用管理访问权限。不要将敏感内容写入目标、元数据或反思文本。

## 显式授权外部写入

授权由调用方根据自己的用户界面和审核流程建立，框架不从用户语句猜测“同意”。以下例子假定已登记名为 `publish`、声明 `external_write` 的工具：

```python
from agent_framework.policy import ApprovalGrant, ExecutionPolicy
from agent_framework.state import ToolCall

task_id = "reviewed-task-1"
context_id = "project-a"
call = ToolCall(tool_name="publish", arguments={"value": 7})
prepared = tools.prepare(call, context_id)
grant = ApprovalGrant(
    task_id=task_id,
    context_id=context_id,
    tool_name=prepared.spec.name,
    input_fingerprint=prepared.input_fingerprint,
)
policy = ExecutionPolicy([grant])
runner = AgentRunner(model, tools, policy=policy)
state = runner.run("Publish the reviewed value", task_id=task_id, context_id=context_id)
```

模型后续提出的工具与参数必须符合该授权。授权在处理函数开始前消耗；同一策略实例不会再次使用它。不要把保存过的授权重新构造为新策略实例，用来自动重试结果未知的写入。

## 接入实际聊天模型

`LangChainDecisionModel(chat_model)` 接收调用方构建的 LangChain 聊天模型。该模型需支持 `bind_tools` 和 `invoke`；适配器使用 OpenAI 格式的工具定义，解析 `AIMessage.tool_calls`，拒绝无效或重复调用 ID。

```python
from agent_framework.models import LangChainDecisionModel

# chat_model 由宿主应用使用其选择的提供商 SDK 创建。
model = LangChainDecisionModel(chat_model)
runner = AgentRunner(model, tools, max_steps=6)
```

框架不选择提供商、不配置服务地址、不创建带凭证的客户端。接入远程模型时，目标、状态及检索到的经验文本会传递给适配器；应用应据此选择可发送的数据。SDK 的超时和重试也由宿主配置。
