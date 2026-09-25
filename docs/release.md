# 发布前操作

公开范围限于当前独立目录。原工程、原 Git 历史和本机运行文件不随源码包发布。

## 本地验收

按快速开始文档安装后，在独立框架目录执行：

```bash
python -m pytest -q
python -m agent_framework.demo
python examples/explicit_approval.py
python -m pip check
python scripts/check_release.py
```

最后一个命令检查许可证配置、预定源码范围、常见凭据格式和个人路径，不替代完整安全审计或原仓库历史扫描。

## 许可证与署名

仓库名称为 `evidence-agent`，作者署名为 `hatetakename`，许可证为 [MIT](../LICENSE)。版权声明、包元数据和 README 已同步。

Python 分发包名保留为 `evidence-agent-framework`，导入名为 `agent_framework`。仓库名称不要求与这两个名称一致；未在 PyPI 发布或声明包名已被预留。

## 导出独立源码

```bash
# 输出目录必须在源码目录之外。
python scripts/check_release.py --archive ../evidence-agent-0.1.0.zip
```

输出为新建 ZIP，不覆盖同名文件。包中包含逐文件 SHA-256 清单；`.git`、缓存、运行数据库、虚拟环境和构建目录不会进入导出包。导出工具发现未知文件、链接或常见密钥格式会拒绝打包，报告只包含位置。

将已审查的独立源码放入新的仓库，使用新的 Git 历史。远程建库、推送、发布版本及软件包分发是后续发布动作；本地准备工具不会执行这些操作。

## Python 分发包

```bash
python -m pip install ".[build]"
python -m build
```

分发前分别检查 wheel 和源码归档的文件列表。在新的虚拟环境安装 wheel，切换到源码目录外运行示例，确认没有靠当前工作目录导入原项目。GitHub CI 配置覆盖 Windows/Linux 与 Python 3.11/3.12；配置存在不表示这些远程任务已经执行。
