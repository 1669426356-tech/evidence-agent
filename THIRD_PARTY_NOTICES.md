# Third-party dependencies

This source tree does not vendor third-party library source code. Dependencies
are installed separately through the Python package manager and retain their
own license terms.

The following direct-dependency license labels were verified from the installed
distribution metadata during local preparation on 2026-09-25:

| Dependency | Tested version | Metadata license | Upstream |
|---|---|---|---|
| LangGraph | 1.2.12 | MIT | https://github.com/langchain-ai/langgraph |
| LangChain Core | 1.6.5 | MIT | https://github.com/langchain-ai/langchain |
| Pydantic | 2.13.5 | MIT | https://github.com/pydantic/pydantic |

This list documents direct dependencies, not a license audit of every transitive
dependency. See `constraints-tested.txt` for the complete tested environment.
The framework itself is licensed under the MIT License, copyright (c) 2026
hatetakename. See the root `LICENSE` for its complete terms.
