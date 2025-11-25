# Critical (!)

- Always use LangChain/LangGraph in version v1.x! See documentation and migration guides.
- Always execute Python/Pip with the binary in the virtual environment: `.venv/bin/python`
- Always consult the documentation (README.md, DEVELOPMENT.md) before making architectural or feature decisions.
- Always document architectural or feature decisions to allow traceability.

# LangChain/LangGraph v1

- Docs LangChain: https://docs.langchain.com/oss/python/langchain/overview
- Docs LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- Migrate LangChain from 0.x to 1.x: https://docs.langchain.com/oss/python/migrate/langchain-v1
- Migrate LangGraph from 0.x to 1.x: https://docs.langchain.com/oss/python/migrate/langgraph-v1

# Git commits
- Never mention Claude or any other AI software/model in the commit messages.
- Commit files bundled by concern and create multiple commits for multiple concerns.
- Commit messages must be in imperative form and as concise as possible ("Add ...", "Fix ...", "Update ...", "Delete ...", etc.)
