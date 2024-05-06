# New RAG

## Environment Setup

```commandline
conda create -p ./venv python --y
conda init zsh
conda activate ./venv
```

- install metal

```commandline
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install --upgrade --quiet  langchain langchain-community langchain-chroma sentence-transformers
```

---

## References

- https://llama-cpp-python.readthedocs.io/en/latest/
- https://python.langchain.com/docs/use_cases/question_answering/quickstart/
- https://python.langchain.com/docs/integrations/vectorstores/chroma/