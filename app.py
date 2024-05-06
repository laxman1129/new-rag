from ingest import load_docs
from rag import query_docs


def main():
    load_docs("./readme.md")
    query_docs()


if __name__ == '__main__':
    main()
