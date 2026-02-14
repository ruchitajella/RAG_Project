from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader=DirectoryLoader(
    path = "Books",
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)
docs = loader.lazy_load()
# print(docs[0].page_content)

for d in docs:
    print(d.metadata)
