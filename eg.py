pdf_loader= DirectoryLoader(path="./data", glob="**/*.pdf", loader_cls="UnstructuredPDFLoader")
text_loader=DirectoryLoader(path="./data",glob="**/*.txt",loader_cls="text_loader")
csv_loader=DirectoryLoader(path="./data",glob="",loader_cls="")