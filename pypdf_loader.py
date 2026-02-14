from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('sample-local-pdf.pdf')
docs=loader.load()
print(len(docs))

