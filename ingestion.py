from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://pib.gov.in/PressReleasePage.aspx?PRID=2089179",
    "https://pib.gov.in/PressReleasePage.aspx?PRID=1897980  ",
    "https://nha.gov.in/PM-JAY",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# vectorestore = Chroma.from_documents(documents=doc_splits,
#                                      collection_name="govt-schemes",
#                                      embedding=OpenAIEmbeddings(),
#                                      persist_directory="chromadb",
#                                      )

retriever = Chroma(
    collection_name="govt-schemes",
    persist_directory="chromadb",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

if __name__ == "__main__":

    questin = input("Ask me: ")
    answer = retriever.invoke(questin)

    print(answer)
