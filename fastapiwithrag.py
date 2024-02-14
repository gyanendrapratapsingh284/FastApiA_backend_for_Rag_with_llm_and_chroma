from fastapi import FastAPI,UploadFile,File
from secrets import token_hex
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import chromadb.config
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader


app = FastAPI(title = "Upload the file using FastApi")

                                                            #
def model_function(File,query):
        
        chunk_size=1000
        chunk_overlap=20
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(File)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        db = Chroma.from_documents(docs, embeddings)

        os.environ["OPENAI_API_KEY"] = "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"
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name)

        chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

        query = query
        matching_docs = db.similarity_search(query)
        answer =  chain.run(input_documents=matching_docs, question=query)
        return answer
@app.post('/predict')
async def predict(file : UploadFile = File(...),q: str | None = None):
        file_ext = file.filename.split(".").pop()
        file_name = token_hex(10)
        file_path = f"{file_name}.{file_ext}"

        with open(file_path,'wb') as f:
            content = await file.read()
            x_x =f.write(content)
            loader = PyPDFLoader(file_path)
            document = loader.load_and_split()
            ans = model_function(document,q)
            return ans