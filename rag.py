"""
JARVIS RAG (Retrieval-Augmented Generation)
Document memory and querying functionality
"""

import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHROMA_DB_DIR, DOCUMENTS_DIR, RAG_TOP_K

# Import llm and embeddings from llm_setup (import here to avoid circular dependency)
def get_llm_and_embeddings():
    """Get llm and embeddings from llm_setup module"""
    from llm_setup import llm, embeddings
    return llm, embeddings


class DocumentMemory:
    def __init__(self):
        self.vectorstore = None
        self.documents_loaded = False
    
    def _ensure_vectorstore_loaded(self):
        """Lazily load existing vectorstore if it exists and embeddings are ready"""
        if self.vectorstore is not None:
            return
        
        try:
            if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
                # Import here to avoid circular dependency and ensure embeddings are initialized
                from llm_setup import embeddings
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=embeddings
                )
                self.documents_loaded = True
        except Exception as e:
            # If loading fails, we'll create a new one when documents are added
            pass
        
    def add_documents(self, file_paths: List[str]):
        """Add documents to the vector database"""
        # Try to load existing vectorstore first
        self._ensure_vectorstore_loaded()
        
        from langchain_community.document_loaders import (
            TextLoader, 
            PyPDFLoader,
            Docx2txtLoader
        )
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                    
                docs = loader.load()
                documents.extend(text_splitter.split_documents(docs))
                print(f"📄 Added: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if documents:
            llm, embeddings = get_llm_and_embeddings()
            
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR
                )
            else:
                self.vectorstore.add_documents(documents)
            
            self.documents_loaded = True
            print(f"✅ {len(documents)} document chunks indexed")
        else:
            print("⚠️ No documents loaded")
    
    def query(self, question: str) -> str:
        """Query the document database"""
        # Try to load existing vectorstore first
        self._ensure_vectorstore_loaded()
        
        if not self.documents_loaded or self.vectorstore is None:
            return "No documents have been added yet. Please upload documents first."
        
        try:
            llm, embeddings = get_llm_and_embeddings()
            
            # Simple retrieval without RetrievalQA chain (reduced for speed)
            docs = self.vectorstore.similarity_search(question, k=RAG_TOP_K)
            
            if not docs:
                return "No relevant information found in documents."
            
            # Combine context and ask the LLM
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Based on the following context from documents, answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error querying documents: {str(e)}"


# Create global instance
doc_memory = DocumentMemory()

