import json
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from sentence_transformers import SentenceTransformer
import tiktoken
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import chromadb
from chromadb.config import Settings

# Configuration
@dataclass
class Config:
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # or "all-MiniLM-L6-v2" for local
    LLM_MODEL: str = "gpt-3.5-turbo"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 5
    VECTOR_DB_PATH: str = "./chroma_db"
    DATA_FILE: str = "jiopay_data.json"

class ChunkingStrategy:
    """Different chunking strategies for text processing"""
    
    @staticmethod
    def fixed_chunking(text: str, chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
        """Fixed-size chunking with overlap"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_idx": start,
                "end_idx": end,
                "tokens": len(chunk_tokens)
            })
            
            start = end - overlap if end < len(tokens) else end
            
        return chunks
    
    @staticmethod
    def semantic_chunking(text: str, model, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Semantic chunking based on sentence similarity"""
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return [{"text": text, "tokens": len(tiktoken.get_encoding("cl100k_base").encode(text))}]
        
        embeddings = model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]
            
            if similarity > similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "text": chunk_text,
                    "tokens": len(tiktoken.get_encoding("cl100k_base").encode(chunk_text))
                })
                current_chunk = [sentences[i]]
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "tokens": len(tiktoken.get_encoding("cl100k_base").encode(chunk_text))
            })
            
        return chunks
    
    @staticmethod
    def structural_chunking(text: str, question: str) -> List[Dict[str, Any]]:
        """Structure-aware chunking preserving Q&A format"""
        return [{
            "text": f"Q: {question}\nA: {text}",
            "question": question,
            "answer": text,
            "tokens": len(tiktoken.get_encoding("cl100k_base").encode(f"Q: {question}\nA: {text}"))
        }]

class EmbeddingProvider:
    """Handles different embedding models"""
    
    def __init__(self, provider: str = "openai", model_name: str = "text-embedding-3-small"):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "openai":
            openai.api_key = Config.OPENAI_API_KEY
        elif provider == "sentence_transformers":
            self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.provider == "openai":
            response = openai.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [item.embedding for item in response.data]
        
        elif self.provider == "sentence_transformers":
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embed_texts([query])[0]

class VectorStore:
    """Vector store using ChromaDB"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )
        self.collection_name = "jiopay_support"
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "JioPay customer support knowledge base"}
            )
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents to vector store"""
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "tokens": chunk.get("tokens", 0),
                "chunk_id": i,
                "timestamp": datetime.now().isoformat()
            }
            if "question" in chunk:
                metadata["question"] = chunk["question"]
                metadata["answer"] = chunk["answer"]
            metadatas.append(metadata)
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "source": f"FAQ #{results['metadatas'][0][i].get('chunk_id', i)}"
            })
        
        return search_results

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_provider = EmbeddingProvider("openai", config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(config.VECTOR_DB_PATH)
        self.chunking_strategy = ChunkingStrategy()
        
        # Initialize OpenAI client
        openai.api_key = config.OPENAI_API_KEY
        
        # Performance tracking
        self.query_stats = []
    
    def ingest_data(self, data_file: str, chunking_method: str = "structural"):
        """Ingest data from JSON file"""
        print(f"Loading data from {data_file}...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_chunks = []
        
        for item in data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if chunking_method == "fixed":
                chunks = self.chunking_strategy.fixed_chunking(
                    answer, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP
                )
                # Add question context to each chunk
                for chunk in chunks:
                    chunk["question"] = question
                    chunk["text"] = f"Q: {question}\nA: {chunk['text']}"
            
            elif chunking_method == "semantic":
                chunks = self.chunking_strategy.semantic_chunking(answer, None)
                for chunk in chunks:
                    chunk["question"] = question
                    chunk["text"] = f"Q: {question}\nA: {chunk['text']}"
            
            elif chunking_method == "structural":
                chunks = self.chunking_strategy.structural_chunking(answer, question)
            
            all_chunks.extend(chunks)
        
        print(f"Generated {len(all_chunks)} chunks")
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in all_chunks]
        print("Generating embeddings...")
        embeddings = self.embedding_provider.embed_texts(texts)
        
        # Store in vector database
        print("Storing in vector database...")
        self.vector_store.add_documents(all_chunks, embeddings)
        
        print(f"Successfully ingested {len(all_chunks)} chunks into vector store")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using retrieved documents"""
        context = "\n\n".join([
            f"[Source {i+1}]: {doc['text']}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        system_prompt = """You are a helpful JioPay customer support assistant. Answer questions based ONLY on the provided context. 
        If the information is not in the context, say "I don't have information about that in my knowledge base."
        Always cite your sources using [Source X] format.
        Be concise and helpful."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        start_time = time.time()
        
        try:
            response = openai.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
        except Exception as e:
            answer = f"I apologize, but I encountered an error: {str(e)}"
            tokens_used = 0
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "answer": answer,
            "latency_ms": latency,
            "tokens_used": tokens_used,
            "sources": retrieved_docs
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """End-to-end query processing"""
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question)
        
        # Generate answer
        result = self.generate_answer(question, retrieved_docs)
        
        total_latency = (time.time() - start_time) * 1000
        
        # Track performance
        self.query_stats.append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": total_latency,
            "tokens_used": result["tokens_used"],
            "num_retrieved": len(retrieved_docs)
        })
        
        result["total_latency_ms"] = total_latency
        
        return result

# Flask API
app = Flask(__name__)
CORS(app)

# Global RAG system instance
config = Config()
rag_system = None

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/ingest", methods=["POST"])
def ingest_data():
    """Ingest data from JSON file"""
    global rag_system
    
    try:
        data = request.get_json()
        chunking_method = data.get("chunking_method", "structural")
        data_file = data.get("data_file", config.DATA_FILE)
        
        rag_system = RAGSystem(config)
        rag_system.ingest_data(data_file, chunking_method)
        
        return jsonify({
            "status": "success",
            "message": "Data ingested successfully",
            "chunking_method": chunking_method
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/query", methods=["POST"])
def query_endpoint():
    """Query the RAG system"""
    global rag_system
    
    if rag_system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized. Please ingest data first."
        }), 400
    
    try:
        data = request.get_json()
        question = data.get("question", "")
        
        if not question:
            return jsonify({
                "status": "error",
                "message": "Question is required"
            }), 400
        
        result = rag_system.query(question)
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": result["answer"],
            "latency_ms": result["total_latency_ms"],
            "tokens_used": result["tokens_used"],
            "sources": [
                {
                    "text": source["text"][:200] + "...",  # Truncate for API response
                    "score": source["score"],
                    "source": source["source"]
                }
                for source in result["sources"]
            ]
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system performance statistics"""
    global rag_system
    
    if rag_system is None or not rag_system.query_stats:
        return jsonify({
            "total_queries": 0,
            "avg_latency_ms": 0,
            "avg_tokens_used": 0
        })
    
    stats = rag_system.query_stats
    return jsonify({
        "total_queries": len(stats),
        "avg_latency_ms": sum(s["latency_ms"] for s in stats) / len(stats),
        "avg_tokens_used": sum(s["tokens_used"] for s in stats) / len(stats),
        "recent_queries": stats[-5:]  # Last 5 queries
    })

if __name__ == "__main__":
    # Initialize system on startup if data file exists
    if os.path.exists(config.DATA_FILE):
        print("Initializing RAG system...")
        rag_system = RAGSystem(config)
        rag_system.ingest_data(config.DATA_FILE)
        print("RAG system ready!")
    
    app.run(debug=True, host="0.0.0.0", port=5000)