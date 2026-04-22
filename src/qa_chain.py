import ollama
from typing import List, Dict, Any
import streamlit as st
from config import LLM_MODEL

class QAChain:
    def __init__(self):
        self.model = LLM_MODEL
        # Use the client method since our debug showed it works
        self.client = ollama.Client(host='http://localhost:11434')
    
    def create_prompt(self, question: str, context_chunks: List[Dict[str, Any]], 
                     conversation_context: str = "") -> str:
        """Create an enhanced prompt with context and conversation history"""
        
        # Build context from retrieved chunks with enhanced metadata
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk["content"]
            metadata = chunk["metadata"]
            filename = metadata.get("filename", "Unknown")
            section = metadata.get("section", "unknown")
            doc_type = metadata.get("doc_type", "general")
            search_type = chunk.get("search_type", "unknown")
            
            # Add search method info
            search_info = ""
            if search_type == "hybrid":
                semantic_score = chunk.get('semantic_score', 0)
                keyword_score = chunk.get('keyword_score', 0)
                search_info = f" (Hybrid: {semantic_score:.2f}S + {keyword_score:.2f}K)"
            elif search_type == "semantic":
                search_info = " (Semantic search)"
            elif search_type == "keyword":
                search_info = " (Keyword search)"
            
            context_parts.append(f"""
--- Source {i}: {filename} | {section} section | {doc_type} document{search_info} ---
{content}
""")
            sources.append(f"Source {i}: {filename} ({section})")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt with conversation awareness
        prompt = f"""You are an intelligent assistant answering questions based ONLY on the provided documents.

INSTRUCTIONS:
1. Give a clear, structured, and concise answer
2. Do NOT repeat the question
3. Do NOT use generic phrases like "based on the provided context"
4. Answer in this format:

Definition:
- Briefly define the concept

Explanation:
- Explain clearly using the document

Example (if available):
- Include real example from document

Conclusion:
- 1–2 line final insight

5. Use simple and clear language
6. Cite sources like (Source 1), (Source 2)
7. If information is missing, say: "Not enough information in the document"
8. Use exact ideas and wording from the document wherever possible
9. Prefer short, sharp sentences instead of long paragraphs
10. Do not add any information that is not present in the document
11. If unsure, copy or closely paraphrase the exact text from the document


{f"PREVIOUS CONTEXT:\n{conversation_context}" if conversation_context else ""}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]], 
                       conversation_context: str = "", question_type: str = "FACT") -> Dict[str, Any]:
        """Generate answer using specialized NLP strategies based on question type"""
        try:
            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information in your documents. Please ensure they are uploaded.",
                    "sources": [],
                    "search_info": {"total_chunks": 0, "search_methods": []},
                    "error": None
                }
            
            # 1. Prepare specialized instructions based on routing
            strategy_prompt = ""
            if question_type == "DEFINITION":
                strategy_prompt = "ACT AS A PRECISION EXTRACTOR. Extract exactly 1-2 sentences defining the concept from the context. Do not add external info."
            elif question_type == "THEME":
                strategy_prompt = "ACT AS A SUMMARIZER. Provide a concise 2-3 sentence summary of the main themes found in the context."
            elif question_type == "REASONING":
                strategy_prompt = "ACT AS AN ANALYST. Explain the logic or process described in the context clearly and concisely."
            else: # FACT
                strategy_prompt = "ACT AS A FACT-FINDER. Give a direct 1-2 sentence answer to the specific question based on the context."

            # 2. Build context
            context_text = "\n".join([f"--- Source {i+1} ---\n{c['content']}" for i, c in enumerate(context_chunks)])
            
            # 3. Create dynamic prompt
            final_prompt = f"""SYSTEM INSTRUCTIONS: {strategy_prompt}
            
CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {question}

FINAL ANSWER (Concise, accurate, 3 sentences max):"""
            
            # 4. Generate response
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': final_prompt}],
                options={'temperature': 0.1, 'top_p': 0.9}
            )
            
            answer = response['message']['content'].strip()
            
            # 5. Format sources and calculate search info
            sources = []
            for chunk in context_chunks:
                metadata = chunk["metadata"]
                sources.append({
                    "filename": metadata.get("filename", "Unknown"),
                    "section": metadata.get("section", "unknown"),
                    "search_type": chunk.get("search_type", "unknown")
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "search_info": {
                    "total_chunks": len(context_chunks),
                    "question_type": question_type
                },
                "error": None
            }
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {"answer": "Error generating answer.", "sources": [], "search_info": {}, "error": str(e)}
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is working"""
        try:
            # Test with the client method that we know works
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Hello, this is a test. Please respond with "Test successful".'
                    }
                ]
            )
            return True
        except Exception as e:
            st.error(f"Ollama test failed: {str(e)}")
            return False