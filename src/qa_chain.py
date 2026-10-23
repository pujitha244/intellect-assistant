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
                       conversation_context: str = "", style: str = "Intermediate") -> Dict[str, Any]:
        """Generate answer using direct context-aware retrieval and dynamic explanation styles"""
        try:
            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information in your documents. Please ensure they are uploaded.",
                    "sources": [],
                    "search_info": {"total_chunks": 0},
                    "error": None
                }
            
            # 1. Build context
            context_text = "\n".join([f"--- Source {i+1} ---\n{c['content']}" for i, c in enumerate(context_chunks)])
            
            # 2. Determine Style
            style_instruction = ""
            if style == "Beginner":
                style_instruction = "Explain this like I am a beginner. Use very simple language, everyday analogies, and avoid complex jargon."
            elif style == "Technical":
                style_instruction = "Provide a highly technical and detailed explanation. Focus on mechanics, architecture, and industry-standard terminology."
            elif style == "Exam-ready":
                style_instruction = "Format perfectly for an exam. Use bullet points, bold key terms, and provide a crisp summary statement at the end."
            else:
                style_instruction = "Provide a clear, balanced explanation suitable for a professional audience. Be concise."

            # 3. Create direct prompt
            final_prompt = f"""You are an intelligent AI assistant. Use the following context to answer the user's question accurately.
            
CONTEXT DOCUMENTS:
{context_text}

USER QUESTION: {question}

EXPLANATION STYLE: {style_instruction}

FINAL ANSWER (Cite sources if applicable, adhere STRICTLY to the explanation style):"""
            
            # 3. Generate response
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': final_prompt}],
                options={'temperature': 0.1, 'top_p': 0.9}
            )
            
            answer = response['message']['content'].strip()
            
            # 4. Format sources
            sources = []
            for chunk in context_chunks:
                metadata = chunk["metadata"]
                sources.append({
                    "filename": metadata.get("filename", "Unknown"),
                    "section": metadata.get("section", "unknown"),
                    "search_type": chunk.get("search_type", "unknown"),
                    "score": chunk.get("score") or chunk.get("hybrid_score"),
                    "distance": chunk.get("distance")
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "search_info": {
                    "total_chunks": len(context_chunks)
                },
                "error": None
            }
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return {"answer": "Error generating answer.", "sources": [], "search_info": {}, "error": str(e)}

    def compare_documents(self, doc1_name: str, doc1_chunks: List[Dict], doc2_name: str, doc2_chunks: List[Dict]) -> str:
        """Compare two documents and highlight differences"""
        try:
            # Limit text to avoid context overflow
            def _get_text(chunks):
                text = "\n".join([c['content'] for c in chunks])
                return text[:5000] # roughly 1000-1500 tokens
            
            doc1_text = _get_text(doc1_chunks)
            doc2_text = _get_text(doc2_chunks)
            
            prompt = f"""You are an expert legal and academic analyst. Compare the following two documents and provide a structured comparison.
Highlight contradictions, missing information in one vs the other, and summarize the key differences.

DOCUMENT 1: {doc1_name}
{doc1_text}

DOCUMENT 2: {doc2_name}
{doc2_text}

Provide your analysis strictly in this format:
### 1. Key Similarities
[List similarities]

### 2. Key Differences
[List differences]

### 3. Contradictions or Missing Clauses
[Highlight missing info]

### 4. Executive Summary
[Brief summary]
"""
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error during comparison: {str(e)}"

    def generate_study_material(self, doc_name: str, chunks: List[Dict], mode: str = "quiz") -> str:
        """Generate quiz or flashcards from a document"""
        try:
            text_content = "\n".join([c['content'] for c in chunks])[:8000]
            
            if mode == "quiz":
                prompt = f"""You are an expert teacher. Write 5 multiple-choice questions based ONLY on the text below.
            
TEXT:
{text_content}

For each question, provide 4 realistic options (A, B, C, D), identify the correct answer, and give a short explanation. 
Do not use placeholders, you must write the actual questions and answers!

Example Format:
Q1. What is the capital of France?
A) Berlin
B) Madrid
C) Paris
D) Rome
Correct Answer: C
Explanation: Paris is the capital of France.
"""
            else: # flashcards
                prompt = f"""You are an expert educator. Create exactly 10 study flashcards based ONLY on the text below.
            
TEXT:
{text_content}

You must output ONLY a valid CSV format. Put the question on the left of the comma, and the answer on the right. 
Do not use placeholders like "Question 1". Write the actual facts!

Example Format:
What is the capital of France?,Paris
Who wrote Hamlet?,William Shakespeare
"""
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating study material: {str(e)}"
            
    def analyze_learning_profile(self, history: List[Dict]) -> str:
        """Analyze chat history to detect weak areas and build a revision plan"""
        try:
            if not history:
                return "Not enough conversation history to analyze learning patterns."
                
            questions = [ex['question'] for ex in history[-10:]]
            q_text = "\n".join([f"- {q}" for q in questions])
            
            prompt = f"""You are an expert AI Tutor analyzing a student's question history.
Based on the following recent questions asked by the student:
{q_text}

1. Identify the Main Subject(s) they are studying.
2. Identify potential "Weak Areas" (topics they ask repeatedly about or seem confused about).
3. Generate a quick 3-step Revision Plan for them.

Format your response exactly like this:
### 📊 Learning Profile
**Main Subjects:** [Subjects]
**Detected Weak Areas:** [Weak areas]

### 📝 Recommended Revision Plan
1. [Step 1]
2. [Step 2]
3. [Step 3]
"""
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error analyzing profile: {str(e)}"
    
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