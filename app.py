import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

# Page configuration
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .clinical-answer {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .document-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all required models and data"""
    try:
        # Load sentence transformer model
        st.info("🔄 Loading Sentence Transformer model...")
        model = SentenceTransformer('sentence_transformer_model')
        
        # Load FAISS index
        st.info("🔄 Loading FAISS index...")
        index = faiss.read_index("faiss.index")
        
        # Load documents metadata
        st.info("🔄 Loading documents metadata...")
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        
        # Load generator model
        st.info("🔄 Loading Qwen language model...")
        tokenizer = AutoTokenizer.from_pretrained(".")
        generator = AutoModelForCausalLM.from_pretrained(
            ".",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        st.success("✅ All models loaded successfully!")
        return model, index, documents, tokenizer, generator
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None

def retrieve_documents(query, model, index, documents, top_k=5):
    """Retrieve relevant documents for a query"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        scores, indices = index.search(query_embedding, top_k * 2)  # Get more to filter duplicates
        
        # Remove duplicates
        results = []
        seen_sources = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(documents):  # Safety check
                doc = documents[idx]
                source = doc.get("source", "")
                
                if source not in seen_sources:
                    seen_sources.add(source)
                    results.append({
                        "score": float(score),
                        "source": source,
                        "text": doc.get("text", ""),
                        "filename": os.path.basename(source) if source else "Unknown"
                    })
                
                if len(results) >= top_k:
                    break
        
        return results
    except Exception as e:
        st.error(f"Error in retrieval: {e}")
        return []

def generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens=400):
    """Generate answer based on retrieved documents"""
    try:
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 documents
            doc_text = doc["text"].strip()
            
            # Clean and truncate text
            if len(doc_text) > 800:
                trunc_point = doc_text[:800].rfind('.')
                if trunc_point > 400:
                    doc_text = doc_text[:trunc_point + 1]
                else:
                    doc_text = doc_text[:800] + "..."
            
            context_parts.append(f"[Document {i+1}]: {doc_text}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following clinical documentation, provide a concise medical assessment.

CLINICAL CONTEXT:
{context}

CLINICAL QUESTION: {query}

MEDICAL ASSESSMENT:"""
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(generator.device)
        
        with torch.no_grad():
            outputs = generator.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode and extract answer
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "MEDICAL ASSESSMENT:" in full_output:
            answer = full_output.split("MEDICAL ASSESSMENT:")[-1].strip()
        else:
            answer = full_output
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Clinical RAG System</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Clinical Document Retrieval and Assessment")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        max_tokens = st.slider("Maximum answer length", 100, 800, 400)
        
        st.header("💡 Sample Queries")
        sample_queries = [
            "What is the likely diagnosis for a patient with slurred speech and facial droop?",
            "What are the key clinical findings in this stroke case?",
            "What diagnostic tests should be performed for suspected stroke?",
            "Patient with sudden severe headache and vomiting — possible diagnosis?",
            "Differentiate between ischemic and hemorrhagic stroke symptoms"
        ]
        
        for query in sample_queries:
            if st.button(f"🗨️ {query[:50]}..."):
                st.session_state.query = query
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Load models
    model, index, documents, tokenizer, generator = load_models()
    
    if model is None or index is None or documents is None:
        st.error("Failed to load required models. Please check if all model files are available.")
        return
    
    # Main query interface
    st.markdown("---")
    st.markdown("### 🔍 Enter Clinical Query")
    
    query = st.text_area(
        "Describe the clinical scenario or ask a medical question:",
        value=st.session_state.query,
        height=100,
        placeholder="e.g., Patient presents with sudden onset of right-sided weakness and aphasia..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_button = st.button("🚀 Search Clinical Database", use_container_width=True)
    
    if search_button and query:
        with st.spinner("🔍 Searching clinical documents and generating assessment..."):
            # Retrieve documents
            retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
            
            # Generate answer
            answer = generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens=max_tokens)
            
            # Store results in session state
            st.session_state.last_results = {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "answer": answer
            }
    
    # Display results if available
    if hasattr(st.session_state, 'last_results'):
        results = st.session_state.last_results
        
        st.markdown("---")
        st.markdown("### 💡 Clinical Assessment")
        
        # Display answer
        with st.container():
            st.markdown('<div class="clinical-answer">', unsafe_allow_html=True)
            st.markdown("**Generated Assessment:**")
            st.write(results["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display retrieved documents
        st.markdown("---")
        st.markdown(f"### 📚 Retrieved Clinical Documents ({len(results['retrieved_docs'])})")
        
        for i, doc in enumerate(results['retrieved_docs']):
            with st.container():
                st.markdown('<div class="document-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Document {i+1}:** {doc['filename']}")
                with col2:
                    score = doc['score']
                    if score > 0.6:
                        st.markdown(f'<span class="score-high">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                    elif score > 0.4:
                        st.markdown(f'<span class="score-medium">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="score-low">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                
                # Document preview
                with st.expander("View Document Content"):
                    doc_text = doc['text'].strip()
                    if len(doc_text) > 1000:
                        st.text(doc_text[:1000] + "...")
                    else:
                        st.text(doc_text)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Clinical RAG System | Powered by FAISS + Sentence Transformers + Qwen</p>
        <p>⚠️ For educational and research purposes only. Not for clinical decision making.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
