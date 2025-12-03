import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer - Streamlit Cloud compatible"""
    try:
        # Use lightweight model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"❌ Error loading Sentence Transformer: {e}")
        return None

@st.cache_resource
def load_faiss_index():
    """Load FAISS index"""
    try:
        if os.path.exists("faiss.index"):
            index = faiss.read_index("faiss.index")
            return index
        else:
            st.error("❌ FAISS index file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")
        return None

@st.cache_resource
def load_documents():
    """Load documents metadata"""
    try:
        if os.path.exists("documents.pkl"):
            with open("documents.pkl", "rb") as f:
                documents = pickle.load(f)
            return documents
        else:
            st.error("❌ Documents file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

def retrieve_documents(query, model, index, documents, top_k=5):
    """Retrieve relevant documents"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        scores, indices = index.search(query_embedding, top_k * 2)
        
        # Remove duplicates
        results = []
        seen_sources = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(documents):
                source = documents[idx]["source"]
                
                if source not in seen_sources:
                    seen_sources.add(source)
                    results.append({
                        "score": float(score),
                        "source": source,
                        "text": documents[idx]["text"],
                        "filename": os.path.basename(source)
                    })
                
                if len(results) >= top_k:
                    break
        
        return results
    except Exception as e:
        st.error(f"Error in retrieval: {e}")
        return []

def main():
    """Main Streamlit application - Document Retrieval Only"""
    
    st.markdown('<div class="main-header">🏥 Clinical Document Retrieval System</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Search for Clinical Documentation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        
        st.header("💡 Sample Clinical Queries")
        sample_queries = [
            "stroke symptoms and diagnosis",
            "patient with facial droop and slurred speech",
            "hypertension treatment guidelines", 
            "diabetes management complications"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.query = query
    
    # Load models
    model = load_sentence_transformer()
    index = load_faiss_index()
    documents = load_documents()
    
    if not all([model, index, documents]):
        st.error("❌ System initialization failed.")
        return
    
    # Query input
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_input(
        "Enter clinical question:",
        value=st.session_state.query,
        placeholder="e.g., stroke symptoms, diabetes management..."
    )
    
    if st.button("🚀 Search Clinical Database", type="primary"):
        if query:
            with st.spinner("🔍 Searching clinical documents..."):
                retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
                
                if retrieved_docs:
                    # Display results
                    st.markdown("### 📋 Search Results")
                    st.markdown(f"**Found {len(retrieved_docs)} relevant documents**")
                    
                    for i, doc in enumerate(retrieved_docs):
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
                            with st.expander("View Clinical Content"):
                                doc_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                                st.text(doc_text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No relevant documents found.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Clinical Document Retrieval System | Powered by FAISS</p>
        <p>⚠️ For educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
