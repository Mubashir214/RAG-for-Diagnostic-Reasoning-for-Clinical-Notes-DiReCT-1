import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Page configuration
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .clinical-answer { background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 1rem 0; }
    .document-box { background-color: #f9f9f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #ff6b6b; margin: 0.5rem 0; }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load only essential models for Streamlit Cloud"""
    try:
        # Load lightweight embedding model
        st.info("🔄 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        
        # Load FAISS index
        if os.path.exists("faiss.index"):
            index = faiss.read_index("faiss.index")
        else:
            st.error("❌ FAISS index not found")
            return None, None, None
            
        # Load documents
        if os.path.exists("documents.pkl"):
            with open("documents.pkl", "rb") as f:
                documents = pickle.load(f)
        else:
            st.error("❌ Documents not found")
            return None, None, None
            
        st.success("✅ All models loaded successfully!")
        return model, index, documents
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

def retrieve_documents(query, model, index, documents, top_k=3):
    """Retrieve documents without generation"""
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
        st.error(f"Retrieval error: {e}")
        return []

def main():
    """Main app - Document Retrieval Only"""
    
    st.markdown('<div class="main-header">🏥 Clinical Document Retrieval System</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Search for Clinical Documentation")
    
    # Load models
    model, index, documents = load_models()
    
    if not all([model, index, documents]):
        st.error("System initialization failed. Please check your model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Documents to retrieve", 1, 5, 3)
        
        st.header("💡 Sample Queries")
        sample_queries = [
            "stroke symptoms and diagnosis",
            "patient with facial droop",
            "hypertension treatment",
            "diabetes management"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.query = query
    
    # Query input
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_area(
        "Enter clinical question:",
        value=st.session_state.query,
        height=80,
        placeholder="e.g., stroke symptoms, diabetes management..."
    )
    
    if st.button("🚀 Search Clinical Database", type="primary"):
        if query:
            with st.spinner("🔍 Searching clinical documents..."):
                retrieved_docs = retrieve_documents(query, model, index, documents, top_k)
                
                if retrieved_docs:
                    st.session_state.retrieved_docs = retrieved_docs
                    st.session_state.current_query = query
                    
                    # Show summary
                    st.success(f"✅ Found {len(retrieved_docs)} relevant documents")
                    
                    # Display documents
                    st.markdown("### 📚 Clinical Documents Found")
                    
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
                                doc_text = doc['text'][:1500] + "..." if len(doc['text']) > 1500 else doc['text']
                                st.text(doc_text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No relevant documents found. Try rephrasing your query.")
    
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
