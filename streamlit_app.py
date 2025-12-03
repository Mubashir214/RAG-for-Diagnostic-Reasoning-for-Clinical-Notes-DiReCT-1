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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .document-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer from CURRENT DIRECTORY files"""
    try:
        # Check if we have all required files in current directory
        required_files = [
            'sentence_bert_config.json',
            'config_sentence_transformers.json', 
            'model.safetensors',
            'modules.json'
        ]
        
        available_files = [f for f in required_files if os.path.exists(f)]
        
        if len(available_files) >= 3:  # Need at least 3 of the 4
            st.info("🔄 Loading sentence transformer from local files...")
            
            # Create a temporary config if needed
            if not os.path.exists('config.json'):
                # Create minimal config
                config = {
                    "_name_or_path": ".",
                    "architectures": ["BertModel"],
                    "model_type": "bert"
                }
                import json
                with open('config.json', 'w') as f:
                    json.dump(config, f)
            
            # Try to load from current directory
            try:
                model = SentenceTransformer('.', local_files_only=True)
                st.success("✅ Loaded from local files!")
                return model
            except Exception as e1:
                st.warning(f"Local load failed: {e1}")
                
                # Fallback to lightweight model
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Using fallback model (all-MiniLM-L6-v2)")
                return model
        else:
            # Use lightweight model
            st.info("🔄 Using lightweight model (all-MiniLM-L6-v2)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Lightweight model loaded!")
            return model
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

@st.cache_resource
def load_faiss_index():
    """Load FAISS index"""
    try:
        if os.path.exists("faiss.index"):
            index = faiss.read_index("faiss.index")
            return index
        else:
            st.error("❌ faiss.index file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading FAISS: {e}")
        return None

@st.cache_resource
def load_documents():
    """Load documents"""
    try:
        if os.path.exists("documents.pkl"):
            with open("documents.pkl", "rb") as f:
                documents = pickle.load(f)
            return documents
        else:
            st.error("❌ documents.pkl file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

def retrieve_documents(query, model, index, documents, top_k=5):
    """Retrieve documents"""
    try:
        # Encode query
        q = model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        scores, idx = index.search(q, top_k * 2)
        
        # Remove duplicates
        results = []
        seen_sources = set()
        
        for score, i in zip(scores[0], idx[0]):
            if i < len(documents):
                source = documents[i].get("source", "")
                
                if source not in seen_sources:
                    seen_sources.add(source)
                    results.append({
                        "score": float(score),
                        "source": source,
                        "text": documents[i].get("text", ""),
                        "filename": os.path.basename(source) if source else f"doc_{i}"
                    })
                
                if len(results) >= top_k:
                    break
        
        return results
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def main():
    """Main Streamlit application"""
    
    st.markdown('<div class="main-header">🏥 Clinical RAG System</div>', unsafe_allow_html=True)
    st.markdown("### Document Retrieval System")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Load components
        model = load_sentence_transformer()
        index = load_faiss_index()
        documents = load_documents()
        
        # Display status
        if model:
            st.success("✅ Embedding Model: Loaded")
        else:
            st.error("❌ Embedding Model: Failed")
            
        if index:
            st.success("✅ FAISS Index: Loaded")
        else:
            st.error("❌ FAISS Index: Failed")
            
        if documents:
            st.success(f"✅ Documents: {len(documents)} loaded")
        else:
            st.error("❌ Documents: Failed")
        
        if documents:
            st.metric("Total Documents", len(documents))
        
        st.header("⚙️ Configuration")
        top_k = st.slider("Documents to retrieve", 1, 10, 5)
        
        st.header("💡 Sample Queries")
        sample_queries = [
            "stroke symptoms",
            "facial droop",
            "hypertension",
            "neurological exam"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.query = query
    
    # Check if components loaded
    if not all([model, index, documents]):
        st.warning("⚠️ Some components failed to load, but retrieval may still work.")
    
    # Main interface
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_input(
        "Enter clinical query:",
        value=st.session_state.query,
        placeholder="e.g., stroke symptoms, facial droop..."
    )
    
    if st.button("🚀 Search Clinical Database", type="primary"):
        if query and model and index and documents:
            with st.spinner("Searching..."):
                results = retrieve_documents(query, model, index, documents, top_k)
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant documents")
                    
                    for i, doc in enumerate(results):
                        with st.container():
                            st.markdown('<div class="document-box">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Document {i+1}:** {doc['filename']}")
                            with col2:
                                score = doc['score']
                                if score > 0.6:
                                    st.markdown(f'<span class="score-high">{score:.3f}</span>', unsafe_allow_html=True)
                                elif score > 0.4:
                                    st.markdown(f'<span class="score-medium">{score:.3f}</span>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<span class="score-low">{score:.3f}</span>', unsafe_allow_html=True)
                            
                            with st.expander("View Content"):
                                st.text(doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text'])
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No documents found")
        elif not query:
            st.error("Please enter a query")
        else:
            st.error("System not fully loaded")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Clinical Document Retrieval System</p>
        <p>Powered by FAISS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
