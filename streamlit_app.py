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
    .model-status {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load YOUR custom sentence transformer from 1_Pooling folder"""
    try:
        st.info("🔄 Loading YOUR trained model...")
        
        # Check if we should use 1_Pooling folder
        if os.path.exists("1_Pooling"):
            model = SentenceTransformer('1_Pooling', local_files_only=True)
            st.success("✅ Loaded from 1_Pooling folder!")
        else:
            # Try current directory
            model = SentenceTransformer('.', local_files_only=True)
            st.success("✅ Loaded from current directory!")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOUR model: {e}")
        # List available files for debugging
        st.write("Available model files:", [f for f in os.listdir('.') if 'json' in f or 'safetensors' in f])
        return None

@st.cache_resource
def load_faiss_index():
    """Load YOUR FAISS index"""
    try:
        if os.path.exists("faiss.index"):
            st.info("🔄 Loading FAISS index...")
            index = faiss.read_index("faiss.index")
            st.success("✅ FAISS index loaded!")
            return index
        else:
            st.error("❌ faiss.index file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading FAISS: {e}")
        return None

@st.cache_resource
def load_documents():
    """Load YOUR documents"""
    try:
        if os.path.exists("documents.pkl"):
            st.info("🔄 Loading clinical documents...")
            with open("documents.pkl", "rb") as f:
                documents = pickle.load(f)
            st.success(f"✅ Loaded {len(documents)} clinical documents!")
            return documents
        else:
            st.error("❌ documents.pkl file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

def retrieve_documents(query, model, index, documents, top_k=5):
    """YOUR retrieval function - exact same as Colab"""
    try:
        # Encode query with YOUR model
        q = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        
        # Search in YOUR FAISS index
        scores, idx = index.search(q, top_k * 2)
        
        # Remove duplicates
        out = []
        seen_sources = set()
        
        for score, i in zip(scores[0], idx[0]):
            if i < len(documents):
                source = documents[i].get("source", "")
                
                # Skip duplicates
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                out.append({
                    "score": float(score),
                    "source": source,
                    "text": documents[i].get("text", ""),
                    "filename": os.path.basename(source) if source else f"doc_{i}"
                })
                
                if len(out) >= top_k:
                    break
        
        return out
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def create_clinical_summary(query, retrieved_docs):
    """Create a clinical summary from retrieved documents"""
    if not retrieved_docs:
        return "No relevant clinical documents found."
    
    summary = []
    summary.append(f"**Clinical Query**: {query}")
    summary.append(f"**Documents Retrieved**: {len(retrieved_docs)} relevant clinical notes")
    
    # Calculate average relevance
    avg_score = np.mean([doc['score'] for doc in retrieved_docs])
    summary.append(f"**Average Relevance Score**: {avg_score:.3f}")
    
    # High relevance documents
    high_rel = [d for d in retrieved_docs if d['score'] > 0.6]
    if high_rel:
        summary.append(f"**High-Relevance Documents**: {len(high_rel)} with score > 0.6")
    
    # Extract clinical themes
    clinical_keywords = {
        'stroke': 'Cerebrovascular',
        'diagnosis': 'Diagnostic',
        'treatment': 'Therapeutic',
        'symptom': 'Symptomatic',
        'patient': 'Clinical Case',
        'history': 'Patient History',
        'examination': 'Clinical Exam'
    }
    
    themes_found = set()
    for doc in retrieved_docs[:3]:  # Check first 3 docs
        text_lower = doc['text'].lower()
        for keyword, theme in clinical_keywords.items():
            if keyword in text_lower:
                themes_found.add(theme)
    
    if themes_found:
        summary.append(f"**Clinical Themes**: {', '.join(themes_found)}")
    
    summary.append("\n**Review the specific clinical documents below for detailed findings.**")
    
    return "\n\n".join(summary)

def main():
    """Main Streamlit application - Using YOUR RAG system"""
    
    st.markdown('<div class="main-header">🏥 YOUR Clinical RAG System</div>', unsafe_allow_html=True)
    st.markdown("### Using Your Trained Model + FAISS + Clinical Documents")
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Load models
        with st.spinner("Loading your RAG system..."):
            model = load_sentence_transformer()
            index = load_faiss_index()
            documents = load_documents()
        
        # Status display
        st.markdown("### ✅ Components Loaded:")
        status_items = [
            ("Your Sentence Transformer", model is not None),
            ("FAISS Index", index is not None),
            ("Clinical Documents", documents is not None and len(documents) > 0)
        ]
        
        for name, status in status_items:
            icon = "✅" if status else "❌"
            st.write(f"{icon} {name}")
        
        if documents:
            st.metric("Total Documents", len(documents))
        
        st.header("⚙️ Configuration")
        top_k = st.slider("Documents to retrieve", 1, 10, 5)
        
        st.header("💡 Sample Queries")
        sample_queries = [
            "stroke symptoms diagnosis",
            "facial droop slurred speech",
            "hypertension treatment",
            "neurological examination findings"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", use_container_width=True):
                st.session_state.query = query
                st.rerun()
    
    # Check if all components loaded
    if not all([model, index, documents]):
        st.error("""
        ❌ System initialization failed. Required:
        1. Your sentence transformer model (1_Pooling folder or local files)
        2. faiss.index file
        3. documents.pkl file
        """)
        return
    
    # Main query interface
    st.markdown("---")
    st.markdown("### 🔍 Clinical Query")
    
    query = st.text_input(
        "Enter clinical question:",
        value=st.session_state.query,
        placeholder="e.g., What are the symptoms of stroke?",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_btn = st.button("🚀 Search Clinical Database", type="primary", use_container_width=True)
    
    # Process query
    if search_btn and query:
        with st.spinner(f"🔍 Searching {len(documents)} clinical documents..."):
            retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
            
            if retrieved_docs:
                # Store in session state
                st.session_state.retrieved_docs = retrieved_docs
                st.session_state.current_query = query
                
                # Create summary
                summary = create_clinical_summary(query, retrieved_docs)
                st.session_state.summary = summary
                
                # Force display
                st.rerun()
            else:
                st.error("No relevant documents found. Try rephrasing your query.")
    
    # Display results if available
    if hasattr(st.session_state, 'retrieved_docs') and st.session_state.retrieved_docs:
        st.markdown("---")
        st.markdown("### 📋 Clinical Summary")
        st.info(st.session_state.summary)
        
        st.markdown("---")
        st.markdown(f"### 📚 Retrieved Documents ({len(st.session_state.retrieved_docs)})")
        
        for i, doc in enumerate(st.session_state.retrieved_docs):
            with st.container():
                st.markdown('<div class="document-box">', unsafe_allow_html=True)
                
                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Document {i+1}:** `{doc['filename']}`")
                with col2:
                    score = doc['score']
                    if score > 0.7:
                        st.markdown(f'<span class="score-high">Score: {score:.3f}</span>', unsafe_allow_html=True)
                    elif score > 0.5:
                        st.markdown(f'<span class="score-medium">Score: {score:.3f}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="score-low">Score: {score:.3f}</span>', unsafe_allow_html=True)
                
                # Document content
                with st.expander("View Clinical Notes"):
                    # Clean and display text
                    doc_text = doc['text'].strip()
                    # Remove excessive whitespace
                    doc_text = ' '.join(doc_text.split())
                    
                    if len(doc_text) > 2000:
                        st.text_area(
                            f"Full Text {i+1}",
                            doc_text[:2000] + "...\n\n[Document truncated - original length: " + str(len(doc_text)) + " chars]",
                            height=300,
                            key=f"doc_text_{i}"
                        )
                    else:
                        st.text_area(
                            f"Full Text {i+1}",
                            doc_text,
                            height=min(400, max(200, len(doc_text) // 4)),
                            key=f"doc_text_{i}"
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding-top: 2rem;'>
        <p><strong>🏥 Your Clinical RAG System</strong></p>
        <p>Embedding: Your trained model | Retrieval: FAISS | Documents: {doc_count}</p>
        <p><em>For educational and research purposes only</em></p>
    </div>
    """.format(doc_count=len(documents) if documents else "Unknown"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
