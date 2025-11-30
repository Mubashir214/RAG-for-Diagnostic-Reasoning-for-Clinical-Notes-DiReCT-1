# app.py - UPDATED with correct file names
import streamlit as st
import os
import sys
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Clinical RAG Assistant",
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
    .clinical-query {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models with your actual file names"""
    try:
        # Check for your actual file names
        required_files = {
            'faiss.index': 'FAISS vector database',
            'documents.pkl': 'Clinical documents data'
        }
        
        # Verify files exist
        for file, description in required_files.items():
            if not os.path.exists(file):
                st.error(f"❌ Missing file: {file} - {description}")
                return None, None, None
        
        # Setup sentence_model folder structure
        if not os.path.exists('sentence_model'):
            st.info("🔧 Setting up model folder structure...")
            os.makedirs('sentence_model', exist_ok=True)
            os.makedirs('sentence_model/1_Pooling', exist_ok=True)
            
            # Move files to sentence_model folder
            files_to_move = [
                'config.json', 'model.safetensors', 'tokenizer_config.json',
                'tokenizer.json', 'vocab.txt', 'special_tokens_map.json',
                'sentence_bert_config.json', 'config_sentence_transformers.json',
                'modules.json', 'README.md'
            ]
            
            for file in files_to_move:
                if os.path.exists(file):
                    os.rename(file, f'sentence_model/{file}')
            
            # Handle pooling config
            if os.path.exists('1_Pooling/config.json'):
                os.rename('1_Pooling/config.json', 'sentence_model/1_Pooling/config.json')
        
        # Import required libraries
        import faiss
        import pickle
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        # Load models with YOUR file names
        st.info("🔄 Loading sentence transformer model...")
        embedding_model = SentenceTransformer('sentence_model')
        
        st.info("📊 Loading FAISS index...")
        faiss_index = faiss.read_index('faiss.index')  # YOUR FILE NAME
        
        st.info("📚 Loading clinical documents...")
        with open('documents.pkl', 'rb') as f:  # YOUR FILE NAME
            documents_data = pickle.load(f)
        
        return embedding_model, faiss_index, documents_data
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None

def retrieve_documents(query, top_k=5):
    """Enhanced retrieval function"""
    if 'faiss_index' not in st.session_state:
        return []
    
    # Encode query
    q = st.session_state.embedding_model.encode([query], convert_to_numpy=True)
    
    # Search in FAISS
    scores, idx = st.session_state.faiss_index.search(q, top_k * 2)
    
    out = []
    seen_sources = set()
    
    for score, i in zip(scores[0], idx[0]):
        if i >= len(st.session_state.documents_data):
            continue
            
        doc = st.session_state.documents_data[i]
        source = doc.get("source", f"doc_{i}")
        
        # Skip duplicates
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        # Medical relevance boosting
        adjusted_score = float(score)
        query_lower = query.lower()
        doc_text_lower = doc["text"].lower() if "text" in doc else ""
        
        # Boost for medical terms
        medical_terms = ['stroke', 'ischemic', 'hemorrhagic', 'heart', 'cardiac', 'infection']
        if any(term in query_lower for term in medical_terms):
            if any(term in doc_text_lower for term in medical_terms):
                adjusted_score += 0.1
        
        out.append({
            "score": adjusted_score,
            "source": source,
            "text": doc["text"] if "text" in doc else "No text content",
            "filename": os.path.basename(source) if "source" in doc else f"Document_{i}",
            "original_score": float(score)
        })
        
        if len(out) >= top_k:
            break
    
    # Sort by score
    out.sort(key=lambda x: x['score'], reverse=True)
    return out

def analyze_clinical_context(query, retrieved_docs):
    """Generate intelligent analysis based on retrieved documents"""
    if not retrieved_docs:
        return "No relevant clinical documents found for this query."
    
    # Analyze the retrieval results
    total_docs = len(retrieved_docs)
    avg_score = sum(doc['score'] for doc in retrieved_docs) / total_docs
    high_quality_docs = sum(1 for doc in retrieved_docs if doc['score'] > 0.6)
    
    # Extract key themes
    themes = set()
    for doc in retrieved_docs[:3]:
        text = doc['text'].lower()
        if any(term in text for term in ['stroke', 'infarct', 'cerebral']):
            themes.add("cerebrovascular")
        if any(term in text for term in ['headache', 'migraine']):
            themes.add("headache")
        if any(term in text for term in ['chest', 'heart', 'cardiac']):
            themes.add("cardiac")
        if any(term in text for term in ['fever', 'infection', 'sepsis']):
            themes.add("infectious")
        if any(term in text for term in ['weakness', 'paralysis']):
            themes.add("motor deficit")
        if any(term in text for term in ['speech', 'aphasia']):
            themes.add("speech disturbance")
    
    themes_str = ", ".join(themes) if themes else "various clinical"
    
    # Generate analysis
    analysis = f"""
## 🩺 Clinical Context Analysis

*Query:* {query}

*Database Search Results:*
- *Documents Found:* {total_docs} relevant clinical cases
- *Retrieval Quality:* {'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} (average relevance: {avg_score:.3f})
- *High-Quality Matches:* {high_quality_docs} documents with strong relevance

*Clinical Themes Identified:*
The retrieved documents primarily discuss *{themes_str}* conditions.

*Key Insights:*
Based on the clinical documentation, several similar cases provide context for your query. Review the specific documents below for detailed clinical information.

---

*Note:* Always consult healthcare professionals for definitive diagnosis and treatment decisions.
"""
    
    return analysis

def main():
    st.markdown('<h1 class="main-header">🏥 Clinical RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        with st.spinner("🔄 Initializing Clinical RAG System..."):
            embedding_model, faiss_index, documents_data = load_models()
            
            if embedding_model and faiss_index and documents_data:
                st.session_state.embedding_model = embedding_model
                st.session_state.faiss_index = faiss_index
                st.session_state.documents_data = documents_data
                st.session_state.models_loaded = True
                
                st.markdown(f'<div class="success-box">✅ <strong>System Ready!</strong> Loaded {len(documents_data)} clinical documents successfully.</div>', unsafe_allow_html=True)
            else:
                st.error("❌ System initialization failed. Please check that all required files are uploaded.")
                st.info("""
                *Required Files:*
                - faiss.index - FAISS vector database
                - documents.pkl - Clinical documents
                - All sentence_model files (config.json, model.safetensors, etc.)
                """)
                return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙ Configuration")
        top_k = st.slider("Documents to retrieve", 1, 10, 5)
        
        st.header("🚀 Quick Queries")
        quick_queries = [
            "Patient with sudden weakness on one side and facial droop",
            "Patient with severe headache, nausea and vomiting",
            "Patient with chest pain radiating to left arm",
            "Patient with fever and low blood pressure",
            "Patient with speech difficulties and arm weakness"
        ]
        
        for i, query in enumerate(quick_queries):
            if st.button(f"{query[:40]}...", key=f"quick_{i}"):
                st.session_state.current_query = query
                if 'last_results' in st.session_state:
                    del st.session_state.last_results
                st.rerun()
    
    # Main query interface
    st.markdown("### 🔍 Enter Clinical Query")
    
    query = st.text_area(
        "Describe the clinical scenario:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="Example: 65-year-old with acute right-sided weakness and slurred speech"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Search Clinical Database", type="primary", use_container_width=True):
            if query.strip():
                # Retrieve documents
                with st.spinner("🔍 Searching clinical database..."):
                    retrieved_docs = retrieve_documents(query, top_k=top_k)
                
                # Generate analysis
                with st.spinner("📊 Analyzing clinical context..."):
                    analysis = analyze_clinical_context(query, retrieved_docs)
                
                # Store results
                st.session_state.last_results = {
                    'query': query,
                    'retrieved_docs': retrieved_docs,
                    'analysis': analysis
                }
                
                st.rerun()
            else:
                st.warning("⚠ Please enter a clinical query.")
    
    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            if 'last_results' in st.session_state:
                del st.session_state.last_results
            st.session_state.current_query = ""
            st.rerun()
    
    # Display results
    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        
        st.markdown("---")
        st.markdown(results['analysis'])
        
        # Display retrieved documents
        st.markdown("### 📚 Retrieved Clinical Documents")
        st.info(f"Found {len(results['retrieved_docs'])} relevant documents:")
        
        for i, doc in enumerate(results['retrieved_docs']):
            # Determine quality indicator
            quality_color = "🟢" if doc['score'] > 0.7 else "🟡" if doc['score'] > 0.5 else "🟠"
            
            with st.expander(f"{quality_color} Document {i+1} | Relevance: {doc['score']:.3f} | {doc.get('filename', 'Clinical Case')}", expanded=i < 2):
                st.write("*Clinical Content:*")
                st.write(doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Relevance Score", f"{doc['score']:.3f}")
                with col_b:
                    st.metric("Document", f"{i+1}/{len(results['retrieved_docs'])}")
        
        if not results['retrieved_docs']:
            st.warning("No documents met the relevance threshold. Try rephrasing your query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <i>Clinical RAG Assistant v1.0 • For educational and research purposes • Always consult healthcare professionals for medical decisions</i>
        </div>
        """,
        unsafe_allow_html=True
    )

if _name_ == "_main_":
    main()
