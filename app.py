import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

# Page configuration
st.set_page_config(
    page_title="Clinical RAG System - Document Retrieval",
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
    """Load sentence transformer from local files with fallback"""
    try:
        # Check if local model exists and load it
        if all(os.path.exists(f) for f in ['sentence_bert_config.json', 'config_sentence_transformers.json']):
            st.info("🔄 Loading Sentence Transformer from local files...")
            model = SentenceTransformer('.', local_files_only=True)
            st.success("✅ Local Sentence Transformer loaded successfully!")
            return model
        else:
            # Fallback to lightweight model
            st.info("🔄 Loading lightweight model (all-MiniLM-L6-v2)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Fallback model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading Sentence Transformer: {e}")
        # Final fallback
        try:
            st.info("🔄 Trying universal sentence encoder...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Universal model loaded successfully!")
            return model
        except:
            st.error("❌ All model loading attempts failed")
            return None

@st.cache_resource
def load_faiss_index():
    """Load FAISS index"""
    try:
        if os.path.exists("faiss.index"):
            st.info("🔄 Loading FAISS index...")
            index = faiss.read_index("faiss.index")
            st.success("✅ FAISS index loaded successfully!")
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
            st.info("🔄 Loading documents metadata...")
            with open("documents.pkl", "rb") as f:
                documents = pickle.load(f)
            st.success(f"✅ Loaded {len(documents)} clinical documents!")
            return documents
        else:
            st.error("❌ Documents file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

def retrieve_documents(query, model, index, documents, top_k=5):
    """Retrieve relevant documents for a query"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        
        # Search in FAISS index
        scores, indices = index.search(query_embedding, top_k * 3)  # Get more to filter duplicates
        
        # Remove duplicates and ensure valid indices
        results = []
        seen_sources = set()
        seen_texts = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(documents) and idx >= 0:  # Safety check
                doc = documents[idx]
                source = doc.get("source", "")
                text = doc.get("text", "").strip()
                
                # Skip if text is too short or duplicate
                if len(text) < 50 or text in seen_texts or source in seen_sources:
                    continue
                    
                seen_sources.add(source)
                seen_texts.add(text[:500])  # Store text fingerprint
                
                results.append({
                    "score": float(score),
                    "source": source,
                    "text": text,
                    "filename": os.path.basename(source) if source else "Unknown",
                    "preview": text[:300] + "..." if len(text) > 300 else text
                })
                
                if len(results) >= top_k:
                    break
        
        return results
    except Exception as e:
        st.error(f"Error in retrieval: {e}")
        return []

def generate_insightful_summary(query, retrieved_docs):
    """Generate a smart summary based on retrieved documents without LLM"""
    
    if not retrieved_docs:
        return "No relevant clinical documents found. Try rephrasing your query."
    
    # Extract key information patterns
    all_text = " ".join([doc["text"] for doc in retrieved_docs])
    
    # Simple pattern matching for common clinical concepts
    findings = []
    
    # Look for diagnosis patterns
    diagnosis_patterns = [
        r'diagnos[ie]s?:?\s*([^\.]+)',
        r'findings?:?\s*([^\.]+)',
        r'impression:?\s*([^\.]+)',
        r'assessment:?\s*([^\.]+)'
    ]
    
    for pattern in diagnosis_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        findings.extend(matches)
    
    # Look for symptom patterns
    symptom_keywords = ['presented with', 'symptoms include', 'complains of', 'exhibiting']
    for keyword in symptom_keywords:
        if keyword in all_text.lower():
            context = all_text.lower().split(keyword)[1][:200]
            findings.append(f"Presenting: {context.strip()}")
    
    # Create summary
    summary_parts = []
    
    summary_parts.append(f"**Query Analysis**: '{query}'")
    summary_parts.append(f"**Documents Found**: {len(retrieved_docs)} relevant clinical documents")
    
    if findings:
        summary_parts.append("**Key Clinical Information Found**:")
        for i, finding in enumerate(set(findings[:5])):  # Remove duplicates, limit to 5
            clean_finding = finding.strip()
            if len(clean_finding) > 20:
                summary_parts.append(f"  • {clean_finding}")
    
    # Add document insights
    high_score_docs = [doc for doc in retrieved_docs if doc["score"] > 0.6]
    if high_score_docs:
        summary_parts.append(f"**High-Relevance Documents**: {len(high_score_docs)} documents with strong relevance")
    
    summary_parts.append("\n**Recommendation**: Review the retrieved clinical documents below for comprehensive details.")
    
    return "\n\n".join(summary_parts)

def extract_clinical_snippets(text, query_terms):
    """Extract relevant snippets from clinical text"""
    sentences = re.split(r'[.!?]', text)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if any(term.lower() in sentence.lower() for term in query_terms) and len(sentence) > 20:
            relevant_sentences.append(sentence)
    
    return relevant_sentences[:3]  # Return top 3 relevant sentences

def main():
    """Main Streamlit application - Retrieval Focused"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Clinical Document Retrieval System</div>', unsafe_allow_html=True)
    st.markdown("### Intelligent Search for Clinical Documentation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        
        st.header("📊 System Status")
        
        # Check file existence
        files_status = {
            "FAISS Index": os.path.exists("faiss.index"),
            "Documents Metadata": os.path.exists("documents.pkl"),
            "Embedding Model": load_sentence_transformer() is not None
        }
        
        for file, exists in files_status.items():
            status = "✅" if exists else "❌"
            st.write(f"{status} {file}")
        
        st.header("💡 Sample Clinical Queries")
        sample_queries = [
            "stroke symptoms and diagnosis",
            "patient with facial droop and slurred speech",
            "hypertension treatment guidelines", 
            "diabetes management complications",
            "chest pain differential diagnosis",
            "neurological examination findings",
            "cardiac arrest protocols",
            "pneumonia treatment antibiotics"
        ]
        
        for query in sample_queries:
            if st.button(f"🔍 {query}", use_container_width=True, key=f"btn_{query[:10]}"):
                st.session_state.query = query
                st.rerun()
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Load essential components
    st.markdown("---")
    
    model = load_sentence_transformer()
    index = load_faiss_index()
    documents = load_documents()
    
    # Check if all essential components are loaded
    if model is None or index is None or documents is None:
        st.error("""
        ❌ System initialization failed. Required components:
        - FAISS index (faiss.index)
        - Documents metadata (documents.pkl) 
        - Sentence embedding model
        """)
        return
    
    st.markdown("### 🔍 Clinical Query Search")
    
    query = st.text_area(
        "Enter clinical question or search terms:",
        value=st.session_state.query,
        height=80,
        placeholder="e.g., stroke symptoms, diabetes management, chest pain evaluation..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Search Clinical Database", use_container_width=True, type="primary")
    
    if search_button and query:
        with st.spinner("🔍 Searching clinical documents..."):
            # Retrieve documents
            retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
            
            # Generate intelligent summary
            summary = generate_insightful_summary(query, retrieved_docs)
            
            # Store results in session state
            st.session_state.last_results = {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "summary": summary
            }
            
            # Force rerun to display results
            st.rerun()
    
    # Display results if available
    if hasattr(st.session_state, 'last_results'):
        results = st.session_state.last_results
        
        st.markdown("---")
        st.markdown("### 📋 Search Summary")
        
        # Display summary
        with st.container():
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(results["summary"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display retrieved documents
        st.markdown("---")
        st.markdown(f"### 📚 Clinical Documents ({len(results['retrieved_docs'])})")
        
        if not results['retrieved_docs']:
            st.info("""
            **No relevant documents found.** Try:
            - Using different keywords
            - Making the query more specific
            - Using clinical terminology
            - Checking the sample queries in the sidebar
            """)
        else:
            for i, doc in enumerate(results['retrieved_docs']):
                with st.container():
                    st.markdown('<div class="document-box">', unsafe_allow_html=True)
                    
                    # Header with score and filename
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**📄 Document {i+1}: {doc['filename']}**")
                    with col2:
                        score = doc['score']
                        if score > 0.7:
                            st.markdown(f'<span class="score-high">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                        elif score > 0.5:
                            st.markdown(f'<span class="score-medium">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="score-low">Relevance: {score:.3f}</span>', unsafe_allow_html=True)
                    
                    # Quick preview
                    st.markdown(f"**Preview:** {doc['preview']}")
                    
                    # Full document content in expander
                    with st.expander("View Full Clinical Content", expanded=False):
                        # Extract query-related snippets
                        query_terms = results['query'].split()
                        snippets = extract_clinical_snippets(doc['text'], query_terms)
                        
                        if snippets:
                            st.markdown("**Relevant Excerpts:**")
                            for snippet in snippets:
                                st.markdown(f"• {snippet}")
                            st.markdown("---")
                        
                        # Full text
                        st.text_area(
                            "Complete Document Text",
                            doc['text'],
                            height=300,
                            key=f"full_doc_{i}",
                            label_visibility="collapsed"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick stats in footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(documents))
    
    with col2:
        if hasattr(st.session_state, 'last_results'):
            st.metric("Last Search Results", len(st.session_state.last_results['retrieved_docs']))
        else:
            st.metric("Ready for Queries", "✓")
    
    with col3:
        st.metric("System Status", "Operational")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🏥 Clinical Document Retrieval System | Powered by FAISS + Sentence Transformers</p>
        <p>⚠️ For educational and research purposes only. Not for clinical decision making.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
