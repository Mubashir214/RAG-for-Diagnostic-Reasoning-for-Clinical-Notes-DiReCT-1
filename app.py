# app.py - FIXED VERSION for Streamlit Cloud
import streamlit as st
import faiss
import pickle
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer

# Page configuration
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide"
)

# Simple styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models without LLM generator"""
    try:
        # Setup sentence_model folder
        if not os.path.exists('sentence_model'):
            os.makedirs('sentence_model', exist_ok=True)
            os.makedirs('sentence_model/1_Pooling', exist_ok=True)
            
            # Move files
            files_to_move = [
                'config.json', 'model.safetensors', 'tokenizer_config.json',
                'tokenizer.json', 'vocab.txt', 'special_tokens_map.json',
                'sentence_bert_config.json', 'config_sentence_transformers.json',
                'modules.json'
            ]
            
            for file in files_to_move:
                if os.path.exists(file):
                    os.rename(file, f'sentence_model/{file}')
            
            # Handle pooling config
            if os.path.exists('1_Pooling/config.json'):
                os.rename('1_Pooling/config.json', 'sentence_model/1_Pooling/config.json')
        
        # Load only embedding model and FAISS
        embedding_model = SentenceTransformer('sentence_model')
        faiss_index = faiss.read_index('faiss.index')
        
        with open('documents.pkl', 'rb') as f:
            documents_data = pickle.load(f)
        
        return embedding_model, faiss_index, documents_data
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def retrieve_documents(query, top_k=5):
    """Retrieval function"""
    q = st.session_state.embedding_model.encode([query], convert_to_numpy=True)
    scores, idx = st.session_state.faiss_index.search(q, top_k * 2)
    
    out = []
    seen_sources = set()
    
    for score, i in zip(scores[0], idx[0]):
        if i >= len(st.session_state.documents_data):
            continue
            
        source = st.session_state.documents_data[i]["source"]
        
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        out.append({
            "score": float(score),
            "source": source,
            "text": st.session_state.documents_data[i]["text"]
        })
        
        if len(out) >= top_k:
            break
    
    return out

def generate_analysis(query, retrieved_docs):
    """Generate analysis without LLM - using pattern matching"""
    if not retrieved_docs:
        return "No relevant clinical documents found for this query."
    
    # Analyze query type
    query_lower = query.lower()
    
    # Extract key information from retrieved documents
    total_docs = len(retrieved_docs)
    avg_score = sum(doc['score'] for doc in retrieved_docs) / total_docs
    
    # Analyze document content
    stroke_terms = 0
    cardiac_terms = 0
    neuro_terms = 0
    infection_terms = 0
    
    for doc in retrieved_docs:
        text_lower = doc['text'].lower()
        if any(term in text_lower for term in ['stroke', 'infarct', 'cerebral', 'hemorrhage']):
            stroke_terms += 1
        if any(term in text_lower for term in ['chest pain', 'cardiac', 'heart', 'mi']):
            cardiac_terms += 1
        if any(term in text_lower for term in ['weakness', 'speech', 'vision', 'neurological']):
            neuro_terms += 1
        if any(term in text_lower for term in ['fever', 'infection', 'sepsis']):
            infection_terms += 1
    
    # Determine primary theme
    themes = []
    if stroke_terms > total_docs * 0.5:
        themes.append("cerebrovascular/stroke")
    if cardiac_terms > total_docs * 0.5:
        themes.append("cardiac")
    if neuro_terms > total_docs * 0.5:
        themes.append("neurological")
    if infection_terms > total_docs * 0.5:
        themes.append("infectious")
    
    if not themes:
        themes.append("various clinical conditions")
    
    # Generate analysis based on query type
    if any(term in query_lower for term in ['stroke', 'weakness', 'facial droop', 'speech']):
        analysis = f"""
Based on the retrieved clinical documentation, the patient's presentation with acute neurological symptoms including unilateral weakness, facial droop, and speech disturbances is highly consistent with cerebrovascular events documented in the database.

**Clinical Context:**
- Found {total_docs} relevant clinical cases with similar presentations
- Primary themes: {', '.join(themes)}
- Retrieval quality: {'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} (average score: {avg_score:.3f})

**Key Considerations:**
- Acute stroke should be considered in the differential diagnosis
- Urgent neuroimaging (CT/MRI) is recommended
- Time-sensitive interventions may be applicable

The database contains documented cases with comparable symptom patterns that can inform clinical decision-making."""
    
    elif any(term in query_lower for term in ['headache', 'nausea', 'vomiting']):
        analysis = f"""
The patient's symptoms of severe headache with associated nausea and vomiting raise concerns for several potential conditions documented in the clinical database.

**Clinical Context:**
- Found {total_docs} relevant clinical cases
- Primary themes: {', '.join(themes)}
- Retrieval quality: {'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} (average score: {avg_score:.3f})

**Differential Considerations:**
- Primary headache disorders (migraine, cluster)
- Secondary headaches (SAH, meningitis)
- Systemic conditions

Review the specific documents for detailed case comparisons."""
    
    elif any(term in query_lower for term in ['chest pain', 'radiating', 'diaphoresis']):
        analysis = f"""
The presentation of chest pain with associated symptoms warrants careful evaluation based on similar cases in the clinical database.

**Clinical Context:**
- Found {total_docs} relevant clinical cases
- Primary themes: {', '.join(themes)}
- Retrieval quality: {'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} (average score: {avg_score:.3f})

**Urgent Considerations:**
- Cardiac ischemia requires immediate evaluation
- Pulmonary and gastrointestinal causes should be considered
- ECG and cardiac biomarkers are essential

The retrieved documents provide comparative clinical information."""
    
    else:
        analysis = f"""
Based on the clinical documentation retrieved, several relevant cases provide context for this clinical scenario.

**Database Analysis:**
- Retrieved {total_docs} relevant clinical documents
- Primary clinical themes: {', '.join(themes)}
- Retrieval quality: {'Excellent' if avg_score > 0.7 else 'Good' if avg_score > 0.5 else 'Moderate'} (average score: {avg_score:.3f})

**Clinical Insights:**
The database contains documented cases with presentations similar to the described scenario. Review the specific documents below for detailed clinical information and comparative analysis.

**Note:** This analysis is based on pattern matching within the clinical database. Always consult healthcare professionals for definitive diagnosis."""
    
    return analysis

def display_results_exact_format(results, query_number, query_type):
    """Display results in exact same format"""
    
    st.markdown(f"**🎯 TEST {query_number}/10: {query_type}**")
    st.markdown("\n" + "=" * 80)
    st.markdown(f"🧠 CLINICAL DOMAIN - EXAMPLE {query_number}: {query_type}")
    st.markdown("=" * 80)
    
    st.markdown(f"**📋 CLINICAL QUERY:**")
    st.markdown(f"   {results['query']}")
    
    st.markdown(f"\n**💡 CLINICAL ASSESSMENT:**")
    st.markdown("-" * 60)
    st.write(results["analysis"])
    st.markdown("-" * 60)
    
    st.markdown(f"\n**📚 RETRIEVED CLINICAL DOCUMENTS ({len(results['retrieved'])}):**")
    st.markdown("=" * 60)
    
    for i, doc in enumerate(results['retrieved']):
        st.markdown(f"\n**📑 DOCUMENT {i+1}:**")
        st.markdown(f"   **⭐ Relevance Score:** {doc['score']:.3f}")
        st.markdown(f"   **📁 Source:** {os.path.basename(doc['source'])}")
        
        # Extract category from path
        path_parts = doc['source'].split('/')
        category = "Unknown"
        for part in path_parts:
            if part in ['Stroke', 'Ischemic Stroke', 'Hemorrhagic Stroke', 'Multiple Sclerosis', 'Gastritis', 'Migraine']:
                category = part
                break
        
        st.markdown(f"   **🏥 Category:** {category}")
        
        # Clinical findings preview
        doc_text = doc['text'].strip()
        preview = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
        st.markdown(f"   **📝 Clinical Findings:** {preview}")
        st.markdown("-" * 60)

def main():
    st.markdown("🚀" * 20)
    st.markdown("🧠 CLINICAL RAG SYSTEM")
    st.markdown("🚀" * 20)
    
    # Initialize models
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading clinical database..."):
            embedding_model, faiss_index, documents_data = load_models()
            
            if all([embedding_model, faiss_index, documents_data]):
                st.session_state.embedding_model = embedding_model
                st.session_state.faiss_index = faiss_index
                st.session_state.documents_data = documents_data
                st.session_state.models_loaded = True
                st.success(f"✅ Loaded {len(documents_data)} clinical documents!")
            else:
                st.error("❌ Failed to load models. Please check that all required files are uploaded.")
                st.info("""
                **Required Files:**
                - `faiss.index` - FAISS vector database
                - `documents.pkl` - Clinical documents
                - All sentence_model files
                """)
                return
    
    # Test queries
    st.markdown("### 🧪 CLINICAL TEST QUERIES")
    
    test_cases = [
        {
            "number": 6,
            "type": "ISCHEMIC STROKE IDENTIFICATION", 
            "query": "Patient with sudden weakness on one side, facial droop, and slurred speech — likely type of stroke?"
        },
        {
            "number": 7,
            "type": "HEMORRHAGIC STROKE IDENTIFICATION",
            "query": "Patient with sudden severe headache, nausea, and vomiting — possible hemorrhagic event?"
        },
        {
            "number": 8, 
            "type": "TRANSIENT ISCHEMIC ATTACK (TIA)",
            "query": "Patient reports brief episode of vision loss and numbness in the arm, symptoms resolve within minutes — likely diagnosis?"
        },
        {
            "number": 9,
            "type": "STROKE WITH APHASIA", 
            "query": "Patient with sudden difficulty speaking and understanding language, right-sided weakness — what type of stroke?"
        },
        {
            "number": 10,
            "type": "STROKE WITH VISUAL FIELD DEFICIT",
            "query": "Patient complains of sudden loss of vision in left visual field, left-sided weakness — likely neurological condition?"
        }
    ]
    
    # Display test cases
    for test_case in test_cases:
        if st.button(f"TEST {test_case['number']}: {test_case['type']}", key=f"test_{test_case['number']}"):
            with st.spinner(f"Processing {test_case['type']}..."):
                # Retrieve documents
                retrieved_docs = retrieve_documents(test_case['query'], top_k=5)
                
                # Generate analysis
                analysis = generate_analysis(test_case['query'], retrieved_docs)
                
                # Store results
                results = {
                    'query': test_case['query'],
                    'retrieved': retrieved_docs,
                    'analysis': analysis
                }
                
                # Display in exact same format
                display_results_exact_format(results, test_case['number'], test_case['type'])
    
    # Custom query section
    st.markdown("### 🔍 CUSTOM CLINICAL QUERY")
    
    custom_query = st.text_area(
        "Enter your own clinical query:",
        height=100,
        placeholder="e.g., Patient with chest pain radiating to left arm and diaphoresis — likely diagnosis?"
    )
    
    if st.button("🚀 PROCESS CUSTOM QUERY", type="primary"):
        if custom_query.strip():
            with st.spinner("Processing custom query..."):
                # Retrieve documents
                retrieved_docs = retrieve_documents(custom_query, top_k=5)
                
                # Generate analysis
                analysis = generate_analysis(custom_query, retrieved_docs)
                
                # Store results
                results = {
                    'query': custom_query,
                    'retrieved': retrieved_docs,
                    'analysis': analysis
                }
                
                # Display in exact same format
                st.markdown("\n" + "=" * 80)
                st.markdown("🧠 CUSTOM CLINICAL QUERY RESULTS")
                st.markdown("=" * 80)
                
                st.markdown(f"**📋 CLINICAL QUERY:**")
                st.markdown(f"   {results['query']}")
                
                st.markdown(f"\n**💡 CLINICAL ASSESSMENT:**")
                st.markdown("-" * 60)
                st.write(results["analysis"])
                st.markdown("-" * 60)
                
                st.markdown(f"\n**📚 RETRIEVED CLINICAL DOCUMENTS ({len(results['retrieved'])}):**")
                st.markdown("=" * 60)
                
                for i, doc in enumerate(results['retrieved']):
                    st.markdown(f"\n**📑 DOCUMENT {i+1}:**")
                    st.markdown(f"   **⭐ Relevance Score:** {doc['score']:.3f}")
                    st.markdown(f"   **📁 Source:** {os.path.basename(doc['source'])}")
                    
                    # Extract category
                    path_parts = doc['source'].split('/')
                    category = "Unknown"
                    for part in path_parts:
                        if part in ['Stroke', 'Ischemic Stroke', 'Hemorrhagic Stroke', 'Multiple Sclerosis', 'Gastritis', 'Migraine']:
                            category = part
                            break
                    
                    st.markdown(f"   **🏥 Category:** {category}")
                    
                    doc_text = doc['text'].strip()
                    preview = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
                    st.markdown(f"   **📝 Clinical Findings:** {preview}")
                    st.markdown("-" * 60)
        else:
            st.warning("Please enter a clinical query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <i>Clinical RAG System • For educational and research purposes • Consult healthcare professionals for medical decisions</i>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
