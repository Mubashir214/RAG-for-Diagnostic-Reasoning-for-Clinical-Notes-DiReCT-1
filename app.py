# app.py
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import os
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Clinical RAG Assistant",
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
    .clinical-query {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .document-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models and indexes with caching"""
    try:
        # Load sentence transformer model
        embedding_model = SentenceTransformer('sentence_model')
        
        # Load FAISS index
        faiss_index = faiss.read_index('clinical_faiss_index.index')
        
        # Load documents
        with open('clinical_documents.pkl', 'rb') as f:
            documents_data = pickle.load(f)
        
        # Load generator model
        generator_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        generator = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        return embedding_model, faiss_index, documents_data, tokenizer, generator
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def retrieve_documents(query, top_k=5):
    """Enhanced retrieval function"""
    if st.session_state.faiss_index is None:
        return []
    
    q = st.session_state.embedding_model.encode([query], convert_to_numpy=True)
    scores, idx = st.session_state.faiss_index.search(q, top_k * 3)
    
    out = []
    seen_sources = set()
    
    for score, i in zip(scores[0], idx[0]):
        source = st.session_state.documents_data[i]["source"]
        
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        # Medical domain boosting
        adjusted_score = float(score)
        query_lower = query.lower()
        doc_text_lower = st.session_state.documents_data[i]["text"].lower()
        
        # Boost stroke-related documents
        if any(term in query_lower for term in ['stroke', 'ischemic', 'hemorrhagic', 'tia', 'neurological']):
            if any(term in doc_text_lower for term in ['stroke', 'cerebral', 'infarct', 'hemorrhage', 'neurological']):
                adjusted_score += 0.15
        
        out.append({
            "score": adjusted_score,
            "source": source,
            "text": st.session_state.documents_data[i]["text"],
            "filename": os.path.basename(source),
            "original_score": float(score)
        })
        
        if len(out) >= top_k:
            break
    
    out.sort(key=lambda x: x['score'], reverse=True)
    return out

def generate_clinical_answer(query, retrieved_docs, max_tokens=400):
    """Generate clinical answer with enhanced prompting"""
    if not retrieved_docs:
        return "No relevant clinical documents found for this query."
    
    # Prepare context
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        doc_text = doc["text"].strip()
        
        # Clean artifacts
        doc_text = re.sub(r'Human:.*?(?=Answer:|$)', '', doc_text, flags=re.DOTALL)
        doc_text = re.sub(r'\s+', ' ', doc_text).strip()
        
        # Smart truncation
        if len(doc_text) > 600:
            trunc_point = doc_text[:600].rfind('.')
            if trunc_point > 300:
                doc_text = doc_text[:trunc_point+1]
            else:
                doc_text = doc_text[:600] + "..."
        
        context_parts.append(f"Document {i+1} (Relevance: {doc['score']:.3f}): {doc_text}")
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt
    prompt = f"""As a clinical AI assistant, analyze the following medical documentation and provide a focused assessment.

CLINICAL DOCUMENTATION:
{context}

CLINICAL QUESTION: {query}

Please provide a concise clinical assessment focusing on:
1. Most likely diagnosis based on the documentation
2. Key supporting evidence from the clinical context
3. Recommended diagnostic or management considerations

CLINICAL ASSESSMENT:"""
    
    inputs = st.session_state.tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    )
    inputs = {k: v.to(st.session_state.generator.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = st.session_state.generator.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            eos_token_id=st.session_state.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    full_output = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "CLINICAL ASSESSMENT:" in full_output:
        answer = full_output.split("CLINICAL ASSESSMENT:")[-1].strip()
    else:
        answer = full_output
    
    # Clean artifacts
    answer = re.sub(r'Human:.*?(?=Answer:|$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'You are a helpful.*?system\.', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏥 Clinical RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        max_tokens = st.slider("Maximum answer length", 100, 800, 400)
        
        st.header("Quick Queries")
        quick_queries = [
            "Patient with sudden weakness and facial droop",
            "Patient with severe headache and vomiting", 
            "Patient with chest pain radiating to left arm",
            "Patient with fever and hypotension",
            "Patient with polyuria and weight loss"
        ]
        
        for query in quick_queries:
            if st.button(query, key=query):
                st.session_state.current_query = query
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading clinical models... This may take a few minutes."):
            embedding_model, faiss_index, documents_data, tokenizer, generator = load_models()
            
            if all([embedding_model, faiss_index, documents_data, tokenizer, generator]):
                st.session_state.embedding_model = embedding_model
                st.session_state.faiss_index = faiss_index
                st.session_state.documents_data = documents_data
                st.session_state.tokenizer = tokenizer
                st.session_state.generator = generator
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please check if all model files are available.")
                return
    
    # Main query interface
    st.markdown("### 🔍 Clinical Query")
    
    query = st.text_area(
        "Enter your clinical question:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="e.g., Patient with sudden weakness on one side and slurred speech — likely diagnosis?"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Process Query", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Retrieving relevant clinical documents..."):
                    retrieved_docs = retrieve_documents(query, top_k=top_k)
                
                with st.spinner("Generating clinical assessment..."):
                    answer = generate_clinical_answer(query, retrieved_docs, max_tokens)
                
                # Store results in session state
                st.session_state.last_results = {
                    'query': query,
                    'retrieved_docs': retrieved_docs,
                    'answer': answer
                }
            else:
                st.warning("Please enter a clinical query.")
    
    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            if 'last_results' in st.session_state:
                del st.session_state.last_results
            st.session_state.current_query = ""
            st.rerun()
    
    # Display results if available
    if 'last_results' in st.session_state:
        results = st.session_state.last_results
        
        st.markdown("---")
        st.markdown("### 💡 Clinical Assessment")
        st.markdown(f'<div class="clinical-query"><strong>Query:</strong> {results["query"]}</div>', unsafe_allow_html=True)
        
        st.markdown("**Assessment:**")
        st.write(results['answer'])
        
        # Retrieved documents section
        st.markdown("### 📚 Retrieved Clinical Documents")
        
        if results['retrieved_docs']:
            for i, doc in enumerate(results['retrieved_docs']):
                with st.expander(f"Document {i+1} | Score: {doc['score']:.3f} | {doc['filename']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Clinical Findings:**")
                        st.write(doc['text'][:800] + "..." if len(doc['text']) > 800 else doc['text'])
                    
                    with col2:
                        st.write("**Metadata:**")
                        st.write(f"**Score:** {doc['score']:.3f}")
                        st.write(f"**Source:** {doc['filename']}")
                        
                        # Extract category from path
                        path_parts = doc['source'].split('/')
                        category = "Unknown"
                        for part in path_parts:
                            if part in ['Stroke', 'Ischemic Stroke', 'Hemorrhagic Stroke', 'Multiple Sclerosis', 'Gastritis']:
                                category = part
                                break
                        st.write(f"**Category:** {category}")
        else:
            st.warning("No relevant documents found for this query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <i>Clinical RAG Assistant - For educational and research purposes only</i>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()