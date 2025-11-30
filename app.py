# app.py - REPLICATES YOUR EXACT TESTING OUTPUT FORMAT
import streamlit as st
import faiss
import pickle
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer

# Page configuration - SIMPLE AND CLEAN
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide"
)

# Remove all custom CSS for clean terminal-like output
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models"""
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
        
        # Load models
        embedding_model = SentenceTransformer('sentence_model')
        faiss_index = faiss.read_index('faiss.index')
        
        with open('documents.pkl', 'rb') as f:
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
    """Retrieval function matching your testing script"""
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

def generate_answer(query, retrieved_docs, max_tokens=400):
    """Generate answer in exact same format as your testing script"""
    context = "\n\n".join(d["text"][:1000] for d in retrieved_docs)

    prompt = f"""
You are a helpful and medically accurate Clinical RAG system.

Context from relevant clinical notes:
{context}

Question:
{query}

Provide a concise clinical reasoning answer.
"""

    inputs = st.session_state.tokenizer(prompt, return_tensors="pt").to(st.session_state.generator.device)
    output = st.session_state.generator.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.3,
        do_sample=True
    )

    return st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)

def display_results_exact_format(results, query_number, query_type):
    """Display results in EXACT same format as your testing script"""
    
    # Header matching your testing format
    st.markdown(f"**🎯 TEST {query_number}/10: {query_type}**")
    st.markdown("\n" + "=" * 80)
    st.markdown(f"🧠 STROKE DOMAIN - EXAMPLE {query_number}: {query_type}")
    st.markdown("=" * 80)
    
    # Query section
    st.markdown(f"**📋 CLINICAL QUERY:**")
    st.markdown(f"   {results['query']}")
    
    # Answer section - EXACT same format
    st.markdown(f"\n**💡 CLINICAL ASSESSMENT:**")
    st.markdown("-" * 60)
    st.text(results["answer"])
    st.markdown("-" * 60)
    
    # Retrieved documents - EXACT same format
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
    # Simple header matching your testing format
    st.markdown("🚀" * 20)
    st.markdown("🧠 CLINICAL RAG SYSTEM")
    st.markdown("🚀" * 20)
    
    # Initialize models
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading clinical models..."):
            embedding_model, faiss_index, documents_data, tokenizer, generator = load_models()
            
            if all([embedding_model, faiss_index, documents_data, tokenizer, generator]):
                st.session_state.embedding_model = embedding_model
                st.session_state.faiss_index = faiss_index
                st.session_state.documents_data = documents_data
                st.session_state.tokenizer = tokenizer
                st.session_state.generator = generator
                st.session_state.models_loaded = True
                st.success(f"✅ Loaded {len(documents_data)} clinical documents!")
            else:
                st.error("❌ Failed to load models")
                return
    
    # Test queries matching your exact testing cases
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
    
    # Display test cases as buttons
    for test_case in test_cases:
        if st.button(f"TEST {test_case['number']}: {test_case['type']}", key=f"test_{test_case['number']}"):
            with st.spinner(f"Processing {test_case['type']}..."):
                # Retrieve documents
                retrieved_docs = retrieve_documents(test_case['query'], top_k=5)
                
                # Generate answer
                answer = generate_answer(test_case['query'], retrieved_docs)
                
                # Store results
                results = {
                    'query': test_case['query'],
                    'retrieved': retrieved_docs,
                    'answer': answer
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
                
                # Generate answer
                answer = generate_answer(custom_query, retrieved_docs)
                
                # Store results
                results = {
                    'query': custom_query,
                    'retrieved': retrieved_docs,
                    'answer': answer
                }
                
                # Display in exact same format
                st.markdown("\n" + "=" * 80)
                st.markdown("🧠 CUSTOM CLINICAL QUERY RESULTS")
                st.markdown("=" * 80)
                
                st.markdown(f"**📋 CLINICAL QUERY:**")
                st.markdown(f"   {results['query']}")
                
                st.markdown(f"\n**💡 CLINICAL ASSESSMENT:**")
                st.markdown("-" * 60)
                st.text(results["answer"])
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
    
    # Performance summary (matching your testing output)
    if 'test_results' in st.session_state:
        st.markdown("\n" + "=" * 80)
        st.markdown("📊 PERFORMANCE SUMMARY")
        st.markdown("=" * 80)
        
        st.markdown("**✅ System Status:** OPERATIONAL")
        st.markdown(f"**📚 Documents Loaded:** {len(st.session_state.documents_data)}")
        st.markdown("**🎯 Ready for Clinical Queries**")

if __name__ == "__main__":
    main()
