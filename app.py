import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
    .model-output {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer from local files"""
    try:
        st.info("🔄 Loading Sentence Transformer...")
        # Try local model first, then fallback
        try:
            model = SentenceTransformer('.', local_files_only=True)
            st.success("✅ Local Sentence Transformer loaded!")
        except:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Fallback Sentence Transformer loaded!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading Sentence Transformer: {e}")
        return None

@st.cache_resource
def load_faiss_index():
    """Load FAISS index"""
    try:
        if os.path.exists("faiss.index"):
            st.info("🔄 Loading FAISS index...")
            index = faiss.read_index("faiss.index")
            st.success("✅ FAISS index loaded!")
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
            st.success(f"✅ Loaded {len(documents)} documents!")
            return documents
        else:
            st.error("❌ Documents file not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading documents: {e}")
        return None

@st.cache_resource
def load_qwen_model():
    """Load Qwen model - USING THE SAME APPROACH AS YOUR COLAB NOTEBOOK"""
    try:
        st.info("🔄 Loading Qwen 2.5-1.5B Model...")
        
        # Use the exact same model name as your Colab notebook
        gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Try to load from local files first, then from HuggingFace
        try:
            # First try local files
            tokenizer = AutoTokenizer.from_pretrained(
                ".",
                trust_remote_code=True,
                local_files_only=True
            )
            generator = AutoModelForCausalLM.from_pretrained(
                ".",
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True
            )
            st.success("✅ Qwen model loaded from local files!")
            
        except Exception as local_error:
            st.warning("⚠️ Local model not found, downloading from HuggingFace...")
            # Fallback to online download (same as Colab)
            tokenizer = AutoTokenizer.from_pretrained(
                gen_model_name,
                trust_remote_code=True
            )
            generator = AutoModelForCausalLM.from_pretrained(
                gen_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            st.success("✅ Qwen model downloaded and loaded!")
        
        return tokenizer, generator
        
    except Exception as e:
        st.error(f"❌ Error loading Qwen model: {e}")
        st.info("💡 Try running: `pip install transformers accelerate`")
        return None, None

def retrieve_documents(query, model, index, documents, top_k=5):
    """Retrieve relevant documents - SAME AS YOUR COLAB NOTEBOOK"""
    try:
        # Encode query
        q = model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        scores, idx = index.search(q, top_k * 2)
        
        # Remove duplicates (same logic as Colab)
        out = []
        seen_sources = set()
        
        for score, i in zip(scores[0], idx[0]):
            if i < len(documents):
                source = documents[i]["source"]
                
                # Skip duplicates
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                out.append({
                    "score": float(score),
                    "source": source,
                    "text": documents[i]["text"],
                    "filename": os.path.basename(source)
                })
                
                if len(out) >= top_k:
                    break
        
        return out
    except Exception as e:
        st.error(f"Error in retrieval: {e}")
        return []

def generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens=400):
    """GENERATE ANSWER - USING THE EXACT SAME CODE FROM YOUR COLAB NOTEBOOK"""
    try:
        # EXACT SAME CONTEXT EXTRACTION AS COLAB
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Limit to avoid duplicates
            doc_text = doc["text"].strip()
            # Take first 800 characters or until sentence end (SAME AS COLAB)
            if len(doc_text) > 800:
                trunc_point = doc_text[:800].rfind('.')
                if trunc_point > 500:
                    doc_text = doc_text[:trunc_point+1]
                else:
                    doc_text = doc_text[:800] + "..."
            context_parts.append(f"Document {i+1}: {doc_text}")

        context = "\n\n".join(context_parts)

        # EXACT SAME PROMPT TEMPLATE AS COLAB
        prompt = f"""Based on the following clinical documentation, provide a concise medical assessment.

CLINICAL DOCUMENTATION:
{context}

CLINICAL QUESTION: {query}

MEDICAL ASSESSMENT:"""

        # EXACT SAME GENERATION PARAMETERS AS COLAB
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(generator.device)

        # EXACT SAME GENERATION CALL AS COLAB
        output = generator.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

        # EXACT SAME OUTPUT PROCESSING AS COLAB
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Split and return only the generated answer
        if "MEDICAL ASSESSMENT:" in full_output:
            answer = full_output.split("MEDICAL ASSESSMENT:")[-1].strip()
        else:
            answer = full_output

        return answer

    except Exception as e:
        return f"Error in generation: {str(e)}"

def main():
    """Main Streamlit application - NOW WITH PROPER MODEL GENERATION"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Clinical RAG System</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Clinical Document Retrieval and Assessment Generation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        max_tokens = st.slider("Maximum answer tokens", 200, 800, 400)
        
        st.header("📊 System Status")
        
        # Load models
        st.info("Loading models...")
        model = load_sentence_transformer()
        index = load_faiss_index()
        documents = load_documents()
        tokenizer, generator = load_qwen_model()
        
        # Status check
        status_checks = {
            "Embedding Model": model is not None,
            "FAISS Index": index is not None,
            "Documents": documents is not None,
            "Qwen Model": generator is not None
        }
        
        for component, status in status_checks.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {component}")
        
        st.header("💡 Sample Clinical Queries")
        sample_queries = [
            "What is the likely diagnosis for a patient with slurred speech and facial droop?",
            "What are the key clinical findings in this stroke case?",
            "What diagnostic tests should be performed for suspected stroke?",
            "Patient with sudden severe headache and vomiting — possible diagnosis?",
            "Differentiate between ischemic and hemorrhagic stroke symptoms"
        ]
        
        for query in sample_queries:
            if st.button(f"🧠 {query[:60]}...", use_container_width=True, key=f"sample_{hash(query)}"):
                st.session_state.query = query
                st.rerun()
    
    # Initialize session state
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Main interface
    st.markdown("---")
    st.markdown("### 🔍 Enter Clinical Query")
    
    query = st.text_area(
        "Describe the clinical scenario or ask a medical question:",
        value=st.session_state.query,
        height=100,
        placeholder="e.g., Patient presents with sudden onset of right-sided weakness and aphasia...",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Generate Clinical Assessment", use_container_width=True, type="primary")
    
    # Process query
    if search_button and query:
        if not all([model, index, documents, generator]):
            st.error("❌ Not all models are loaded. Check system status in sidebar.")
            return
            
        with st.spinner("🔍 Retrieving clinical documents..."):
            retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
            
            if not retrieved_docs:
                st.error("❌ No relevant documents found. Try rephrasing your query.")
                return
            
            # Show retrieval results first
            st.session_state.retrieved_docs = retrieved_docs
            st.session_state.current_query = query
            
            # Then generate answer
            with st.spinner("🤔 Generating clinical assessment (this may take 20-30 seconds)..."):
                answer = generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens)
                st.session_state.generated_answer = answer
    
    # Display results
    if hasattr(st.session_state, 'generated_answer'):
        st.markdown("---")
        st.markdown("### 💡 AI Clinical Assessment")
        
        # Display the model-generated answer (SAME AS COLAB OUTPUT)
        with st.container():
            st.markdown('<div class="model-output">', unsafe_allow_html=True)
            st.markdown("**Qwen Model Assessment:**")
            st.write(st.session_state.generated_answer)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display retrieved documents
        st.markdown("---")
        st.markdown(f"### 📚 Supporting Clinical Documents ({len(st.session_state.retrieved_docs)})")
        
        for i, doc in enumerate(st.session_state.retrieved_docs):
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
                    doc_text = doc['text'].strip()
                    st.text_area(
                        f"Content {i+1}",
                        doc_text,
                        height=200,
                        key=f"doc_{i}",
                        label_visibility="collapsed"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with model info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", len(documents) if documents else 0)
    
    with col2:
        st.metric("AI Model", "Qwen 2.5-1.5B")
    
    with col3:
        if hasattr(st.session_state, 'generated_answer'):
            st.metric("Assessment Generated", "✅")
        else:
            st.metric("Ready", "🔍")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🏥 Clinical RAG System | FAISS + Sentence Transformers + Qwen 2.5-1.5B</p>
        <p>⚠️ For educational and research purposes only. Not for clinical decision making.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
