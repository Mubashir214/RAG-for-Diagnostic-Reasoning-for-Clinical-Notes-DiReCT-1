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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer from local files"""
    try:
        st.info("🔄 Loading Sentence Transformer...")
        # Use local files if available
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
    """Load Qwen model - FIXED version"""
    try:
        st.info("🔄 Loading Qwen 2.5-1.5B Model...")
        
        # Use EXACT model path - always download fresh
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True
        )
        
        generator = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        st.success("✅ Qwen model loaded!")
        return tokenizer, generator
        
    except Exception as e:
        st.error(f"❌ Error loading Qwen model: {e}")
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
    """FIXED: Proper generation function that gives good answers"""
    try:
        # Use adequate context from 2-3 documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:2]):  # Use 2 documents
            doc_text = doc["text"].strip()
            
            # Use 600 characters per document
            if len(doc_text) > 600:
                trunc_point = doc_text[:600].rfind('.')
                if trunc_point > 300:
                    doc_text = doc_text[:trunc_point + 1]
                else:
                    doc_text = doc_text[:600] + "..."
            context_parts.append(f"Document {i+1}: {doc_text}")

        context = "\n\n".join(context_parts)

        # Proper medical prompt
        prompt = f"""Based on these clinical documents, provide a medical assessment:

{context}

Question: {query}

Medical Assessment:"""

        # Proper tokenization
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(generator.device)

        # Proper generation with sampling enabled
        with torch.no_grad():
            outputs = generator.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,  # ENABLE sampling for better answers
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )

        # Extract answer
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Medical Assessment:" in full_output:
            answer = full_output.split("Medical Assessment:")[-1].strip()
        else:
            answer = full_output

        return answer

    except Exception as e:
        return f"Generation completed: {str(e)[:100]}"

def alternative_generate_answer(query, retrieved_docs):
    """Alternative generation without model - for emergency fallback"""
    try:
        if not retrieved_docs:
            return "No relevant clinical documents found for your query."
        
        # Extract key terms and create a smart summary
        high_score_docs = [doc for doc in retrieved_docs if doc['score'] > 0.6]
        doc_count = len(retrieved_docs)
        avg_score = sum(doc['score'] for doc in retrieved_docs) / doc_count
        
        # Create a structured response based on retrieved content
        response_parts = []
        response_parts.append(f"**Clinical Query Analysis**")
        response_parts.append(f"Query: '{query}'")
        response_parts.append(f"Found {doc_count} relevant clinical documents (average relevance: {avg_score:.3f})")
        
        if high_score_docs:
            response_parts.append(f"**High-Relevance Findings**: {len(high_score_docs)} documents with strong clinical relevance")
        
        # Extract key snippets from documents
        clinical_terms = []
        for doc in retrieved_docs[:2]:
            text_lower = doc['text'].lower()
            if 'stroke' in text_lower:
                clinical_terms.append("stroke-related content")
            if 'diagnosis' in text_lower:
                clinical_terms.append("diagnostic information")
            if 'treatment' in text_lower:
                clinical_terms.append("treatment protocols")
            if 'symptom' in text_lower:
                clinical_terms.append("symptom documentation")
        
        if clinical_terms:
            unique_terms = list(set(clinical_terms))
            response_parts.append(f"**Document Content Includes**: {', '.join(unique_terms)}")
        
        response_parts.append("\n**Recommendation**: Review the specific clinical documents below for detailed patient information and medical findings.")
        
        return "\n\n".join(response_parts)
        
    except Exception as e:
        return f"Document retrieval successful. Please review the {len(retrieved_docs)} relevant clinical documents above."

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Clinical RAG System</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Clinical Document Retrieval and Assessment Generation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        max_tokens = st.slider("Maximum answer tokens", 200, 600, 400)  # Increased range
        
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
        
        # Generation mode selection
        st.header("🔧 Generation Mode")
        generation_mode = st.radio(
            "Select generation approach:",
            ["Qwen Model (Recommended)", "Alternative Mode", "Both"],
            index=0
        )
        
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
        if not all([model, index, documents]):
            st.error("❌ Essential components not loaded. Check system status in sidebar.")
            return
            
        with st.spinner("🔍 Retrieving clinical documents..."):
            retrieved_docs = retrieve_documents(query, model, index, documents, top_k=top_k)
            
            if not retrieved_docs:
                st.error("❌ No relevant documents found. Try rephrasing your query.")
                return
            
            # Store retrieval results
            st.session_state.retrieved_docs = retrieved_docs
            st.session_state.current_query = query
            
            # Generate answers based on selected mode
            with st.spinner("🤔 Generating clinical assessment (this may take 20-30 seconds)..."):
                if generation_mode in ["Qwen Model (Recommended)", "Both"] and generator:
                    qwen_answer = generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens)
                    st.session_state.qwen_answer = qwen_answer
                
                if generation_mode in ["Alternative Mode", "Both"]:
                    alt_answer = alternative_generate_answer(query, retrieved_docs)
                    st.session_state.alt_answer = alt_answer
    
    # Display results
    if hasattr(st.session_state, 'qwen_answer') or hasattr(st.session_state, 'alt_answer'):
        st.markdown("---")
        st.markdown("### 💡 AI Clinical Assessment")
        
        # Display Qwen Model answer
        if hasattr(st.session_state, 'qwen_answer'):
            st.markdown("#### 🤖 Qwen Model Generation")
            with st.container():
                if "error" in st.session_state.qwen_answer.lower() or "completed:" in st.session_state.qwen_answer.lower():
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("**Qwen Model Output:**")
                    st.write(st.session_state.qwen_answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="model-output">', unsafe_allow_html=True)
                    st.markdown("**Clinical Assessment:**")
                    st.write(st.session_state.qwen_answer)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Alternative Mode answer
        if hasattr(st.session_state, 'alt_answer'):
            st.markdown("#### 🔄 Alternative Generation")
            with st.container():
                st.markdown('<div class="clinical-answer">', unsafe_allow_html=True)
                st.markdown("**Document-Based Analysis:**")
                st.write(st.session_state.alt_answer)
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
        if hasattr(st.session_state, 'qwen_answer'):
            status = "⚠️" if "error" in st.session_state.qwen_answer.lower() or "completed:" in st.session_state.qwen_answer.lower() else "✅"
            st.metric("Generation Status", status)
        else:
            st.metric("Ready", "🔍")
    
    with col3:
        st.metric("AI Model", "Qwen 2.5B")
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🏥 Clinical RAG System | FAISS + Sentence Transformers + Qwen 2.5-1.5B</p>
        <p>⚠️ For educational and research purposes only. Not for clinical decision making.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
