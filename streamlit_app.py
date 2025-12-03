@st.cache_resource
def load_qwen_model():
    """DISABLED for Streamlit Cloud compatibility"""
    st.warning("⚠️ Qwen model disabled - Streamlit Cloud memory limits")
    return None, None

def generate_answer(query, retrieved_docs, tokenizer, generator, max_tokens=400):
    """Document-only version for Streamlit Cloud"""
    try:
        if not retrieved_docs:
            return "No relevant documents found."
        
        # Create a smart document-based summary
        summary_parts = []
        summary_parts.append(f"**Clinical Query Analysis**: '{query}'")
        summary_parts.append(f"**Documents Found**: {len(retrieved_docs)} relevant clinical records")
        
        # Show relevance scores
        high_relevance = [d for d in retrieved_docs if d['score'] > 0.6]
        if high_relevance:
            summary_parts.append(f"**High-Relevance Documents**: {len(high_relevance)} with strong clinical correlation")
        
        # Extract clinical themes
        clinical_themes = set()
        for doc in retrieved_docs[:3]:
            text = doc['text'].lower()
            if any(term in text for term in ['stroke', 'ischemic', 'hemorrhagic']):
                clinical_themes.add("cerebrovascular events")
            if any(term in text for term in ['diagnosis', 'diagnosed']):
                clinical_themes.add("diagnostic information")
            if any(term in text for term in ['treatment', 'therapy', 'medication']):
                clinical_themes.add("treatment protocols")
            if any(term in text for term in ['symptom', 'presenting', 'complains']):
                clinical_themes.add("clinical presentation")
        
        if clinical_themes:
            summary_parts.append(f"**Document Themes**: {', '.join(clinical_themes)}")
        
        summary_parts.append("\n**Recommendation**: Review the retrieved clinical documents below for comprehensive patient information.")
        
        return "\n\n".join(summary_parts)
        
    except Exception as e:
        return f"Document analysis completed. {len(retrieved_docs)} relevant records found."
