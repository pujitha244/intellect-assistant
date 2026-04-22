import streamlit as st
import os
import time
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
from pathlib import Path

# --- STANDALONE DEPENDENCY BRIDGE ---
# We have copied required libraries (fpdf) and models (en_core_web_sm) 
# into the local 'libs' folder to make this project 100% standalone.
local_libs_path = str(Path(__file__).parent / "libs")
if local_libs_path not in sys.path:
    sys.path.append(local_libs_path)

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.qa_chain import QAChain
from src.conversation_manager import ConversationManager
from config import PAGE_TITLE, PAGE_ICON, UPLOADS_DIR

# --- PAGE CONFIG ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light Mode'

def inject_custom_css():
    theme = st.session_state.theme
    
    if theme == 'Dark Mode':
        # Dark Theme Tokens
        bg_color = "#0e1117"
        text_color = "#ffffff"
        sec_text = "#c9d1d9"
        card_bg = "#161b22"
        border_color = "#30363d"
        bot_bubble = "#1e252e"
        user_bubble = "linear-gradient(135deg, #007bff, #0056b3)"
        user_text = "#ffffff"
        input_bg = "#161b22"
    else:
        # Light Theme Tokens
        bg_color = "#ffffff"
        text_color = "#000000"
        sec_text = "#333333"
        card_bg = "#f5f5f5"
        border_color = "#d0d7de"
        bot_bubble = "#f0f2f6"
        user_bubble = "linear-gradient(135deg, #007bff, #0056b3)"
        user_text = "#ffffff"
        input_bg = "#f0f2f6"

    st.markdown(f"""
    <style>
        /* Main container styling */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {card_bg} !important;
            border-right: 1px solid {border_color};
        }}
        
        /* Global Text color overrides */
        .stMarkdown, .stText, h1, h2, h3, p, li, label, .stSelectbox label, .stSlider label {{
            color: {text_color} !important;
        }}
        
        .stCaption {{
            color: {sec_text} !important;
        }}

        /* Chat Bubble Styling */
        .chat-bubble {{
            padding: 16px 20px;
            border-radius: 18px;
            max-width: 80%;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        
        .user-bubble {{
            background: {user_bubble};
            color: {user_text} !important;
            border-bottom-right-radius: 4px;
        }}
        
        .bot-bubble {{
            background-color: {bot_bubble};
            color: {text_color} !important;
            border-bottom-left-radius: 4px;
            border: 1px solid {border_color};
        }}
        
        /* Widget Styling for Visibility */
        .stSelectbox div[data-baseweb="select"] {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        div[data-baseweb="popover"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}

        /* Confidence Meter Styling */
        .confidence-meter {{
            height: 4px;
            width: 100%;
            background-color: {border_color};
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff4b4b, #ffa421, #00d4ff);
        }}

        /* Container styling */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {card_bg} !important;
            border: 1px solid {border_color};
            border-radius: 12px;
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {text_color};
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 2px solid #007bff !important;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# --- UTILITY FUNCTIONS ---

def question_type_detector(query: str) -> str:
    """Standalone: Detect question intent for NLP routing"""
    q = query.lower().strip()
    if any(w in q for w in ['what is', 'define', 'meaning of', 'definition']):
        return "DEFINITION"
    elif any(w in q for w in ['summary', 'main idea', 'summarize', 'overview', 'theme']):
        return "THEME"
    elif any(w in q for w in ['why', 'how', 'explain', 'reason']):
        return "REASONING"
    elif any(w in q for w in ['who', 'when', 'where', 'list', 'name']):
        return "FACT"
    return "FACT"

def extract_keywords(text: str, top_n: int = 5) -> list:
    """Standalone: Extract top keywords using TF-IDF"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100, ngram_range=(1, 1))
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        indices = scores.argsort()[::-1]
        return [feature_names[i] for i in indices[:top_n]]
    except:
        return []

def sanitize_for_pdf(text):
    """Remove characters that Arial/Latin-1 cannot handle to prevent crashes"""
    if not text: return ""
    return text.encode('ascii', 'ignore').decode('ascii')

def get_pdf_bytes(question, answer, sources):
    if not FPDF_AVAILABLE:
        return None
    try:
        from fpdf.enums import XPos, YPos
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_margins(left=15, top=15, right=15)
        pdf.add_page()
        
        # Set auto page break
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Fixed width for safe rendering (A4 is 210mm - 30mm margins = 180mm)
        safe_w = 180
        
        # Header
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(safe_w, 10, text="Knowledge Assistant Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Question Section
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(safe_w, 10, text="Question:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", '', 11)
        pdf.multi_cell(safe_w, 8, text=sanitize_for_pdf(question))
        pdf.ln(5)
        
        # Answer Section
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(safe_w, 10, text="Answer:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", '', 11)
        pdf.multi_cell(safe_w, 8, text=sanitize_for_pdf(answer))
        pdf.ln(10)
        
        # Sources Section
        pdf.set_font("helvetica", 'B', 10)
        pdf.cell(safe_w, 10, text="Sources Used:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", '', 9)
        for i, s in enumerate(sources):
            source_txt = f"[{i+1}] {s['filename']} (Section: {s.get('section', 'N/A')})"
            pdf.multi_cell(safe_w, 6, text=sanitize_for_pdf(source_txt))
        
        # Return as bytes
        # In fpdf2, output() returns a bytearray; Streamlit requires bytes()
        return bytes(pdf.output())
    except Exception as e:
        st.error(f"PDF Rendering Error: {str(e)}")
        return None

def get_full_history_txt(history):
    out = "CONVERSATION HISTORY\n" + "="*30 + "\n\n"
    for ex in history:
        out += f"TIME: {ex['timestamp']}\n"
        out += f"USER: {ex['question']}\n"
        out += f"AI: {ex['answer']}\n"
        out += "-"*30 + "\n\n"
    return out

# --- INITIALIZATION ---
@st.cache_resource
def initialize_components():
    return DocumentProcessor(), VectorStore(), QAChain()

def main():
    # Load components
    doc_processor, vector_store, qa_chain = initialize_components()
    
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    
    conv_mgr = st.session_state.conversation_manager
    history = conv_mgr.get_history()

    # --- SIDEBAR: HISTORY & CONTROLS ---
    with st.sidebar:
        st.title("📜 History")
        
        # Theme Toggle - Move to top for faster state update
        new_theme = st.selectbox(
            "🎨 Interface Theme", 
            ["Light Mode", "Dark Mode"], 
            index=0 if st.session_state.theme == "Light Mode" else 1
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
            
        # NLP Insights Toggle
        st.session_state.show_nlp_insights = st.toggle("🧪 Show NLP Intelligence", value=True)
        
        st.markdown("---")
        
        if not history:
            st.info("No conversations yet.")
        else:
            # Search in history
            h_search = st.text_input("🔍 Search history...", key="hist_search")
            
            # Display history items
            for i, ex in enumerate(reversed(history)):
                if h_search.lower() in ex['question'].lower() or h_search.lower() in ex['answer'].lower():
                    # Clickable history item
                    label = f"{'⭐ ' if ex.get('favorite') else ''}{ex['question'][:30]}..."
                    if st.button(label, key=f"hist_{i}", use_container_width=True):
                        st.session_state.current_question = ex['question']
            
            st.markdown("---")
            if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
                conv_mgr.clear_history()
                st.rerun()
            
            st.download_button(
                "📥 Export Full Chat",
                data=get_full_history_txt(history),
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d')}.txt",
                use_container_width=True
            )

    # --- MAIN UI TABS ---
    st.title("🧠 Intellect Assistant")
    tab_chat, tab_docs, tab_analytics, tab_settings = st.tabs([
        "💬 Chat Interface", 
        "📁 Document Library", 
        "📊 System Analytics", 
        "⚙️ Settings"
    ])

    # --- TAB 1: CHAT ---
    with tab_chat:
        # Chat display container
        chat_container = st.container()
        
        with chat_container:
            for i, ex in enumerate(history):
                # User Message
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                        <div class="chat-bubble user-bubble">
                            <b>You:</b><br>{ex['question']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Assistant Message
                # Calculate confidence for display
                avg_score = sum(s.get('score', 0.5) for s in ex['sources']) / len(ex['sources']) if ex['sources'] else 0.5
                conf_pct = min(100, int(avg_score * 100))
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                        <div class="chat-bubble bot-bubble">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                <b>Assistant</b>
                                <span style="font-size: 10px; opacity: 0.7;">Confidence: {conf_pct}%</span>
                            </div>
                            <div class="confidence-meter"><div class="confidence-fill" style="width: {conf_pct}%;"></div></div>
                            <div style="margin-top: 10px;">{ex['answer']}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # 🧪 NLP Insights (Conditional)
                if st.session_state.get('show_nlp_insights', True):
                    # Extract from metadata
                    meta = ex.get('metadata', {})
                    q_type = meta.get('q_type', 'GENERAL')
                    keywords = meta.get('keywords', [])
                    
                    st.markdown(f"""
                    <div style="margin-left: 50px; margin-bottom: 20px;">
                        <span style="font-size: 12px; color: #007bff; font-weight: bold;">🧪 NLP ANALYTICS:</span>
                        <span style="font-size: 12px; opacity: 0.8;"> Type: <code>{q_type}</code> | Keywords: <code>{", ".join(keywords) if keywords else "None"}</code></span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action Buttons (Download/Favorite/Source)
                # Keep these as streamlit buttons for functionality
                col_act1, col_act2, col_act3, col_act4 = st.columns([1, 1, 1, 2])
                
                with col_act1:
                    if st.button("⭐", key=f"fav_{i}", help="Mark as Favorite"):
                        ex['favorite'] = not ex.get('favorite', False)
                        st.rerun()
                
                with col_act2:
                    if FPDF_AVAILABLE:
                        pdf_data = get_pdf_bytes(ex['question'], ex['answer'], ex['sources'])
                        if pdf_data:
                            st.download_button("📄 PDF", data=pdf_data, file_name=f"answer_{i}.pdf", key=f"dl_pdf_{i}")
                    else:
                        st.button("📄 PDF (Disabled)", key=f"dl_pdf_dis_{i}", help="Install fpdf2 to enable PDF export")
                
                with col_act3:
                    st.download_button("📝 TXT", data=ex['answer'], file_name=f"answer_{i}.txt", key=f"dl_txt_{i}")
                
                # Sources Expander
                with st.expander("📚 View Sources"):
                    for s in ex['sources']:
                        st.markdown(f"**{s.get('filename','Unknown')}** | Section: `{s.get('section','N/A')}`")
                        preview = s.get('content_preview', s.get('content', ''))
                        if preview:
                            st.info(preview[:300] + '...' if len(preview) > 300 else preview)
                        st.markdown("---")

        # Chat Input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add placeholders for loading
            with st.chat_message("user", avatar="👤"):
                st.write(prompt)
                
            with st.chat_message("assistant", avatar="🤖"):
                with st.status("🔍 Searching knowledge base...", expanded=True) as status:
                    st.write("Retreiving relevant documents...")
                    
                    # Check Ollama
                    if not qa_chain.test_ollama_connection():
                        st.error("❌ Ollama not found. Run `ollama serve`.")
                        st.stop()
                    
                    # Process with NLP Logic
                    st.write("Analyzing intent & retrieving knowledge...")
                    q_type = question_type_detector(prompt)
                    enhanced_query = conv_mgr.get_enhanced_query(prompt)
                    relevant_chunks = vector_store.similarity_search(enhanced_query)
                    
                    if not relevant_chunks:
                        st.warning("No context found. Answer might be generic.")
                    
                    st.write("Synthesizing answer with specialized NLP strategy...")
                    result = qa_chain.generate_answer(
                        question=prompt, 
                        context_chunks=relevant_chunks, 
                        conversation_context=conv_mgr.get_conversation_context(prompt),
                        question_type=q_type
                    )
                    
                    # Extract keywords from the final answer
                    ans_keywords = extract_keywords(result['answer'])
                    
                    status.update(label=f"✅ {q_type} Answer Generated!", state="complete")
                
                st.write(result['answer'])
                # Store with NLP metadata
                nlp_meta = {"q_type": q_type, "keywords": ans_keywords}
                conv_mgr.add_exchange(prompt, result['answer'], result['sources'], nlp_meta)
                st.rerun()

    # --- TAB 2: DOCUMENTS ---
    with tab_docs:
        st.header("📂 Document Management")
        
        # Upload Area
        uploaded_files = st.file_uploader(
            "Add new knowledge sources", 
            type=['pdf', 'docx', 'txt', 'md'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process & Index Documents", type="primary"):
                with st.status("Indexing documents...") as status:
                    for f in uploaded_files:
                        st.write(f"Processing `{f.name}`...")
                        text = doc_processor.process_uploaded_file(f)
                        if text:
                            chunks = doc_processor.chunk_text(text, f.name)
                            vector_store.add_documents(chunks)
                    status.update(label="Processing Complete!", state="complete")
                st.rerun()

        st.markdown("---")
        st.subheader("📊 Library Insights")
        
        # Calculate global library stats
        files_list = doc_processor.get_uploaded_files()
        
        # Safely call get_all_documents
        all_chunks = []
        if hasattr(vector_store, 'get_all_documents'):
            all_chunks = vector_store.get_all_documents()
        
        if all_chunks:
            # 💡 ADVANCED: Dynamic Topic Modeling with LLM
            if 'dynamic_topics' not in st.session_state or st.session_state.get('last_lib_count') != len(all_chunks):
                with st.spinner("🧠 LLM is analyzing document topics..."):
                    def name_topic(keywords):
                        prompt = f"Give a professional 3-word title for a knowledge topic characterized by these words: {', '.join(keywords)}. Return ONLY the title."
                        try:
                            # Use the existing QA chain's LLM
                            response = qa_chain.llm.invoke(prompt)
                            return response.strip().replace('"', '')
                        except:
                            return f"Topic: {keywords[0].title()}"

                    st.session_state.dynamic_topics = doc_processor.generate_dynamic_topics(all_chunks, llm_callback=name_topic)
                    st.session_state.last_lib_count = len(all_chunks)
            
            stats = doc_processor.get_document_stats(all_chunks, topic_model=st.session_state.dynamic_topics)
            
            c_ins1, c_ins2, c_ins3 = st.columns(3)
            c_ins1.metric("Unique Files", len(files_list))
            c_ins2.metric("Knowledge Nodes", stats["total_chunks"])
            c_ins3.metric("Vocabulary (Chars)", f"{stats['total_characters']:,}")
            
            # Topic Distribution Visualization
            st.write("**Topic Distribution (Machine Learning Clustering):**")
            # Sort and limit to top 5
            sorted_topics = dict(sorted(stats['topic_distribution'].items(), key=lambda item: item[1], reverse=True)[:5])
            topic_df = pd.DataFrame(list(sorted_topics.items()), columns=['Topic Name', 'Density'])
            st.bar_chart(topic_df.set_index('Topic Name'), use_container_width=True)
            
            # Metadata Highlights
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.info(f"📅 **Contextual Timelines:** Found in {stats['has_dates']} knowledge segments.")
            with col_h2:
                st.success(f"🔢 **Data Intelligence:** Found in {stats['has_numbers']} segments.")
        else:
            st.info("Upload documents to activate AI Topic Discovery.")

        st.markdown("---")
        st.subheader("Current Library")
        
        if not files_list:
            st.info("No documents indexed yet.")
        else:
            # Display files in a clean table-like layout
            for fname in files_list:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.markdown(f"📄 **{fname}**")
                    if c2.button("🔄 Reprocess", key=f"repr_{fname}"):
                        st.toast(f"Reprocessing {fname}...")
                    if c3.button("🗑️ Delete", key=f"del_{fname}", type="secondary"):
                        if vector_store.delete_documents_by_filename(fname):
                            f_path = UPLOADS_DIR / fname
                            if f_path.exists(): f_path.unlink()
                            st.rerun()

    # --- TAB 3: ANALYTICS ---
    with tab_analytics:
        st.header("📊 System Performance & Insights")
        stats = vector_store.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Knowledge Chunks", stats["total_chunks"])
        col2.metric("Conversations", len(history))
        col3.metric("Ollama Status", "Online" if qa_chain.test_ollama_connection() else "Offline")
        
        # Technical Toggle
        if st.toggle("Show Advanced Technical Details"):
            st.subheader("🔍 Vector Store Metadata")
            st.json(stats)
            
            if history:
                st.subheader("⚡ Search Metrics (Latest)")
                last_sources = history[-1]['sources']
                df = pd.DataFrame(last_sources)
                if not df.empty:
                    st.dataframe(df[['filename', 'score', 'search_type']])

        # NLP Techniques Section
        st.markdown("---")
        st.subheader("🧪 NLP Techniques (spaCy Analysis)")
        if history:
            last_exchange = history[-1]
            q_text = last_exchange['question']
            a_text = last_exchange['answer']
            
            # Perform NLP Analysis on both
            q_meta = doc_processor.get_nlp_metadata(q_text)
            a_meta = doc_processor.get_nlp_metadata(a_text)
            
            if "error" in q_meta:
                st.warning(q_meta["error"])
            
            # --- Analysis for Question ---
            st.markdown("#### 👤 Latest Question Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Concepts & Entities (NER + Ranking):**")
                if q_meta['entities']:
                    # Render importance badge colors
                    imp_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "N/A": "⚪"}
                    rows = [{"Concept": e.get("concept",""), "Type": e.get("type",""), "Importance": f"{imp_colors.get(e.get('importance',''),'⚪')} {e.get('importance','')}"}
                            for e in q_meta['entities']]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No concepts detected in the question.")
            with col2:
                st.write("**POS Tagging (Meaningful Tokens):**")
                if q_meta['pos_tags']:
                    st.dataframe(pd.DataFrame(q_meta['pos_tags']), use_container_width=True, hide_index=True)
                else:
                    st.info("No meaningful tokens found.")

            st.markdown("---")
            
            # --- Analysis for Answer ---
            st.markdown("#### 🤖 Latest Answer Analysis")
            col3, col4 = st.columns(2)
            with col3:
                st.write("**Concepts & Entities (NER + Ranking):**")
                if a_meta['entities']:
                    imp_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "N/A": "⚪"}
                    rows = [{"Concept": e.get("concept",""), "Type": e.get("type",""), "Importance": f"{imp_colors.get(e.get('importance',''),'⚪')} {e.get('importance','')}"}
                            for e in a_meta['entities']]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No concepts detected in the answer.")
            with col4:
                st.write("**POS Tagging (Meaningful Tokens):**")
                if a_meta['pos_tags']:
                    st.dataframe(pd.DataFrame(a_meta['pos_tags']), use_container_width=True, hide_index=True)
                else:
                    st.info("No meaningful tokens found.")
        else:
            st.info("Ask a question first to see NLP analytics (POS & Entities)!")


    # --- TAB 4: SETTINGS ---
    with tab_settings:
        st.header("⚙️ Application Settings")

        st.subheader("🔧 Search & Retrieval")
        st.selectbox("Search Strategy", ["Hybrid (Recommended)", "Semantic Only", "Keyword Only"], index=0)
        st.slider("Retrieval Chunk Limit", 1, 10, 5)
        st.checkbox("Enable Conversation Memory", value=True)

        st.markdown("---")
        st.subheader("🛠️ System Maintenance")
        st.info("If you see 'AttributeError' or the system behaves unexpectedly after an update, click **Reset System Cache** to force-reload all NLP components.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🔄 Reset System Cache", type="primary", use_container_width=True):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state.clear()
                st.success("✅ Cache cleared! The page will reload with fresh components.")
                st.rerun()
        with col_m2:
            if st.button("🔥 Factory Reset (Delete Everything)", type="secondary", use_container_width=True):
                if vector_store.clear_collection():
                    for f in UPLOADS_DIR.glob("*"): f.unlink()
                    conv_mgr.clear_history()
                    st.warning("All data has been wiped.")
                    st.rerun()

        st.markdown("---")
        st.subheader("📋 System Status")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("NLP Engine", "spaCy")
        col_s2.metric("Vector DB", "ChromaDB")
        col_s3.metric("Search Mode", "Hybrid")

if __name__ == "__main__":
    main()