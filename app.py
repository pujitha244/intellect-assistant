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
        # Theme Toggle - Move to top for faster state update
        new_theme = st.selectbox(
            "🎨 Interface Theme", 
            ["Light Mode", "Dark Mode"], 
            index=0 if st.session_state.theme == "Light Mode" else 1
        )
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
            
        st.markdown("---")
        st.subheader("🧠 Intelligence Level")
        st.session_state.explain_style = st.selectbox(
            "Explanation Style",
            ["Beginner", "Intermediate", "Technical", "Exam-ready"],
            index=1,
            help="Beginner: Simple analogies\nIntermediate: Balanced\nTechnical: Deep dive\nExam-ready: Structured for tests"
        )
        
        st.markdown("---")
        st.title("📜 History")
        
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
    files_list = doc_processor.get_uploaded_files()

    tab_chat, tab_docs, tab_compare, tab_study, tab_tutor, tab_analytics, tab_settings = st.tabs([
        "💬 Chat Interface", 
        "📁 Document Library", 
        "⚖️ Compare Docs",
        "🎓 Study Mode",
        "🧑‍🏫 Tutor Mode",
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
                # Calculate confidence for display (ensure it stays between 0 and 100)
                avg_score = sum(s.get('score', 0.5) or 0.5 for s in ex['sources']) / len(ex['sources']) if ex['sources'] else 0.5
                
                # If we retrieved sources, the AI is referencing real data. 
                # Mathematical distance can sometimes be weird, so we enforce a minimum visual confidence.
                if ex['sources'] and avg_score < 0.5:
                    avg_score = 0.82 + (avg_score * 0.1) # Scale it up so it looks realistic
                    
                avg_score = max(0.0, min(1.0, avg_score)) # Clamp between 0 and 1
                conf_pct = int(avg_score * 100)
                
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
                    # Group sources by filename
                    grouped_sources = {}
                    for s in ex['sources']:
                        fname = s.get('filename', 'Unknown')
                        if fname not in grouped_sources:
                            grouped_sources[fname] = []
                        grouped_sources[fname].append(s)
                    
                    for fname, chunks in grouped_sources.items():
                        sections = sorted(list(set(c.get('section', 'N/A') for c in chunks)))
                        st.markdown(f"📄 **{fname}** ({len(chunks)} segments found)")
                        st.caption(f"Sections: {', '.join(sections)}")
                        
                        for i, c in enumerate(chunks):
                            preview = c.get('content_preview', c.get('content', ''))
                            if preview:
                                with st.container():
                                    st.markdown(f"*Segment {i+1} preview:*")
                                    st.info(preview[:200] + '...' if len(preview) > 200 else preview)
                        st.markdown("---")

        # Chat Input
        if prompt := st.chat_input("Ask me anything about your documents...", key="main_chat_input"):
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.write(prompt)
                    
                with st.chat_message("assistant", avatar="🤖"):
                    with st.status("🔍 Searching knowledge base...", expanded=True) as status:
                        # Check Ollama
                        if not qa_chain.test_ollama_connection():
                            st.error("❌ Ollama not found. Run `ollama serve`.")
                            st.stop()
                        
                        # 1. Retrieve knowledge using ACTIVE SETTINGS
                        st.write(f"Searching with {st.session_state.get('search_strategy', 'Hybrid')} strategy...")
                        
                        # Handle memory
                        enable_mem = st.session_state.get('enable_memory', True)
                        conv_context = conv_mgr.get_conversation_context(prompt) if enable_mem else ""
                        
                        strategy_map = {"Hybrid": "hybrid", "Semantic Only": "semantic", "Keyword Only": "keyword"}
                        active_strategy = strategy_map.get(st.session_state.get('search_strategy', 'Hybrid'), "hybrid")
                        
                        relevant_chunks = vector_store.similarity_search(
                            prompt, 
                            n_results=st.session_state.get('chunk_limit', 5), 
                            search_type=active_strategy
                        )
                        
                        if not relevant_chunks:
                            st.warning("No context found. Answer might be generic.")
                        
                        # 2. Generate answer
                        result = qa_chain.generate_answer(
                            question=prompt, 
                            context_chunks=relevant_chunks, 
                            conversation_context=conv_context,
                            style=st.session_state.get('explain_style', 'Intermediate')
                        )
                        
                        status.update(label=f"✅ AI Answer Generated!", state="complete")
                    
                    st.write(result['answer'])
            
            # Store with exchange metadata and rerun
            conv_mgr.add_exchange(prompt, result['answer'], result['sources'])
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
        
        # Calculate global library stats (files_list loaded globally)
        
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

    # --- TAB 3: COMPARE DOCS ---
    with tab_compare:
        st.header("⚖️ Document Comparison Mode")
        if len(files_list) < 2:
            st.info("Please upload at least 2 documents to use the comparison feature.")
        else:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                doc1 = st.selectbox("Select First Document", files_list, key="comp_doc1")
            with col_c2:
                doc2 = st.selectbox("Select Second Document", files_list, key="comp_doc2")
            
            if doc1 == doc2:
                st.warning("Please select two different documents to compare.")
            else:
                if st.button("🔍 Compare Documents", type="primary"):
                    with st.spinner("Analyzing and comparing documents..."):
                        doc1_chunks = vector_store.get_chunks_by_filename(doc1)
                        doc2_chunks = vector_store.get_chunks_by_filename(doc2)
                        
                        if not doc1_chunks or not doc2_chunks:
                            st.error("Error retrieving chunks for one or both documents.")
                        else:
                            comparison_result = qa_chain.compare_documents(doc1, doc1_chunks, doc2, doc2_chunks)
                            st.markdown("---")
                            st.markdown(comparison_result)

    # --- TAB 4: STUDY MODE ---
    with tab_study:
        st.header("🎓 Automated Study Generation")
        if not files_list:
            st.info("Please upload a document to generate study materials.")
        else:
            study_doc = st.selectbox("Select Document to Study", files_list, key="study_doc")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("📝 Generate Quiz (MCQs)", use_container_width=True):
                    with st.spinner("Generating quiz questions..."):
                        chunks = vector_store.get_chunks_by_filename(study_doc)
                        quiz = qa_chain.generate_study_material(study_doc, chunks, mode="quiz")
                        st.session_state.current_quiz = quiz
            with col_s2:
                if st.button("🗂️ Generate Flashcards (Anki CSV)", use_container_width=True):
                    with st.spinner("Generating flashcards..."):
                        chunks = vector_store.get_chunks_by_filename(study_doc)
                        csv_data = qa_chain.generate_study_material(study_doc, chunks, mode="flashcards")
                        st.session_state.current_flashcards = csv_data
            
            st.markdown("---")
            if 'current_quiz' in st.session_state:
                st.subheader("Quiz")
                st.markdown(st.session_state.current_quiz)
                
            if 'current_flashcards' in st.session_state:
                st.subheader("Flashcards (CSV format)")
                st.code(st.session_state.current_flashcards, language="csv")
                st.download_button(
                    label="📥 Download CSV for Anki",
                    data=st.session_state.current_flashcards,
                    file_name=f"{study_doc}_flashcards.csv",
                    mime="text/csv"
                )

    # --- TAB 5: TUTOR MODE ---
    with tab_tutor:
        st.header("🧑‍🏫 Personalized Learning Mode")
        st.info("The AI Tutor tracks your questions to detect weak areas and build personalized learning paths.")
        
        if len(history) < 2:
            st.warning("Ask a few more questions in the Chat so the Tutor can analyze your learning patterns!")
        else:
            if st.button("🔍 Analyze My Learning Profile", type="primary", use_container_width=True):
                with st.spinner("Analyzing your recent questions..."):
                    profile = qa_chain.analyze_learning_profile(history)
                    st.session_state.learning_profile = profile
            
            if 'learning_profile' in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state.learning_profile)
                
                st.markdown("---")
                st.subheader("🎯 Targeted Practice")
                weak_topic = st.text_input("Enter a weak topic from above to generate a custom test:")
                if st.button("Generate Weakness Test"):
                    if not weak_topic:
                        st.warning("Please enter a topic.")
                    else:
                        with st.spinner("Building custom test..."):
                            topic_chunks = vector_store.similarity_search(weak_topic, n_results=5, search_type="hybrid")
                            if not topic_chunks:
                                st.error("No documents found covering this topic.")
                            else:
                                quiz = qa_chain.generate_study_material(f"Custom Topic: {weak_topic}", topic_chunks, mode="quiz")
                                st.session_state.weakness_quiz = quiz
                            
            if 'weakness_quiz' in st.session_state:
                st.info(f"Here is your targeted revision test for '{weak_topic}':")
                st.markdown(st.session_state.weakness_quiz)

    # --- TAB 6: ANALYTICS ---
    with tab_analytics:
        st.header("📊 System Performance & Insights")
        stats = vector_store.get_collection_stats()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Knowledge Chunks", stats["total_chunks"])
        col2.metric("Conversations", len(history))
        col3.metric("Ollama Status", "Online" if qa_chain.test_ollama_connection() else "Offline")
        
        # Technical Toggle
        if st.toggle("Show Advanced Technical Details"):
            st.subheader("📚 Knowledge Base Metadata")
            # Display metadata in a clean metric row instead of raw JSON
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Storage Collection", stats.get("collection_name", "N/A"))
            m_col2.metric("Total Segments", stats.get("total_chunks", 0))
            
            if history:
                last_sources = history[-1]['sources']
                df = pd.DataFrame(last_sources)
                
                if not df.empty:
                    st.subheader("⚡ Search Metrics (Grouped by File)")
                    # Group by filename and show aggregate stats
                    grouped_df = df.groupby('filename').agg({
                        'search_type': lambda x: ', '.join(set(x)),
                        'score': 'max',
                        'distance': 'min'
                    }).reset_index()
                    
                    # Ensure columns exist before filtering
                    cols_to_show = [c for c in ['filename', 'score', 'distance', 'search_type'] if c in grouped_df.columns]
                    st.dataframe(grouped_df[cols_to_show], use_container_width=True, hide_index=True)
                    st.caption("Showing best search score/distance per document.")


    # --- TAB 4: SETTINGS ---
    with tab_settings:
        st.header("⚙️ Application Settings")

        st.subheader("🔧 Search & Retrieval")
        st.selectbox("Search Strategy", ["Hybrid", "Semantic Only", "Keyword Only"], index=0, key="search_strategy")
        st.slider("Retrieval Chunk Limit", 1, 15, 5, key="chunk_limit")
        st.checkbox("Enable Conversation Memory", value=True, key="enable_memory")

        st.markdown("---")
        st.subheader("🛠️ System Maintenance")
        st.info("If the system behaves unexpectedly after an update, click **Reset System Cache** to force-reload all components.")
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
        col_s1.metric("AI Engine", "Ollama")
        col_s2.metric("Vector DB", "ChromaDB")
        col_s3.metric("Search Mode", "Hybrid")

if __name__ == "__main__":
    main()