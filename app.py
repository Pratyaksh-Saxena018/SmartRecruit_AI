# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer, util
# import fitz  # PyMuPDF
# import plotly.graph_objects as go
#
# # --- PAGE CONFIGURATION (The Professional Look) ---
# st.set_page_config(page_title="AI Recruiter Pro", layout="wide")
#
#
# # --- 1. LOAD RESOURCES (Cached for Speed) ---
# @st.cache_resource
# def load_assets():
#     # Load the Brain (Day 2)
#     try:
#         model = SentenceTransformer('my_custom_resume_model')
#     except:
#         st.warning("⚠️ Custom model not found. Using generic model.")
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#
#     # Load the Search Engine (Day 3)
#     index = faiss.read_index("job_index.faiss")
#
#     # Load the Database (Day 3)
#     jobs_db = pd.read_pickle("jobs_metadata.pkl")
#
#     return model, index, jobs_db
#
#
# model, index, jobs_db = load_assets()
#
#
# # --- 2. UTILITY FUNCTIONS ---
# def extract_text_from_pdf(uploaded_file):
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text
#
#
# def find_best_matches(resume_text, k=5):
#     # 1. Vectorize Resume (Tower A)
#     resume_vec = model.encode([resume_text])
#
#     # 2. Search FAISS Index (Efficiency Engine)
#     # This searches 47,000 jobs in < 0.01 seconds
#     D, I = index.search(np.array(resume_vec).astype('float32'), k)
#
#     # 3. Retrieve Job Details
#     matches = jobs_db.iloc[I[0]].copy()
#     matches['score'] = D[0]  # The raw distance score
#     return matches
#
#
# def generate_xai_explanation(resume_text, job_desc):
#     # Novelty: Simple "Skill Gap" Analysis
#     # We extract keywords from Job and check if they exist in Resume
#
#     # Common tech keywords to look for (You can expand this list)
#     tech_keywords = [
#         "python", "java", "c++", "aws", "azure", "docker", "kubernetes",
#         "react", "angular", "node.js", "sql", "nosql", "machine learning",
#         "communication", "leadership", "agile", "scrum", "devops", "django", "flask"
#     ]
#
#     job_lower = str(job_desc).lower()
#     resume_lower = str(resume_text).lower()
#
#     found = []
#     missing = []
#
#     for skill in tech_keywords:
#         if skill in job_lower:
#             if skill in resume_lower:
#                 found.append(skill)
#             else:
#                 missing.append(skill)
#
#     return found, missing
#
#
# # --- 3. THE UI LAYOUT ---
# st.title("🚀 SmartRecruit: AI-Powered Candidate Fitness System")
# st.markdown("### *Siamese Network Architecture • FAISS Efficiency • XAI Explained*")
# st.divider()
#
# # Sidebar: Resume Upload
# with st.sidebar:
#     st.header("1. Candidate Profile")
#     uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
#
#     if uploaded_file:
#         resume_text = extract_text_from_pdf(uploaded_file)
#         st.success("✅ Resume Processed")
#         st.caption(f"Extracted {len(resume_text)} characters.")
#
#         # Mini Radar Chart for "Profile Strength" (Visual Eye Candy)
#         categories = ['Technical', 'Experience', 'Education', 'Soft Skills']
#         values = [8, 6, 7, 9]  # Placeholder values for visual demo
#         fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
#         fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
#                           height=200)
#         st.plotly_chart(fig, use_container_width=True)
#
# # Main Area
# if uploaded_file:
#     st.header("2. Job Market Analysis")
#
#     # The Search Button
#     if st.button("🔍 Find Best Matching Jobs (Scan 47,000+ Listings)"):
#         with st.spinner("Running Siamese Network Inference..."):
#             matches = find_best_matches(resume_text)
#
#         st.subheader("Top 5 Strategic Matches")
#
#         for i, (idx, row) in enumerate(matches.iterrows()):
#             # Calculate a percentage match from the distance score
#             # Lower distance = better match. We invert it for display.
#             match_score = round((1 / (1 + row['score'] * 0.5)) * 100, 1)
#             if match_score > 95: match_score = 95.0  # Cap it
#
#             with st.expander(f"Match #{i + 1}: {row['Job_Titles']} ({match_score}% Fit)"):
#                 col1, col2 = st.columns([2, 1])
#
#                 with col1:
#                     st.markdown(f"**Job Skills:** {row['Skills']}")
#                     st.info("Why this job?")
#
#                     # XAI: Explainability
#                     found, missing = generate_xai_explanation(resume_text, row['search_text'])
#
#                     st.write("✅ **Matched Skills:**")
#                     st.markdown(" ".join([f"`{s}`" for s in found]))
#
#                     if missing:
#                         st.write("❌ **Critical Gaps (Missing):**")
#                         st.markdown(" ".join([
#                                                  f"<span style='color:red; background-color: #ffe6e6; padding: 2px; border-radius: 4px;'>{s}</span>"
#                                                  for s in missing]), unsafe_allow_html=True)
#                     else:
#                         st.success("Perfect Skill Match!")
#
#                 with col2:
#                     # Gauge Chart for Fitness
#                     fig_gauge = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=match_score,
#                         title={'text': "Fitness Score"},
#                         gauge={'axis': {'range': [None, 100]},
#                                'bar': {'color': "green" if match_score > 70 else "orange"}}
#                     ))
#                     fig_gauge.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
#                     st.plotly_chart(fig_gauge, use_container_width=True)
#
# else:
#     st.info("👈 Please upload a resume PDF to begin the analysis.")


import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Recruiter Pro", layout="wide")


# --- 1. LOAD RESOURCES ---
@st.cache_resource
def load_assets():
    try:
        model = SentenceTransformer('my_custom_resume_model')
    except:
        st.warning("⚠️ Custom model not found. Using generic model.")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    index = faiss.read_index("job_index.faiss")
    jobs_db = pd.read_pickle("jobs_metadata.pkl")
    return model, index, jobs_db


model, index, jobs_db = load_assets()


# --- 2. UTILITY FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_match_score(resume_text, job_text):
    # Direct Siamese Match (Resume Vector vs Job Vector)
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)


def generate_xai_explanation(resume_text, job_desc):
    # Simple XAI: Keyword Overlap
    tech_keywords = [
        "python", "java", "c++", "aws", "azure", "docker", "kubernetes",
        "react", "angular", "node.js", "sql", "nosql", "machine learning",
        "communication", "leadership", "agile", "scrum", "devops", "django", "flask",
        "tensorflow", "pytorch", "pandas", "numpy", "git", "linux"
    ]
    job_lower = str(job_desc).lower()
    resume_lower = str(resume_text).lower()

    found = []
    missing = []
    for skill in tech_keywords:
        if skill in job_lower:
            if skill in resume_lower:
                found.append(skill)
            else:
                missing.append(skill)
    return found, missing


def render_gauge(score, idx="0"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fitness Score"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "green" if score > 75 else "orange" if score > 50 else "red"}}
    ))
    fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, width="stretch", key=f"gauge_{idx}")


# --- 3. UI LAYOUT ---
st.title("🚀 SmartRecruit: AI-Powered Candidate Fitness System")

# SIDEBAR: CONTROLS
with st.sidebar:
    st.header("⚙️ Configuration")
    app_mode = st.radio("Select Mode:", ["Market Analysis", "Target Job Match"])

    st.divider()
    st.subheader("1. Candidate")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf", key="resume_up")

    resume_text = ""
    if uploaded_resume:
        resume_text = extract_text_from_pdf(uploaded_resume)
        st.success("✅ Resume Loaded")

        # Radar Chart
        categories = ['Tech', 'Exp', 'Edu', 'Soft']
        values = [8, 7, 6, 9]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=200,
                          margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, width="stretch", key="radar_main")

# MAIN AREA LOGIC
if not uploaded_resume:
    st.info("👈 Start by uploading your Resume in the Sidebar.")
    st.stop()

# --- MODE A: MARKET ANALYSIS (One-to-Many) ---
if app_mode == "Market Analysis":
    st.header("🔍 Market Analysis")
    st.write("Scanning 47,000+ Jobs to find your best matches...")

    if st.button("Run Global Search"):
        # 1. Vectorize Resume
        resume_vec = model.encode([resume_text])
        # 2. Search FAISS
        D, I = index.search(np.array(resume_vec).astype('float32'), 5)

        matches = jobs_db.iloc[I[0]].copy()
        matches['score'] = D[0]

        st.subheader("Top 5 Strategic Matches")
        for i, (idx, row) in enumerate(matches.iterrows()):
            # Convert distance to %
            match_score = round((1 / (1 + row['score'] * 0.5)) * 100, 1)
            if match_score > 95: match_score = 95.0

            with st.expander(f"#{i + 1}: {row['Job_Titles']} ({match_score}% Match)"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write(f"**Skills:** {row['Skills']}")
                    found, missing = generate_xai_explanation(resume_text, row['search_text'])
                    if missing:
                        st.write("❌ **Missing Skills:**")
                        st.markdown(" ".join(
                            [f"<span style='color:red; bg-color: #ffe6e6; padding: 2px;'>{s}</span>" for s in missing]),
                                    unsafe_allow_html=True)
                    else:
                        st.success("All critical skills matched!")
                with c2:
                    render_gauge(match_score, idx=i)

# --- MODE B: TARGET JOB MATCH (One-to-One) ---
elif app_mode == "Target Job Match":
    st.header("🎯 Target Job Match")
    st.write("Upload a specific Job Description (PDF/Text) to check your fitness.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        job_input_method = st.radio("Input Method", ["Paste Text", "Upload PDF"])

        target_job_text = ""
        if job_input_method == "Paste Text":
            target_job_text = st.text_area("Paste Job Description here:", height=300)
        else:
            uploaded_job = st.file_uploader("Upload Job Flyer (PDF)", type="pdf")
            if uploaded_job:
                target_job_text = extract_text_from_pdf(uploaded_job)
                st.success("Job PDF Loaded")

    with col2:
        st.subheader("Fitness Analysis")
        if target_job_text and resume_text:
            if st.button("Analyze Fit"):
                # 1. Calculate Siamese Score
                score = get_match_score(resume_text, target_job_text)

                # 2. XAI Analysis
                found, missing = generate_xai_explanation(resume_text, target_job_text)

                # 3. Display
                render_gauge(score, idx="target")

                st.divider()
                st.markdown("### 🧠 Explainability Report")

                st.success(f"**Matched Skills ({len(found)}):**")
                st.write(", ".join(found) if found else "None detected.")

                st.error(f"**Missing Skills ({len(missing)}):**")
                if missing:
                    st.write(", ".join(missing))
                    st.warning("⚠️ Recommendation: Add these keywords to your resume contextually.")
                else:
                    st.balloons()
                    st.write("No critical skill gaps found!")
        else:
            st.info("Waiting for Resume and Job Description...")