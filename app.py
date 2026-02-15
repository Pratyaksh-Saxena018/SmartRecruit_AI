import os
import re
import random
import platform
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import plotly.graph_objects as go
from PIL import Image
import pytesseract

# --- CONFIGURATION ---
st.set_page_config(page_title="SmartRecruit AI", layout="wide")

# --- CRITICAL TESSERACT SETUP ---
if platform.system() == "Windows":
    # ⚠️ This must match where you installed Tesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'my_custom_resume_model')
    index_path = os.path.join(current_dir, 'job_index.faiss')
    db_path = os.path.join(current_dir, 'jobs_metadata.pkl')

    try:
        model = SentenceTransformer(model_path)
    except:
        st.warning(" Custom model not found. Using generic model.")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    index = faiss.read_index(index_path)
    jobs_db = pd.read_pickle(db_path)
    return model, index, jobs_db


model, index, jobs_db = load_assets()

# --- 2. LOGIC CORE ---

SKILL_TAXONOMY = {
    "Frontend": {"html", "css", "javascript", "react", "angular", "vue", "typescript", "redux", "ui/ux", "figma"},
    "Backend": {"node", "express", "django", "flask", "java", "spring", "api", "c#", ".net", "ruby", "go", "php"},
    "AI/Data": {"python", "pandas", "numpy", "tensorflow", "pytorch", "scikit", "machine learning", "ai", "nlp",
                "data analysis", "r"},
    "Database": {"sql", "mysql", "mongodb", "postgresql", "nosql", "oracle", "redis", "elasticsearch", "pl/sql"},
    "Cloud/Ops": {"aws", "azure", "gcp", "docker", "kubernetes", "cloud", "jenkins", "linux", "cicd", "terraform"}
}

# Old (outdated/legacy) skill → current/modern equivalent(s). Used to weigh experience: seasoned + legacy = boost; newbie + legacy = penalize.
LEGACY_SKILL_MAP = {
    # Frontend legacy → modern
    "jquery": "react",
    "jquery ui": "react",
    "angularjs": "angular",
    "backbone": "react",
    "ember": "react",
    "knockout": "react",
    "extjs": "react",
    "bootstrap 3": "bootstrap",
    "grunt": "webpack",
    "gulp": "webpack",
    "bower": "npm",
    # Backend / runtime
    "php 5": "php",
    "asp classic": ".net",
    "vb6": ".net",
    "vb.net": ".net",
    "coldfusion": "node",
    "perl": "python",
    "ruby on rails 3": "ruby",
    "struts": "spring",
    "ejb": "spring",
    "servlet": "spring",
    "jsp": "spring",
    "hibernate 3": "spring",
    "soap": "api",
    "wsdl": "api",
    "corba": "api",
    # Data / infra
    "hadoop": "python",
    "mapreduce": "python",
    "pig": "python",
    "hive": "sql",
    "sqoop": "python",
    "flume": "python",
    "mahout": "scikit",
    "weka": "scikit",
    "sas": "python",
    "stata": "python",
    # DevOps / version control
    "svn": "git",
    "cvs": "git",
    "ant": "cicd",
    "maven": "cicd",
    "cruisecontrol": "jenkins",
    # Other
    "flash": "javascript",
    "actionscript": "javascript",
    "flex": "react",
    "silverlight": "javascript",
}
# Normalize keys for lookup (single space, lowercase)
LEGACY_SKILL_MAP = {k.strip().lower(): v for k, v in LEGACY_SKILL_MAP.items()}


def extract_job_skills(job_text):
    """
    Extract required skill set from job source: DB search_text (Market Analysis) or job description text.
    Returns (skills_set, domains_set). If job uses a generic domain name (e.g. "frontend", "backend"),
    that domain is included for intelligent question generation.
    """
    if not job_text:
        return set(), set()
    text = job_text.lower()
    skills = set()
    domains = set()
    for domain, kws in SKILL_TAXONOMY.items():
        if domain.lower() in text or any(w in text for w in domain.lower().split("/")):
            domains.add(domain)
        for kw in kws:
            if kw in text:
                skills.add(kw)
                domains.add(domain)
    for legacy in LEGACY_SKILL_MAP:
        if legacy in text:
            skills.add(legacy)
            modern = LEGACY_SKILL_MAP[legacy]
            for domain, kws in SKILL_TAXONOMY.items():
                if modern in kws:
                    domains.add(domain)
                    break
    return skills, domains


def extract_resume_skills(resume_text):
    """Extract skill set from resume (taxonomy + legacy). Returns (skills_set, domains_set)."""
    if not resume_text:
        return set(), set()
    text = resume_text.lower()
    skills = set()
    domains = set()
    for domain, kws in SKILL_TAXONOMY.items():
        if domain.lower() in text or any(w in text for w in domain.lower().split("/")):
            domains.add(domain)
        for kw in kws:
            if kw in text:
                skills.add(kw)
                domains.add(domain)
    for legacy in LEGACY_SKILL_MAP:
        if legacy in text:
            skills.add(legacy)
            modern = LEGACY_SKILL_MAP[legacy]
            for domain, kws in SKILL_TAXONOMY.items():
                if modern in kws:
                    domains.add(domain)
                    break
    return skills, domains


def get_common_domains_and_skills(job_text, resume_text):
    """
    Match job required skills with resume skills. Returns domains that have overlap
    (for question selection). If a generic domain name appears in job/resume, we use
    that domain for question generation. Returns (common_domains, job_skills, resume_skills).
    """
    job_skills, job_domains = extract_job_skills(job_text)
    resume_skills, resume_domains = extract_resume_skills(resume_text)
    common_skills = job_skills & resume_skills
    # Domains where at least one skill matches (job wants it and resume has it)
    common_domains = set()
    for domain, kws in SKILL_TAXONOMY.items():
        job_has = any(kw in job_skills for kw in kws)
        resume_has = any(kw in resume_skills for kw in kws)
        if job_has and resume_has:
            common_domains.add(domain)
    # If no overlap, use all domains that job needs (so we still ask questions)
    if not common_domains and job_domains:
        common_domains = job_domains
    if not common_domains:
        common_domains = set(SKILL_TAXONOMY.keys())
    return list(common_domains), job_skills, resume_skills


def get_experience_level(resume_text):
    """
    Infer experience level from resume (0–10). Used for realtime weighting:
    high experience + legacy skills → weigh more; newbie + legacy → weigh less.
    """
    if not resume_text:
        return 4.0
    text = resume_text.lower()
    # Years of experience (e.g. "5 years", "10+ years")
    years = 0
    year_patterns = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp\.?|in\s+)", text, re.I)
    if year_patterns:
        years = max(int(x) for x in year_patterns)
    elif re.search(r"\d+\+?\s*years?", text):
        nums = re.findall(r"(\d+)\+?\s*years?", text)
        if nums:
            years = max(int(x) for x in nums)
    # Role-based level (overrides or complements years)
    if any(k in text for k in ["principal", "architect", "distinguished", "fellow"]):
        return 10.0
    if any(k in text for k in ["lead", "manager", "head of", "director", "vp", "cto"]):
        return 9.0
    if any(k in text for k in ["senior", "sr.", "staff", "senior engineer"]):
        return 8.0
    if any(k in text for k in ["mid", "mid-level", "associate", "engineer ii", "engineer 2"]):
        return 6.0
    if any(k in text for k in ["junior", "jr.", "entry", "graduate", "fresher", "intern"]):
        return 3.0
    # Map years to 0–10 scale
    if years >= 10:
        return 9.0
    if years >= 7:
        return 8.0
    if years >= 5:
        return 7.0
    if years >= 3:
        return 6.0
    if years >= 1:
        return 5.0
    return 4.0


def classify_skill_freshness(resume_lower, job_lower, exact_matches, conceptual_matches):
    """
    Classify matched skills (exact + conceptual) as legacy vs current relative to job.
    Returns (legacy_matched_set, current_matched_set) for realtime score weighting.
    - Legacy: job wants a skill and candidate matched via an outdated skill (or job asks for legacy).
    - Current: job wants modern skill and candidate has it.
    """
    all_matched = set((s.lower() if isinstance(s, str) else s) for s in (exact_matches + conceptual_matches))
    legacy_matched = set()
    current_matched = set()
    modern_to_legacy = {v.lower(): k for k, v in LEGACY_SKILL_MAP.items()}
    for s in all_matched:
        s_low = s.lower() if isinstance(s, str) else s
        if s_low in LEGACY_SKILL_MAP:
            legacy_matched.add(s_low)
        elif s_low in modern_to_legacy and modern_to_legacy[s_low] in resume_lower:
            legacy_matched.add(s_low)
        else:
            current_matched.add(s_low)
    return legacy_matched, current_matched


def extract_content(uploaded_file):
    """HD OCR Reader (3x Zoom for better accuracy)"""
    text = ""
    try:
        if "image" in uploaded_file.type:
            try:
                img = Image.open(uploaded_file).convert('L')
                text = pytesseract.image_to_string(img)
            except:
                return ""
        elif "pdf" in uploaded_file.type:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                t = page.get_text()
                if len(t) > 5:
                    text += t
                else:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
                        text += pytesseract.image_to_string(img)
                    except:
                        pass
    except:
        return ""
    return text


def calculate_profile_scores(text):
    """Radar Chart Logic (Education, Experience, Tech, Soft Skills)"""
    text = text.lower()
    scores = {}

    # Education
    edu = 5
    if any(k in text for k in ['phd', 'doctorate']):
        edu = 10
    elif any(k in text for k in ['master', 'm.tech', 'mba']):
        edu = 9
    elif any(k in text for k in ['b.tech', 'b.e', 'bachelor']):
        edu = 8
    scores['Education'] = edu

    # Experience
    exp = 4
    if any(k in text for k in ['principal', 'architect', 'lead', 'manager']):
        exp = 10
    elif any(k in text for k in ['senior', 'sr.']):
        exp = 8
    elif any(k in text for k in ['mid', 'associate']):
        exp = 6
    scores['Experience'] = exp

    # Tech Stack
    tech_cnt = sum(
        1 for k in ['python', 'java', 'c++', 'sql', 'aws', 'react', 'node', 'docker', 'linux', 'git'] if k in text)
    scores['Tech Stack'] = min(3 + (tech_cnt * 0.8), 10)

    # Soft Skills
    soft_cnt = sum(1 for k in ['team', 'communication', 'agile', 'leadership', 'problem solving'] if k in text)
    scores['Soft Skills'] = min(4 + (soft_cnt * 1.5), 10)

    return scores


def analyze_gap_logic(resume_text, job_text):
    """
    Intelligent Gap Analysis:
    Splits skills into Direct Match, Conceptual Match (AI inferred), and Missing.
    """
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    exact_matches = set()
    conceptual_matches = set()
    missing_skills = set()

    # Identify skills needed for this job (current + legacy if job mentions them)
    target_skills = set()
    for cat, kws in SKILL_TAXONOMY.items():
        for kw in kws:
            if kw in job_lower:
                target_skills.add(kw)
    for legacy_skill in LEGACY_SKILL_MAP:
        if legacy_skill in job_lower:
            target_skills.add(legacy_skill)

    # AI Comparison
    resume_emb = model.encode(resume_text, convert_to_tensor=True)

    for skill in target_skills:
        if skill in resume_lower:
            exact_matches.add(skill)
        else:
            skill_emb = model.encode(skill, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(skill_emb, resume_emb).item()
            # If similarity > 0.25, it implies the *concept* exists in resume
            if similarity > 0.25:
                conceptual_matches.add(skill)
            else:
                missing_skills.add(skill)

    return list(exact_matches), list(conceptual_matches), list(missing_skills)


def calculate_smart_score(resume_text, job_text):
    """
    Weighted Score (realtime):
    50% Semantic Match + 50% Skill Coverage, then experience–skill-freshness adjustment.
    - Experienced candidate with outdated skills → experience weighed more (boost).
    - Newbie with outdated skills → weighed less (penalize).
    """
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_text, convert_to_tensor=True)
    semantic_score = util.pytorch_cos_sim(emb1, emb2).item()

    exact, conceptual, missing = analyze_gap_logic(resume_text, job_text)
    total_skills = len(exact) + len(conceptual) + len(missing)

    if total_skills == 0:
        coverage_score = 0.5
    else:
        coverage_score = (len(exact) + (len(conceptual) * 0.8)) / total_skills

    base_score = (semantic_score * 0.5) + (coverage_score * 0.5)

    # --- Realtime experience × skill-freshness weighting ---
    experience_level = get_experience_level(resume_text)
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()
    legacy_matched, current_matched = classify_skill_freshness(resume_lower, job_lower, exact, conceptual)
    total_matched = len(legacy_matched) + len(current_matched)

    experience_weight = 1.0
    if total_matched > 0 and len(legacy_matched) > 0:
        legacy_ratio = len(legacy_matched) / total_matched
        # High experience (e.g. 7+): weigh experience more — seasoned with legacy still valuable
        if experience_level >= 7.0:
            experience_weight = 1.0 + (0.15 * legacy_ratio)  # up to +15% for highly experienced
        # Low experience (e.g. < 5): newbie with old stack — weigh less
        elif experience_level < 5.0:
            experience_weight = 1.0 - (0.20 * legacy_ratio)  # up to -20% for newbie + legacy
        # Mid (5–7): slight penalty so they're not boosted like seniors
        else:
            experience_weight = 1.0 - (0.05 * legacy_ratio)

    weighted_score = base_score * experience_weight
    return round(min(max(weighted_score * 100, 0), 99.9), 1)


# --- QUIZ: Two-level score — Level 1 = weighted score; Level 2 = quiz on common skills ---
# Question format: domain, type (coding/theory), difficulty (junior/mid/senior), question, options, correct_index (0-based)
# Prefer coding; use HOTS theory when only theory matters. All topics covered.
QUIZ_QUESTION_BANK = [
    # Frontend — coding
    {"domain": "Frontend", "type": "coding", "difficulty": "junior", "q": "What will `console.log(typeof []);` output in JavaScript?", "opts": ["object", "array", "undefined", "string"], "correct": 0},
    {"domain": "Frontend", "type": "coding", "difficulty": "junior", "q": "In React, which hook runs after every render including the first?", "opts": ["useEffect with empty deps", "useLayoutEffect", "useEffect with no deps array", "useMemo"], "correct": 2},
    {"domain": "Frontend", "type": "coding", "difficulty": "mid", "q": "Given `const [a, ...rest] = [1,2,3,4];` what is `rest`?", "opts": ["[2,3,4]", "[1,2,3,4]", "1", "[1,2,3]"], "correct": 0},
    {"domain": "Frontend", "type": "coding", "difficulty": "mid", "q": "In CSS, which has higher specificity: `.a.b` or `#x`?", "opts": ["#x", ".a.b", "Same", "Depends on order"], "correct": 0},
    {"domain": "Frontend", "type": "coding", "difficulty": "senior", "q": "What is the time complexity of React's reconciliation when the tree structure changes?", "opts": ["O(n³) in worst case", "O(n) with heuristics", "O(log n)", "O(n²)"], "correct": 1},
    # Frontend — HOTS theory
    {"domain": "Frontend", "type": "theory", "difficulty": "mid", "q": "Why might you avoid inline object creation in JSX for React props?", "opts": ["Breaks SSR", "Creates new reference each render, breaking memoization", "Not valid JSX", "Slower parsing"], "correct": 1},
    {"domain": "Frontend", "type": "theory", "difficulty": "senior", "q": "How does the Virtual DOM improve performance over direct DOM manipulation in complex UIs?", "opts": ["Eliminates reflows", "Batches updates and minimizes real DOM operations via diffing", "Uses Web Workers", "Caches all DOM nodes"], "correct": 1},
    # Backend — coding
    {"domain": "Backend", "type": "coding", "difficulty": "junior", "q": "In Python, what does `'x' in {'x': 1}` return?", "opts": ["True", "False", "1", "KeyError"], "correct": 0},
    {"domain": "Backend", "type": "coding", "difficulty": "junior", "q": "What HTTP method is typically used for creating a resource in REST?", "opts": ["POST", "GET", "PUT", "CREATE"], "correct": 0},
    {"domain": "Backend", "type": "coding", "difficulty": "mid", "q": "In Node.js, what will `setImmediate(() => console.log(1)); setTimeout(() => console.log(2), 0);` print first?", "opts": ["1", "2", "Non-deterministic", "Neither runs"], "correct": 2},
    {"domain": "Backend", "type": "coding", "difficulty": "mid", "q": "In Django, which runs first: `middleware process_request` or `view`?", "opts": ["process_request", "view", "Same request", "Depends on order in MIDDLEWARE"], "correct": 0},
    {"domain": "Backend", "type": "coding", "difficulty": "senior", "q": "In a distributed system, why might you choose idempotency keys for POST requests?", "opts": ["To cache responses", "To safely retry without duplicate side effects", "To compress payloads", "To enforce ordering"], "correct": 1},
    # Backend — HOTS theory
    {"domain": "Backend", "type": "theory", "difficulty": "senior", "q": "When would you prefer eventual consistency over strong consistency?", "opts": ["Bank transfers", "When availability and partition tolerance matter more than immediate consistency", "Medical records", "Never"], "correct": 1},
    # AI/Data — coding
    {"domain": "AI/Data", "type": "coding", "difficulty": "junior", "q": "In pandas, what does `df.dropna(axis=1)` do?", "opts": ["Drops rows with NaN", "Drops columns with NaN", "Fills NaN with 0", "Drops first column"], "correct": 1},
    {"domain": "AI/Data", "type": "coding", "difficulty": "junior", "q": "What is the default activation in the output layer for binary classification in neural nets?", "opts": ["ReLU", "Sigmoid", "Softmax", "Tanh"], "correct": 1},
    {"domain": "AI/Data", "type": "coding", "difficulty": "mid", "q": "In numpy, `a = np.array([1,2,3]); b = a; b[0]=99`. What is `a[0]`?", "opts": ["1", "99", "Error", "None"], "correct": 1},
    {"domain": "AI/Data", "type": "coding", "difficulty": "mid", "q": "For imbalanced classes, which metric is more informative than accuracy?", "opts": ["F1 or AUC-ROC", "Accuracy", "MSE", "R²"], "correct": 0},
    {"domain": "AI/Data", "type": "coding", "difficulty": "senior", "q": "What does gradient clipping prevent in RNNs?", "opts": ["Underfitting", "Exploding gradients", "Slow convergence", "Overfitting"], "correct": 1},
    # AI/Data — HOTS theory
    {"domain": "AI/Data", "type": "theory", "difficulty": "senior", "q": "Why might a model with high training accuracy but low validation accuracy benefit from dropout?", "opts": ["Increases capacity", "Reduces overfitting by preventing co-adaptation", "Speeds training", "Fixes data leakage"], "correct": 1},
    # Database — coding
    {"domain": "Database", "type": "coding", "difficulty": "junior", "q": "In SQL, which clause filters rows after grouping?", "opts": ["WHERE", "HAVING", "GROUP BY", "FILTER"], "correct": 1},
    {"domain": "Database", "type": "coding", "difficulty": "junior", "q": "What does a UNIQUE constraint ensure?", "opts": ["No NULLs", "All values in column are distinct (and one NULL allowed in SQL)", "Primary key", "Index only"], "correct": 1},
    {"domain": "Database", "type": "coding", "difficulty": "mid", "q": "In a B-tree index, why is range query on the leading column efficient?", "opts": ["Keys are hashed", "Keys are sorted, so range is contiguous", "No disk I/O", "Cache only"], "correct": 1},
    {"domain": "Database", "type": "coding", "difficulty": "mid", "q": "What isolation level prevents dirty reads but not non-repeatable reads?", "opts": ["READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"], "correct": 1},
    {"domain": "Database", "type": "coding", "difficulty": "senior", "q": "When would you choose a covering index over a regular index?", "opts": ["When table is small", "When all selected columns are in the index (avoid table lookup)", "When writes dominate", "Never"], "correct": 1},
    # Database — HOTS theory
    {"domain": "Database", "type": "theory", "difficulty": "senior", "q": "Why might you denormalize in a read-heavy analytics DB?", "opts": ["To reduce storage", "To reduce joins and improve read latency at cost of write complexity", "To enforce constraints", "To avoid indexes"], "correct": 1},
    # Cloud/Ops — coding
    {"domain": "Cloud/Ops", "type": "coding", "difficulty": "junior", "q": "In Docker, what does `COPY . /app` do in a Dockerfile?", "opts": ["Copies host . into container /app", "Copies container root to /app", "Copies /app to host", "Invalid syntax"], "correct": 0},
    {"domain": "Cloud/Ops", "type": "coding", "difficulty": "junior", "q": "Which Kubernetes resource defines desired replica count for pods?", "opts": ["Pod", "Deployment", "Service", "ReplicaSet only"], "correct": 1},
    {"domain": "Cloud/Ops", "type": "coding", "difficulty": "mid", "q": "In AWS, which service is object storage?", "opts": ["EBS", "S3", "EFS", "EC2"], "correct": 1},
    {"domain": "Cloud/Ops", "type": "coding", "difficulty": "mid", "q": "What does `docker run --rm` do?", "opts": ["Restart on failure", "Remove container after it exits", "Read-only filesystem", "Run as root"], "correct": 1},
    {"domain": "Cloud/Ops", "type": "coding", "difficulty": "senior", "q": "In K8s, what is the main role of a readiness probe?", "opts": ["Restart unhealthy pods", "Decide when to send traffic to the pod", "Scale down", "Replace node"], "correct": 1},
    # Cloud/Ops — HOTS theory
    {"domain": "Cloud/Ops", "type": "theory", "difficulty": "senior", "q": "Why use blue-green deployment over rolling?", "opts": ["Uses less resources", "Instant rollback by switching traffic back", "No load balancer needed", "Faster first deploy"], "correct": 1},
]

# Experience level to difficulty band for question selection (prefer matching band, then neighbors)
def _experience_to_difficulty(exp_level):
    if exp_level >= 7:
        return ["senior", "mid"]
    if exp_level >= 5:
        return ["mid", "senior", "junior"]
    return ["junior", "mid"]


def select_quiz_questions(common_domains, experience_level, n=10):
    """
    Select 10 questions: cover all common domains, experience-appropriate, prefer coding over theory.
    Returns list of questions with dynamic choice allocation (options shuffled, correct index updated).
    """
    difficulties = _experience_to_difficulty(experience_level)
    pool = [q for q in QUIZ_QUESTION_BANK if q["domain"] in common_domains and q["difficulty"] in difficulties]
    if len(pool) < n:
        pool = list(QUIZ_QUESTION_BANK)
    pool.sort(key=lambda x: (x["domain"], 0 if x["type"] == "coding" else 1, x["difficulty"]))
    selected = []
    # One question per domain first (coding preferred) to cover all topics
    for domain in common_domains:
        for q in pool:
            if q["domain"] == domain and q not in selected:
                selected.append(q)
                break
    # Fill to n, prefer coding
    for q in pool:
        if len(selected) >= n:
            break
        if q not in selected:
            selected.append(q)
    selected = selected[:n]
    # Dynamic choice allocation: shuffle options and update correct index
    out = []
    for q in selected:
        opts = list(q["opts"])
        correct_val = opts[q["correct"]]
        random.shuffle(opts)
        new_correct = opts.index(correct_val)
        out.append({"domain": q["domain"], "type": q["type"], "q": q["q"], "opts": opts, "correct": new_correct})
    return out


def score_quiz_answers(questions, answers):
    """Score quiz: answers is list of selected option indices (0-based). Returns 0–100."""
    if not questions or len(answers) != len(questions):
        return 0.0
    correct = sum(1 for i, a in enumerate(answers) if a == questions[i]["correct"])
    return round((correct / len(questions)) * 100.0, 1)


def combined_two_level_score(weighted_score, quiz_score, quiz_weight=0.3):
    """Level 1 (weighted) + Level 2 (quiz). quiz_weight=0.3 => 70% weighted + 30% quiz."""
    return round((1 - quiz_weight) * weighted_score + quiz_weight * quiz_score, 1)


# --- UI LAYOUT ---
st.title("SmartRecruit AI: Advanced Ranking System")

with st.sidebar:
    st.header("Candidate Profile")
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "png", "jpg"], key="resume_up")

    resume_text = ""
    if uploaded_resume:
        with st.spinner("Processing (HD OCR)..."):
            resume_text = extract_content(uploaded_resume)
        if resume_text:
            st.success("Resume Processed")

            # --- FEATURE 1: 4-AXIS RADAR CHART ---
            scores = calculate_profile_scores(resume_text)
            fig = go.Figure(data=go.Scatterpolar(
                r=list(scores.values()),
                theta=list(scores.keys()),
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=300,
                title="Candidate Profile Map",
                margin=dict(t=40, b=20, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    app_mode = st.radio("Select Mode:", ["Market Analysis", "Target Job Match"])

if not uploaded_resume:
    st.info("Please upload a resume to begin.")
    st.stop()

# --- MODE A: MARKET ANALYSIS ---
# Flow: (1) Quiz first → (2) Score quiz → (3) Run job search separately → (4) Results to main screen with combined score
if app_mode == "Market Analysis":
    st.header("Global Market Analysis")
    role = st.selectbox("Role Preference:", ["Any", "Software Engineer", "Data Scientist", "Manager"])

    # Initialize market quiz state (resume-based domains; no job yet)
    if "market_quiz_submitted" not in st.session_state:
        st.session_state.market_quiz_submitted = False
    if "market_quiz_questions" not in st.session_state:
        _, resume_domains = extract_resume_skills(resume_text)
        common_domains = list(resume_domains) if resume_domains else list(SKILL_TAXONOMY.keys())
        exp = get_experience_level(resume_text)
        st.session_state.market_quiz_questions = select_quiz_questions(common_domains, exp, 10)

    # Step 1: Show quiz first; hide job search until quiz is submitted
    if not st.session_state.market_quiz_submitted:
        st.subheader("Step 1: Skill Quiz")
        st.caption("Complete the quiz first. Your score will be added to the weighted fit score for each job. Then run the job search.")
        questions = st.session_state.market_quiz_questions
        for i, q in enumerate(questions):
            st.radio(
                f"**Q{i+1}** [{q['domain']}] {q['q']}",
                options=q["opts"],
                key=f"market_quiz_q_{i}",
                index=None,
            )
        if st.button("Submit Quiz", key="market_submit_quiz"):
            answers = []
            for i, q in enumerate(questions):
                val = st.session_state.get(f"market_quiz_q_{i}")
                if val is not None:
                    answers.append(q["opts"].index(val))
                else:
                    answers.append(-1)
            if -1 not in answers:
                st.session_state.market_quiz_score = score_quiz_answers(questions, answers)
                st.session_state.market_quiz_submitted = True
                st.rerun()
            else:
                st.warning("Please answer all 10 questions before submitting.")
    else:
        # Step 2: Quiz done — show job search and results on main screen
        st.success(f"Quiz completed. Your quiz score: **{st.session_state.market_quiz_score}%** — it will be combined with each job's fit score.")
        if st.button("Run Job Search", key="market_run_search"):
            query = f"{role} {role} {resume_text}" if role != "Any" else resume_text
            vec = model.encode([query])
            D, I = index.search(np.array(vec).astype('float32'), 5)
            matches = jobs_db.iloc[I[0]].copy()
            st.session_state.market_matches = matches
            st.rerun()

        if st.session_state.get("market_matches") is not None:
            matches = st.session_state.market_matches
            quiz_score = st.session_state.market_quiz_score
            st.subheader("Results: Top 5 Strategic Matches")
            st.caption(f"Combined score = 70% weighted fit + 30% quiz ({quiz_score}%)")

            for i, (idx, row) in enumerate(matches.iterrows()):
                job_text = row["search_text"]
                level1 = calculate_smart_score(resume_text, job_text)
                combined = combined_two_level_score(level1, quiz_score, 0.3)
                exact, conceptual, missing = analyze_gap_logic(resume_text, job_text)

                with st.expander(f"Rank #{i + 1}: {row['Job_Titles']} — Combined: {combined}%"):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown("### Gap Analysis")
                        if exact: st.success(f"**Direct Matches:** {', '.join(exact)}")
                        if conceptual: st.info(f"**Conceptual Matches:** {', '.join(conceptual)}")
                        if missing: st.error(f"**Missing:** {', '.join(missing)}")
                    with c2:
                        st.metric("Level 1 (weighted)", f"{level1}%")
                        st.metric("Quiz", f"{quiz_score}%")
                        st.metric("Combined", f"{combined}%")

# --- MODE B: TARGET JOB MATCH ---
# Flow: (1) User enters job → (2) Quiz appears first → (3) Score quiz → (4) Display results (fit + combined) on main screen
elif app_mode == "Target Job Match":
    st.header("Target Job Fit Check")

    job_input_text = st.text_area("Paste Job Description:")
    uploaded_job = st.file_uploader("Or Upload Job Description (PDF/Image)", type=["pdf", "png", "jpg"])

    if uploaded_job:
        extracted = extract_content(uploaded_job)
        if extracted:
            job_input_text = extracted
            st.success("Job Description Extracted from File!")

    # Start analysis: compute level1 and prepare quiz (don't show results yet)
    if st.button("Start Analysis", key="target_start"):
        if not job_input_text:
            st.error("Please provide a Job Description.")
        else:
            score = calculate_smart_score(resume_text, job_input_text)
            exact, conceptual, missing = analyze_gap_logic(resume_text, job_input_text)
            st.session_state.target_level1 = score
            st.session_state.target_job_text = job_input_text
            st.session_state.target_exact = exact
            st.session_state.target_conceptual = conceptual
            st.session_state.target_missing = missing
            st.session_state.target_quiz_submitted = False
            common_domains, _, _ = get_common_domains_and_skills(job_input_text, resume_text)
            exp = get_experience_level(resume_text)
            st.session_state.target_quiz_questions = select_quiz_questions(common_domains, exp, 10)
            st.rerun()

    # Show quiz first when we have target job and quiz not yet submitted
    if job_input_text and st.session_state.get("target_job_text") == job_input_text and not st.session_state.get("target_quiz_submitted", False):
        st.subheader("Step 1: Skill Quiz")
        st.caption("Complete the quiz. Your score will be combined with the job fit score; then results will be shown.")
        questions = st.session_state.target_quiz_questions
        for i, q in enumerate(questions):
            st.radio(
                f"**Q{i+1}** [{q['domain']}] {q['q']}",
                options=q["opts"],
                key=f"target_quiz_q_{i}",
                index=None,
            )
        if st.button("Submit Quiz", key="target_submit_quiz"):
            answers = []
            for i, q in enumerate(questions):
                val = st.session_state.get(f"target_quiz_q_{i}")
                if val is not None:
                    answers.append(q["opts"].index(val))
                else:
                    answers.append(-1)
            if -1 not in answers:
                st.session_state.target_quiz_score = score_quiz_answers(questions, answers)
                st.session_state.target_quiz_submitted = True
                st.session_state.target_combined = combined_two_level_score(
                    st.session_state.target_level1, st.session_state.target_quiz_score, 0.3
                )
                st.rerun()
            else:
                st.warning("Please answer all 10 questions before submitting.")

    # Step 2: After quiz submitted — display results on main screen
    elif job_input_text and st.session_state.get("target_job_text") == job_input_text and st.session_state.get("target_quiz_submitted"):
        score = st.session_state.target_level1
        exact = st.session_state.get("target_exact", [])
        conceptual = st.session_state.get("target_conceptual", [])
        missing = st.session_state.get("target_missing", [])
        quiz_score = st.session_state.target_quiz_score
        combined = st.session_state.target_combined

        st.subheader("Results")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=combined,
                title={'text': "Combined Score (70% Fit + 30% Quiz)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "green" if combined > 75 else "orange" if combined > 50 else "red"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Level 1 (weighted fit)", f"{score}%")
            st.metric("Quiz score", f"{quiz_score}%")
            st.metric("Combined", f"{combined}%")
        st.subheader("Gap Analysis")
        if exact: st.write(f"**Direct Match:** {', '.join(exact)}")
        if conceptual: st.write(f"**Inferred (Logical):** {', '.join(conceptual)}")
        if missing: st.write(f"**Missing:** {', '.join(missing)}")
        if not missing:
            st.success("Perfect Match!")