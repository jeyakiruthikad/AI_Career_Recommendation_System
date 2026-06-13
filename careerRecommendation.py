import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import os, re

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f4f6f9; color: #1a202c; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.landing {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 60%, #1a4f7a 100%);
    border-radius: 16px; padding: 2.8rem 3rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.landing::after {
    content:''; position:absolute; top:-40px; right:-40px;
    width:220px; height:220px; background:rgba(255,255,255,0.04); border-radius:50%;
}
.landing-tag {
    display:inline-block; background:rgba(255,255,255,0.12); color:#a8cbf0;
    font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
    padding:0.3rem 0.85rem; border-radius:999px; margin-bottom:1rem;
    border:1px solid rgba(255,255,255,0.15);
}
.landing-title {
    font-family:'DM Sans',sans-serif; font-size:2.2rem; font-weight:700;
    color:#ffffff; line-height:1.2; margin-bottom:0.75rem; letter-spacing:-0.02em;
}
.landing-desc { color:#a8cbf0; font-size:0.97rem; line-height:1.7; max-width:680px; }
.landing-pills { margin-top:1.4rem; display:flex; flex-wrap:wrap; gap:0.5rem; }
.landing-pill {
    background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.18);
    color:#d0e6fa; font-size:0.78rem; padding:0.28rem 0.75rem; border-radius:999px;
}
.step-row { display:flex; gap:0; margin-bottom:1.5rem; }
.step {
    flex:1; text-align:center; padding:0.9rem 0.5rem;
    background:#ffffff; border:1px solid #e2e8f0; position:relative;
}
.step:first-child { border-radius:10px 0 0 10px; }
.step:last-child  { border-radius:0 10px 10px 0; }
.step + .step { border-left:none; }
.step-num { font-family:'DM Sans',sans-serif; font-size:1.2rem; font-weight:700; color:#1e3a5f; }
.step-label { font-size:0.75rem; color:#718096; margin-top:0.2rem; }

.section-header { display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem; padding-bottom:0.6rem; border-bottom:2px solid #e2e8f0; }
.section-icon { background:#1e3a5f; color:#ffffff; border-radius:8px; width:32px; height:32px; display:flex; align-items:center; justify-content:center; font-size:0.9rem; flex-shrink:0; }
.section-title { font-family:'DM Sans',sans-serif; font-size:1.05rem; font-weight:700; color:#1a202c; }
.section-sub { font-size:0.78rem; color:#718096; margin-top:0.1rem; }

.metric-row { display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }
.metric-card { flex:1; min-width:160px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem 1.4rem; border-top:3px solid #1e3a5f; }
.metric-label { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#718096; margin-bottom:0.4rem; }
.metric-value { font-family:'DM Sans',sans-serif; font-size:1.7rem; font-weight:700; color:#1e3a5f; line-height:1.1; }
.metric-sub { font-size:0.78rem; color:#a0aec0; margin-top:0.25rem; }

.panel { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; }

.career-rank { display:flex; align-items:center; gap:1rem; padding:0.85rem 0; border-bottom:1px solid #f0f4f8; }
.career-rank:last-child { border-bottom:none; }
.rank-num { font-family:'DM Sans',sans-serif; font-size:1.1rem; font-weight:700; color:#cbd5e0; width:28px; text-align:center; }
.rank-num.top { color:#1e3a5f; }
.rank-name { font-weight:600; font-size:0.95rem; color:#1a202c; }
.rank-name.top { color:#1e3a5f; }
.rank-badge { font-size:0.68rem; font-weight:600; background:#ebf4ff; color:#1e3a5f; padding:0.15rem 0.55rem; border-radius:999px; margin-left:0.5rem; border:1px solid #bee3f8; }
.rank-field { font-size:0.75rem; color:#718096; margin-top:0.15rem; }
.rank-salary strong { color:#2d6a4f; font-size:0.9rem; display:block; }
.rank-salary { font-size:0.78rem; color:#718096; text-align:right; }

.skill-gap-item { display:flex; align-items:center; justify-content:space-between; padding:0.7rem 0; border-bottom:1px solid #f0f4f8; }
.skill-gap-item:last-child { border-bottom:none; }
.skill-name { font-weight:500; font-size:0.9rem; color:#2d3748; }
.skill-tag { font-size:0.7rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:999px; background:#fff5f5; color:#c53030; border:1px solid #fed7d7; }

.course-item { display:flex; align-items:flex-start; gap:0.85rem; padding:0.75rem 0; border-bottom:1px solid #f0f4f8; }
.course-item:last-child { border-bottom:none; }
.course-idx { font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; color:#ffffff; background:#1e3a5f; border-radius:6px; padding:0.2rem 0.5rem; min-width:2rem; text-align:center; margin-top:2px; flex-shrink:0; }
.course-name { font-size:0.9rem; color:#2d3748; font-weight:500; line-height:1.4; }
.course-skill { font-size:0.75rem; color:#718096; margin-top:0.15rem; }

.readiness-bar-bg { background:#edf2f7; border-radius:999px; height:8px; overflow:hidden; }
.readiness-bar-fill { height:100%; border-radius:999px; }

.suggestion { display:flex; align-items:flex-start; gap:0.7rem; padding:0.6rem 0; border-bottom:1px solid #f0f4f8; font-size:0.88rem; color:#4a5568; }
.suggestion:last-child { border-bottom:none; }
.sug-icon { color:#d69e2e; font-size:0.8rem; margin-top:2px; flex-shrink:0; }

.data-badge { display:inline-block; background:#ebf4ff; color:#1e3a5f; border:1px solid #bee3f8; border-radius:6px; padding:0.2rem 0.6rem; font-size:0.72rem; font-weight:600; margin-right:0.4rem; }

[data-testid="stSidebar"] label { color:#2d3748 !important; font-size:0.875rem !important; font-weight:500 !important; }
[data-testid="stSidebar"] .stMarkdown h4 { color:#1e3a5f; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem; }
[data-testid="stSidebar"] hr { border-color:#e2e8f0; margin:1rem 0; }

.stButton button { background:#1e3a5f !important; color:#ffffff !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:0.92rem !important; padding:0.6rem 1.2rem !important; width:100% !important; }
.stButton button:hover { background:#2d5986 !important; }
.stTextInput input, .stTextArea textarea { border:1px solid #e2e8f0 !important; border-radius:8px !important; font-size:0.88rem !important; background:#fafbfc !important; color:#1a202c !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color:#1e3a5f !important; box-shadow:0 0 0 2px rgba(30,58,95,0.1) !important; }
div[data-testid="stSelectbox"] > div { border:1px solid #e2e8f0 !important; border-radius:8px !important; background:#fafbfc !important; }

.empty-state { text-align:center; padding:4rem 2rem; color:#a0aec0; }
.empty-icon { font-size:3.5rem; margin-bottom:1rem; }
.empty-title { font-size:1.1rem; font-weight:600; color:#718096; margin-bottom:0.4rem; }
.empty-sub { font-size:0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────

# Look in the same folder as app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "tech":    "Tech_Data_Cleaned.csv",
    "job":     "all_job_post.csv",
    "success": "education_career_success.csv",
    "ai":      "AI-based_Career_Recommendation_System.csv",
}

@st.cache_data
def load_all_data():
    def try_load(filename):
        # 1. Same folder as app.py
        path1 = os.path.join(APP_DIR, filename)
        if os.path.exists(path1):
            return pd.read_csv(path1)
        # 2. data/ subfolder inside app folder
        path2 = os.path.join(APP_DIR, "data", filename)
        if os.path.exists(path2):
            return pd.read_csv(path2)
        return None

    missing_files = []
    loaded = {}
    for key, fname in FILES.items():
        df = try_load(fname)
        if df is None:
            missing_files.append(fname)
        loaded[key] = df

    if missing_files:
        st.error(
            f"**CSV files not found.** Make sure these files are in the same folder as `app.py`:\n\n"
            + "\n".join(f"- `{f}`" for f in missing_files)
            + f"\n\n📁 App is running from: `{APP_DIR}`"
        )
        st.stop()

    tech    = loaded["tech"].drop_duplicates()
    tech["Subfields"]              = tech["Subfields"].fillna("")
    tech["Programming Languages "] = tech["Programming Languages "].fillna("")
    tech.fillna("", inplace=True)

    job     = loaded["job"].drop_duplicates().fillna("")
    success = loaded["success"].drop_duplicates().fillna(0)
    ai      = loaded["ai"].drop_duplicates().fillna("")

    return tech, job, success, ai

@st.cache_data
def build_salary_model(success_hash):
    """
    Derive average salary per Field_of_Study from education_career_success.
    GPA is on 4.0 scale in that dataset; we convert user's 10-pt GPA.
    """
    _, _, success, _ = load_all_data()
    # Map field of study → avg salary
    salary_by_field = success.groupby("Field_of_Study")["Starting_Salary"].mean().to_dict()
    # Salary multiplier by GPA bucket
    success["gpa_bucket"] = pd.cut(success["University_GPA"],
                                   bins=[0,3.0,3.3,3.6,3.9,4.1],
                                   labels=["<3.0","3.0-3.3","3.3-3.6","3.6-3.9","3.9+"])
    gpa_mult = success.groupby("gpa_bucket", observed=True)["Starting_Salary"].mean()
    return salary_by_field, gpa_mult

@st.cache_resource
def build_tfidf(_tech_hash):
    tech, _, _, _ = load_all_data()
    tech["career_profile"] = (
        tech["Field"].astype(str) + " " +
        tech["Subfields"].astype(str) + " " +
        tech["Programming Languages "].astype(str) + " " +
        tech["Tools"].astype(str) + " " +
        tech["Skills"].astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    vecs  = tfidf.fit_transform(tech["career_profile"])
    return tfidf, vecs

# ── Skill & Course Logic ──────────────────────────────────────────────────────
priority_skills = [
    "Python","Statistics","Machine Learning","Deep Learning",
    "TensorFlow","PyTorch","Scikit-learn","Feature engineering",
    "Data preprocessing","Model evaluation","Docker","Git",
    "Cloud Architecture","Google Cloud","AWS","MLOps","Spark","Kubernetes","SQL"
]

course_db = {
    "Python":               ["Python for Everybody (Coursera)","Complete Python Bootcamp (Udemy)"],
    "SQL":                  ["SQL for Data Science (Coursera)","Advanced SQL Masterclass (Udemy)"],
    "Machine Learning":     ["Machine Learning Specialization – Andrew Ng (Coursera)","Applied Machine Learning (Coursera)"],
    "Deep Learning":        ["Deep Learning Specialization (Coursera)","FastAI Practical Deep Learning"],
    "TensorFlow":           ["TensorFlow Developer Certificate (Google)"],
    "PyTorch":              ["PyTorch for Deep Learning (Udemy)"],
    "Scikit-learn":         ["Machine Learning with Scikit-learn (DataCamp)"],
    "Feature engineering":  ["Feature Engineering for ML (Udemy)"],
    "Model evaluation":     ["Machine Learning Model Evaluation (DataCamp)"],
    "Data preprocessing":   ["Data Cleaning & Preprocessing (DataCamp)"],
    "Statistics":           ["Statistics for Data Science (Coursera)","Mathematics for ML (Coursera)"],
    "Docker":               ["Docker Essentials (Udemy)","Docker for Developers (LinkedIn Learning)"],
    "Kubernetes":           ["Kubernetes for Beginners (Udemy)","Docker & Kubernetes Bootcamp"],
    "AWS":                  ["AWS Cloud Practitioner","AWS Solutions Architect Associate"],
    "Google Cloud":         ["Google Cloud Fundamentals","Associate Cloud Engineer"],
    "MLOps":                ["MLOps Zoomcamp (DataTalks.Club)"],
    "Spark":                ["Apache Spark for Big Data (Udemy)"],
    "Git":                  ["Git & GitHub Bootcamp (Udemy)","Version Control with Git (Atlassian)"],
    "Cloud Architecture":   ["AWS Solutions Architect","Google Cloud Architect"],
    "Data Analysis":        ["Data Analysis with Python (IBM)","Google Data Analytics Certificate"],
    "Data Visualization":   ["Tableau Fundamentals","Power BI for Data Analysis"],
    "Cybersecurity":        ["Google Cybersecurity Certificate","CompTIA Security+"],
    "Ethical Hacking":      ["Certified Ethical Hacker Preparation"],
    "Linux":                ["Linux Administration (Linux Foundation)","Linux for Developers"],
    "Communication":        ["Business Communication Skills (Coursera)"],
    "Problem-solving":      ["Problem Solving for Developers (Udemy)"],
    "Critical thinking":    ["Critical Thinking & Problem Solving (Coursera)"],
    "Java":                 ["Java Programming Masterclass (Udemy)"],
    "JavaScript":           ["The Complete JavaScript Course (Udemy)"],
    "R":                    ["R Programming (Coursera)","Data Analysis with R (DataCamp)"],
}

def get_requirements(career, tech):
    row = tech[tech["Job roles"] == career]
    if row.empty:
        return []
    reqs = []
    for col in ["Programming Languages ", "Tools", "Skills"]:
        val = str(row.iloc[0][col])
        reqs.extend([x.strip() for x in re.split(r"[,;]+", val) if x.strip()])
    return list(set(reqs))

def skill_gap(career, student_skills, tech):
    required    = get_requirements(career, tech)
    s_lower     = [s.lower().strip() for s in student_skills]
    missing     = [s for s in required if s.lower().strip() not in s_lower]
    technical   = [s for s in missing if s in priority_skills]
    technical   = sorted(technical, key=lambda s: priority_skills.index(s) if s in priority_skills else 99)
    return technical[:5]

def recommend_courses(missing):
    recs = {}
    for skill in missing:
        for key in course_db:
            if key.lower() in skill.lower() or skill.lower() in key.lower():
                recs[skill] = course_db[key]
                break
    return recs

def career_learning_path(career, tech):
    reqs = get_requirements(career, tech)
    path = []
    for s in reqs:
        if s in course_db:
            path.extend(course_db[s])
        else:
            for k in course_db:
                if s.lower() in k.lower() or k.lower() in s.lower():
                    path.extend(course_db[k])
                    break
    return list(dict.fromkeys(path))[:7]

def estimate_salary(field_of_study, gpa_10, internships, salary_by_field, gpa_mult):
    """Use real data from education_career_success.csv to estimate salary."""
    # Map user education field → dataset field
    field_map = {
        "Computer Science": "Computer Science", "Data Science": "Computer Science",
        "Engineering": "Engineering", "Business": "Business",
        "Finance": "Finance", "Marketing": "Marketing",
        "Arts": "Arts", "Psychology": "Psychology",
        "Medicine": "Medicine", "Law": "Law",
        "Education": "Education", "Nursing": "Nursing",
    }
    ds_field = field_map.get(field_of_study, "Computer Science")
    base     = salary_by_field.get(ds_field, 75000)

    # GPA multiplier: convert 10pt → 4pt scale
    gpa_4 = round((gpa_10 / 10) * 4, 2)
    if   gpa_4 >= 3.9: mult = gpa_mult.get("3.9+", base) / base
    elif gpa_4 >= 3.6: mult = gpa_mult.get("3.6-3.9", base) / base
    elif gpa_4 >= 3.3: mult = gpa_mult.get("3.3-3.6", base) / base
    elif gpa_4 >= 3.0: mult = gpa_mult.get("3.0-3.3", base) / base
    else:              mult = gpa_mult.get("<3.0", base) / base

    # Internship bonus (~3% per internship, capped at 5)
    intern_bonus = 1 + min(internships, 5) * 0.03
    estimated    = base * mult * intern_bonus
    return round(estimated, -2)   # round to nearest 100

def calc_readiness(gpa, intern, proj, cert, soft, net):
    return round(min((gpa/10)*35 + min(intern,5)*6 + min(proj,10)*2.5 +
                     min(cert,10)*2 + (soft/100)*15 + (net/100)*10, 100), 2)

def readiness_info(score):
    if score >= 85: return "Excellent", "#276749", "#c6f6d5"
    if score >= 70: return "Good",      "#1e3a5f", "#bee3f8"
    if score >= 55: return "Average",   "#744210", "#fefcbf"
    return "Needs Improvement", "#c53030", "#fed7d7"

def readiness_tips(gpa, intern, proj, cert, soft, net):
    tips = []
    if gpa   < 8:  tips.append("Improve academic performance — aim for a GPA above 8.0")
    if intern < 3: tips.append("Complete at least one more internship for real-world exposure")
    if proj   < 6: tips.append("Build more industry-relevant projects and publish on GitHub")
    if cert   < 5: tips.append("Earn additional professional certifications in your field")
    if soft   < 75:tips.append("Strengthen communication and presentation skills")
    if net    < 70:tips.append("Attend hackathons, workshops, and grow your LinkedIn network")
    return tips

def run_analysis(skills, interests, education, field_of_study,
                 gpa, intern, proj, cert, soft, net,
                 tech, tfidf, vecs, salary_by_field, gpa_mult, ai_df):

    # ── Career Matching (TF-IDF on real Tech_Data_Cleaned) ──
    query = f"{skills} {interests} {education}"
    qv    = tfidf.transform([query])
    sim   = cosine_similarity(qv, vecs)[0]
    top   = sim.argsort()[-3:][::-1]
    careers = [tech.iloc[i]["Job roles"] for i in top]
    fields  = [tech.iloc[i]["Field"]     for i in top]
    best    = careers[0]

    # ── AI dataset cross-check ──
    ai_match = None
    s_lower  = skills.lower()
    for _, row in ai_df.iterrows():
        ai_skills = str(row["Skills"]).lower()
        if any(s.strip() in ai_skills for s in skills.split(",")):
            ai_match = row["Recommended_Career"]
            break

    # ── Skill Gap ──
    s_list  = [s.strip() for s in skills.split(",")]
    missing = skill_gap(best, s_list, tech)

    # ── Salary (from real education data) ──
    salary_usd = estimate_salary(field_of_study, gpa, intern, salary_by_field, gpa_mult)
    salary_inr = round(salary_usd * 83.5, -3)   # approx USD→INR

    # ── Readiness ──
    score = calc_readiness(gpa, intern, proj, cert, soft, net)

    return {
        "careers":       careers,
        "fields":        fields,
        "best":          best,
        "ai_match":      ai_match,
        "salary_usd":    salary_usd,
        "salary_inr":    salary_inr,
        "missing":       missing,
        "skill_courses": recommend_courses(missing),
        "career_path":   career_learning_path(best, tech),
        "score":         score,
        "readiness":     readiness_info(score),
        "tips":          readiness_tips(gpa, intern, proj, cert, soft, net),
    }

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Loading datasets…"):
    tech_df, job_df, success_df, ai_df = load_all_data()
    salary_by_field, gpa_mult          = build_salary_model(0)
    tfidf, vecs                        = build_tfidf(0)

# ── Landing ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="landing">
  <div class="landing-tag">📊 AI-Powered · Career Intelligence</div>
  <div class="landing-title">AI Career Recommendation System</div>
  <div class="landing-desc">
    Enter your academic background, technical skills, and experience profile to receive
    personalised career recommendations, a targeted skill gap analysis, and a curated
    course roadmap<strong style="color:#fff">
    {len(tech_df):,} real job profiles</strong>.
  </div>
  <div class="landing-pills">
    <span class="landing-pill">🎯 Career Matching</span>
    <span class="landing-pill">🔍 Skill Gap Analysis</span>
    <span class="landing-pill">📚 Course Suggestions</span>
    <span class="landing-pill">📈 Readiness Score</span>
    <span class="landing-pill">💰 Data-Driven Salary</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Dataset badges
st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <span style="font-size:0.78rem;color:#718096;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Datasets loaded: </span>
  <span class="data-badge">Tech Profiles · {len(tech_df):,} rows</span>
  <span class="data-badge">Job Posts · {len(job_df):,} rows</span>
  <span class="data-badge">Career Success · {len(success_df):,} rows</span>
  <span class="data-badge">AI Recommendations · {len(ai_df):,} rows</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-row">
  <div class="step"><div class="step-num">01</div><div class="step-label">Enter your profile</div></div>
  <div class="step"><div class="step-num">02</div><div class="step-label">AI matches careers</div></div>
  <div class="step"><div class="step-num">03</div><div class="step-label">View skill gaps</div></div>
  <div class="step"><div class="step-num">04</div><div class="step-label">Get your roadmap</div></div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 👤 Personal Profile")
    skills    = st.text_area("Your Skills (comma-separated)",
                             value="Python, SQL, Machine Learning", height=90)
    interests = st.text_input("Interests / Domain", value="Artificial Intelligence")
    education = st.selectbox("Highest Education",
                             ["Bachelor","Master","PhD","Diploma","Self-taught"])
    field_of_study = st.selectbox("Field of Study",
                                  ["Computer Science","Data Science","Engineering",
                                   "Business","Finance","Marketing","Arts",
                                   "Psychology","Medicine","Law","Education","Nursing"])

    st.markdown("---")
    st.markdown("#### 🎓 Academic Record")
    gpa    = st.slider("GPA (out of 10)", 0.0, 10.0, 8.5, 0.1)
    intern = st.slider("Internships completed", 0, 10, 2)
    proj   = st.slider("Projects built", 0, 20, 5)
    cert   = st.slider("Certifications earned", 0, 15, 3)

    st.markdown("---")
    st.markdown("#### 🤝 Soft Skills & Network")
    soft = st.slider("Soft skills (out of 100)", 0, 100, 80)
    net  = st.slider("Networking score (out of 100)", 0, 100, 70)

    st.markdown("")
    go = st.button("🔍  Analyse My Profile")

# ── Results ───────────────────────────────────────────────────────────────────
if go:
    with st.spinner("Running career analysis…"):
        r = run_analysis(skills, interests, education, field_of_study,
                         gpa, intern, proj, cert, soft, net,
                         tech_df, tfidf, vecs, salary_by_field, gpa_mult, ai_df)

    lvl, lvl_color, lvl_bg = r["readiness"]

    # ── Metric Row ──
    ai_note = f'<div class="metric-sub" style="color:#276749">AI dataset also suggests: {r["ai_match"]}</div>' if r["ai_match"] else '<div class="metric-sub">Based on 1,161 tech profiles</div>'
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">Best Career Match</div>
        <div class="metric-value" style="font-size:1.1rem">{r['best']}</div>
        {ai_note}
      </div>
      <div class="metric-card">
        <div class="metric-label">Estimated Salary (USD)</div>
        <div class="metric-value">${r['salary_usd']:,.0f}</div>
        <div class="metric-sub">≈ ₹{r['salary_inr']:,.0f} / year · based on real data</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Career Readiness</div>
        <div class="metric-value">{r['score']}<span style="font-size:1rem">/100</span></div>
        <div style="margin:0.5rem 0 0.25rem">
          <div class="readiness-bar-bg">
            <div class="readiness-bar-fill" style="width:{r['score']}%;background:{lvl_color}"></div>
          </div>
        </div>
        <div class="metric-sub" style="color:{lvl_color};font-weight:600">{lvl}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Skill Gaps Found</div>
        <div class="metric-value">{len(r['missing'])}</div>
        <div class="metric-sub">Critical skills to bridge</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # Career Recommendations
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🎯</div>
          <div><div class="section-title">Career Recommendations</div>
          <div class="section-sub">Matched from 1,161 real tech job profiles</div></div>
        </div>""", unsafe_allow_html=True)

        rows = ""
        for i, (c, f) in enumerate(zip(r["careers"], r["fields"])):
            badge = '<span class="rank-badge">Best Match</span>' if i == 0 else ""
            rows += f"""
            <div class="career-rank">
              <div class="rank-num {'top' if i==0 else ''}">{i+1}</div>
              <div style="flex:1">
                <div class="rank-name {'top' if i==0 else ''}">{c}{badge}</div>
                <div class="rank-field">{f}</div>
              </div>
            </div>"""
        st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

        # Improvement Suggestions
        if r["tips"]:
            st.markdown("""
            <div class="section-header" style="margin-top:1.2rem">
              <div class="section-icon">💡</div>
              <div><div class="section-title">Improvement Suggestions</div>
              <div class="section-sub">Actions to strengthen your profile</div></div>
            </div>""", unsafe_allow_html=True)
            tips_html = "".join(f'<div class="suggestion"><span class="sug-icon">▲</span>{t}</div>' for t in r["tips"])
            st.markdown(f'<div class="panel">{tips_html}</div>', unsafe_allow_html=True)

    with col2:
        # Skill Gap
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🔍</div>
          <div><div class="section-title">Skill Gap Analysis</div>
          <div class="section-sub">Skills required for your best match role</div></div>
        </div>""", unsafe_allow_html=True)

        if r["missing"]:
            items = "".join(f'<div class="skill-gap-item"><span class="skill-name">{s}</span><span class="skill-tag">Missing</span></div>' for s in r["missing"])
            st.markdown(f'<div class="panel">{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel"><div style="color:#276749;font-weight:600">✅ No critical skill gaps — your profile aligns well!</div></div>', unsafe_allow_html=True)

        # Course Suggestions
        st.markdown("""
        <div class="section-header" style="margin-top:1.2rem">
          <div class="section-icon">📚</div>
          <div><div class="section-title">Course Suggestions</div>
          <div class="section-sub">Curated courses to fill your skill gaps</div></div>
        </div>""", unsafe_allow_html=True)

        if r["skill_courses"]:
            items = ""
            idx = 1
            for skill, courses in r["skill_courses"].items():
                for course in courses:
                    items += f'<div class="course-item"><span class="course-idx">{idx:02d}</span><div><div class="course-name">{course}</div><div class="course-skill">Covers: {skill}</div></div></div>'
                    idx += 1
            st.markdown(f'<div class="panel">{items}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel"><div style="color:#718096;font-size:0.9rem">No additional courses needed.</div></div>', unsafe_allow_html=True)

    # ── Learning Path ──
    st.markdown("""
    <div class="section-header" style="margin-top:0.5rem">
      <div class="section-icon">🗺️</div>
      <div><div class="section-title">Recommended Learning Path</div>
      <div class="section-sub">Complete course roadmap for your best match career</div></div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, course in enumerate(r["career_path"]):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="course-item" style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.6rem">
              <span class="course-idx">{i+1:02d}</span>
              <div class="course-name">{course}</div>
            </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">📊</div>
      <div class="empty-title">Ready to discover your ideal career path?</div>
      <div class="empty-sub">Fill in your profile on the left sidebar and click <strong>Analyse My Profile</strong></div>
    </div>
    """, unsafe_allow_html=True)