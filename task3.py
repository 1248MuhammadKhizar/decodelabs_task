import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_roles = [
    {
        "role": "Data Scientist",
        "skills": "python sql machine learning data analysis statistics numpy pandas scikit-learn visualization"
    },
    {
        "role": "DevOps Engineer",
        "skills": "aws docker kubernetes git ci cd linux bash terraform ansible cloud automation"
    },
    {
        "role": "Backend Developer",
        "skills": "java python sql apis rest databases django flask nodejs postgresql mongodb"
    },
    {
        "role": "Frontend Developer",
        "skills": "javascript html css react vuejs typescript nodejs figma ui ux responsive design"
    },
    {
        "role": "Machine Learning Engineer",
        "skills": "python tensorflow pytorch deep learning neural networks nlp computer vision numpy pandas gpu"
    },
    {
        "role": "Cloud Architect",
        "skills": "aws azure gcp cloud networking security terraform infrastructure automation scalability"
    },
    {
        "role": "Cybersecurity Analyst",
        "skills": "network security penetration testing linux firewalls encryption threat analysis siem python"
    },
    {
        "role": "Full Stack Developer",
        "skills": "javascript python react nodejs sql mongodb html css rest apis docker git"
    },
    {
        "role": "Data Engineer",
        "skills": "python sql spark hadoop etl pipelines kafka airflow postgresql data warehousing cloud"
    },
    {
        "role": "Mobile App Developer",
        "skills": "swift kotlin java react native flutter ios android firebase rest apis ui"
    },
    {
        "role": "AI Research Engineer",
        "skills": "python pytorch tensorflow mathematics statistics optimization transformers nlp reinforcement learning papers"
    },
    {
        "role": "Systems Administrator",
        "skills": "linux windows networking bash powershell servers virtualization vmware docker monitoring backup"
    },
]

df = pd.DataFrame(job_roles)

print("=" * 55)
print("   DECODELABS — TECH STACK RECOMMENDER (Project 3)")
print("=" * 55)
print("\nAvailable skill keywords you can enter:")
all_skills = set()
for row in job_roles:
    for word in row["skills"].split():
        all_skills.add(word)
print(", ".join(sorted(all_skills)))

print("\nEnter at least 3 skills (comma-separated).")
print("Example: python, docker, aws\n")

raw_input = input("Your skills: ")
user_skills_list = [s.strip().lower() for s in raw_input.split(",") if s.strip()]

while len(user_skills_list) < 3:
    print("Please enter at least 3 skills.")
    raw_input = input("Your skills: ")
    user_skills_list = [s.strip().lower() for s in raw_input.split(",") if s.strip()]

user_profile_text = " ".join(user_skills_list)

all_documents = df["skills"].tolist() + [user_profile_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

item_vectors = tfidf_matrix[:-1]
user_vector = tfidf_matrix[-1]

scores = cosine_similarity(user_vector, item_vectors).flatten()

df["similarity_score"] = scores

df_sorted = df.sort_values("similarity_score", ascending=False)

top_n = 3
top_recommendations = df_sorted.head(top_n)

print("\n" + "=" * 55)
print(f"   TOP {top_n} CAREER PATH RECOMMENDATIONS FOR YOU")
print("=" * 55)

for rank, (_, row) in enumerate(top_recommendations.iterrows(), start=1):
    match_percent = round(row["similarity_score"] * 100, 2)
    print(f"\n  Rank {rank}: {row['role']}")
    print(f"  Match Score: {match_percent}%")
    print(f"  Core Skills: {row['skills']}")

print("\n" + "=" * 55)
print("Full Similarity Scores (All Roles):")
print("=" * 55)
for _, row in df_sorted.iterrows():
    bar_length = int(row["similarity_score"] * 30)
    bar = "█" * bar_length + "-" * (30 - bar_length)
    print(f"  {row['role']:<28} [{bar}] {round(row['similarity_score']*100, 1)}%")

print("\nRecommendation complete.")