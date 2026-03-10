from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import spacy
import fitz  # PyMuPDF
import re

app = Flask(__name__)

# Load VDAB data
vdab_file_path = 'VDAB_data.csv'
vdab_jobs = pd.read_csv(vdab_file_path).head(100)
vdab_jobs['text'] = vdab_jobs['title'].fillna('') + ' ' + vdab_jobs['employer'].fillna('') + ' ' + vdab_jobs['location'].fillna('') + ' ' + vdab_jobs['description'].fillna('')

# Load LinkedIn data
linkedin_file_path = 'Linkedin_data.csv'
linkedin_jobs = pd.read_csv(linkedin_file_path).head(100)
linkedin_jobs['text'] = linkedin_jobs['Title'].fillna('') + ' ' + linkedin_jobs['Address'].fillna('') + ' ' + linkedin_jobs['Job Description'].fillna('')

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Clean the job descriptions
vdab_jobs['clean_text'] = vdab_jobs['text'].apply(clean_text)
linkedin_jobs['clean_text'] = linkedin_jobs['text'].apply(clean_text)

def extract_text_from_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = text.strip()
    return clean_text(text)

def calculate_similarity(doc1, doc2):
    return nlp(doc1).similarity(nlp(doc2))

def find_similar_jobs(jobs, query_text, keyword=None, top_n=5, is_linkedin=False):
    query_vec = nlp(query_text)
    jobs['similarity'] = jobs['clean_text'].apply(lambda x: calculate_similarity(query_vec.text, x))
    
    if is_linkedin:
        if keyword:
            filtered_jobs = jobs[(jobs['Title'].str.contains(keyword, case=False, na=False)) | 
                                 (jobs['Job Description'].str.contains(keyword, case=False, na=False))]
        else:
            filtered_jobs = jobs
        filtered_jobs = filtered_jobs.sort_values(by='similarity', ascending=False).head(top_n)
        filtered_jobs['similarity_percentage'] = (filtered_jobs['similarity'] * 100).round(2)
        return filtered_jobs[['Title', 'Address', 'Link', 'similarity_percentage']]
    
    else:
        if keyword:
            filtered_jobs = jobs[(jobs['title'].str.contains(keyword, case=False, na=False)) | 
                                 (jobs['description'].str.contains(keyword, case=False, na=False))]
        else:
            filtered_jobs = jobs
        filtered_jobs = filtered_jobs.sort_values(by='similarity', ascending=False).head(top_n)
        filtered_jobs['similarity_percentage'] = (filtered_jobs['similarity'] * 100).round(2)
        return filtered_jobs[['title', 'employer', 'location', 'url', 'similarity_percentage']]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    return render_template('upload.html', name=name)

@app.route('/keyword', methods=['POST'])
def keyword():
    name = request.form['name']
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = f"./{uploaded_file.filename}"
        uploaded_file.save(file_path)
        extracted_text = extract_text_from_pdf(file_path)
        return render_template('keyword.html', name=name, extracted_text=extracted_text)
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process():
    extracted_text = request.form['extracted_text']
    keyword = request.form.get('keyword', None)
    vdab_results = find_similar_jobs(vdab_jobs, extracted_text, keyword)
    linkedin_results = find_similar_jobs(linkedin_jobs, extracted_text, keyword, is_linkedin=True)
    if vdab_results.empty and linkedin_results.empty:
        return render_template('results.html', message="No job found for the given keyword.")
    return render_template('results.html', vdab_results=vdab_results, linkedin_results=linkedin_results)

if __name__ == '__main__':
    app.run(debug=True)
