from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the necessary data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

job_df = pickle.load(open('job_df.pkl', 'rb'))

def get_match_category(score):
    if score <= 0.30:
        return "Poor match"
    elif score <= 0.50:
        return "Similar match"
    elif score <= 0.75:
        return "Decent match"
    elif score <= 0.90:
        return "Great match"
    else:
        return "Perfect match"

def get_similar_jobs(query, tfidf_matrix, tfidf_vectorizer, job_df, top_n=25):
    # Convert the query into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query])

    # Calculate cosine similarity between the query vector and all job vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of top similar jobs
    top_indices = similarity_scores.argsort(axis=1)[0][-top_n:][::-1]

    # Get similarity scores of top similar jobs
    top_scores = similarity_scores[0, top_indices]
    
    # Create a list to store the top similar job postings
    similar_jobs = []
    
    # Append details of top similar job postings to the list
    for i, score in zip(top_indices, top_scores):
        job_details = {
            'jobId': job_df.iloc[i]['jobId'],
            'title': job_df.iloc[i]['title'],
            'company': job_df.iloc[i]['company'],
            'type': job_df.iloc[i]['type'],
            'mode': job_df.iloc[i]['remote'],
            'place': job_df.iloc[i]['place'],
            'link': job_df.iloc[i]['link'],
            'match_category': get_match_category(score)
        }
        similar_jobs.append(job_details)

    return similar_jobs

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Define the route for job recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Code to retrieve the search query and get recommendations
    query = request.form['query']
    similar_jobs = get_similar_jobs(query, tfidf_matrix, tfidf_vectorizer, job_df)

    # Render the recommendations template with the results and query
    return render_template('recommendations.html', results=similar_jobs, query=query)


if __name__ == '__main__':
    app.run(debug=True)
