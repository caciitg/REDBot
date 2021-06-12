import torch
from transformers import BatchEncoding

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, url_for, request

from model import EmotionRoBERTa
from config import CONFIG


device = CONFIG.device
tokenizer = CONFIG.tokenizer

model = EmotionRoBERTa()
model.to(device)
model.eval()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        inputs: BatchEncoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=CONFIG.max_len,
            return_token_type_ids=True,
            padding='max_length'
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(dim=0).to(device)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(dim=0).to(device)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(dim=0).to(device)

        model.load_state_dict(torch.load(CONFIG.model_path))

        outputs = model(ids, mask, token_type_ids)
        outputs = np.array(torch.sigmoid(outputs).detach().cpu())

        emotion2genre_matrix = np.array(
            [[24, 21, 19, 13, 5, 53, 21, 10, 13, 12, 13, 4, 11, 5, 15, 19, 1, 3, 17, 16, 7, 19, 18, 3],   # anger
             [7, 10, 18, 21, 7, 59, 7, 10, 15, 23, 15, 3, 2, 11, 16, 7, 0, 3, 27, 24, 8, 10, 4, 2],       # fear
             [36, 26, 50, 31, 12, 53, 27, 14, 24, 28, 32, 8, 7, 20, 22, 27, 3, 7, 38, 37, 17, 32, 14, 9], # joy
             [9, 14, 15, 17, 9, 52, 10, 10, 19, 23, 12, 6, 4, 1, 15, 13, 3, 3, 32, 12, 10, 16, 3, 5],     # sadness
             [31, 14, 35, 20, 7, 35, 24, 13, 12, 13, 15, 7, 12, 10, 8, 23, 3, 7, 19, 23, 12, 30, 8, 7]]   # surprise
        )

        emotion2genre_matrix = normalize(emotion2genre_matrix, axis=1, norm='l2')

        outputs = np.matmul(outputs, emotion2genre_matrix)
        outputs = normalize(outputs, axis=1, norm='l2')

        ones = np.ones((596, 24))
        outputs = outputs * ones

        movie_df = pd.read_csv("movies.csv")
        genre_cols = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
                      'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Musical', 'Mystery',
                      'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
        vectors = movie_df[genre_cols].values
        normalized_vectors = normalize(vectors, axis=1, norm='l2')
        similarity = cosine_similarity(outputs, normalized_vectors)
        movie_df['similarity'] = similarity[0]

        movies = movie_df.sort_values(by=['similarity', 'avg_vote', 'year'], ascending=False) \
            .head(20)['original_title'] \
            .sample(frac=0.5)
        movie_1 = movies.iloc[0]
        movie_2 = movies.iloc[1]
        movie_3 = movies.iloc[2]
        movie_4 = movies.iloc[3]
        movie_5 = movies.iloc[4]

    return render_template('results.html', movie_1=movie_1, movie_2=movie_2,
                           movie_3=movie_3, movie_4=movie_4, movie_5=movie_5)


if __name__ == '__main__':
    app.run()
