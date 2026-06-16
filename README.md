# Movie Recommender System

A content-based movie recommender system built using the TMDB 5000 dataset that suggests similar movies based on metadata features.

## How It Works

Movies are represented as tag vectors combining overview, genres, keywords, cast, and crew. A CountVectorizer converts these into a bag-of-words matrix, and cosine similarity is computed across all movie pairs to find the closest matches.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (CountVectorizer, cosine similarity)
- TMDB 5000 Movies & Credits dataset

## Results

Evaluated using Precision@K (genre overlap as relevance proxy) across 5 test queries:

| Movie | Precision@5 |
|---|---|
| Avatar | 1.00 |
| The Dark Knight | 1.00 |
| Inception | 1.00 |
| Toy Story | 1.00 |
| The Godfather | 0.80 |

**Average Precision@5: 0.96**

## Usage

Run all cells in `Movie_Recommender_System.ipynb` sequentially, then call:

```python
recommend('Movie Title')
```

## Dataset

TMDB 5000 Movies and Credits dataset containing metadata for ~5000 movies including genres, cast, crew, keywords, and overview.
