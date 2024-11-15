from fastapi import FastAPI, HTTPException
import uvicorn
import sqlite3

import polars as pl

from typing import List
from pathlib import Path

from recommender.models.data_models import Movie, Review, MovieReviewResponse

app = FastAPI(app="Recommender API")
db_path = "/Users/marc/Documents/collab/collab.db"
uri = f"sqlite://{db_path}"

df_movies = pl.read_database_uri("SELECT * FROM movies", uri=uri)


@app.get("/")
async def welcome():
    return "Hello friend :)"


@app.get("/movies/{keyword}")
async def search_movie_ids(keyword: str) -> List[Movie]:
    df_selection = df_movies.filter(
        pl.col("title").str.to_lowercase().str.contains(keyword)
    )

    df_selection = df_selection.rename({"movieId": "movie_id"})

    return df_selection.to_dicts()


@app.post("/users/")
async def submit_review(review: Review):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    user = review.user
    movie_id = review.movie_id
    review_score = review.review_score

    cursor.execute(
        """
        INSERT INTO reviews (user, movie_id, review_score)
        VALUES (?, ?, ?)
        """,
        (user, movie_id, review_score),
    )

    conn.commit()

    return f"Review of {user} added"


@app.get("/users/{user_name}")
async def get_user_reviews(user_name: str) -> List[MovieReviewResponse]:
    list_reviews = pl.read_database_uri(
        f"""
        SELECT
            r.movie_id
            ,  m.title
            , m.genres
            , r.user
            ,r.review_score
        FROM reviews r
        INNER JOIN movies m ON r.movie_id = m.movieId
        WHERE r.user = '{user_name}'
        """,
        uri=uri,
    ).to_dicts()

    return list_reviews


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
