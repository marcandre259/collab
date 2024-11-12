from fastapi import FastAPI, HTTPException
import uvicorn
import sqlite3

import polars as pl

from typing import List

from recommender.models.data_models import Movie, Review

app = FastAPI(app="Recommender API")

uri = "sqlite://collab.db"

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
    conn = sqlite3.connect("collab.db")
    cursor = conn.cursor()

    user = review.user
    movie_id = review.movie_id
    review_score = review.review_score

    cursor.execute(
        """
        INSERT INTO reviews (user, movie_id, review_score)
        VALUES (?, ?, ?)
        """,
        (user, movie_id, review_score)
    )

    conn.commit()

    return f"Review of {user} added"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
