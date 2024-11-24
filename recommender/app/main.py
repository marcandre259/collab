from fastapi import FastAPI, HTTPException
import uvicorn
import sqlite3

import polars as pl

from typing import List
from pathlib import Path

import torch
from recommender.models.data_models import (
    Movie,
    Review,
    ReviewDelete,
    MovieReviewResponse,
    Recommendation,
)
from functions import nn_model

from sqlalchemy import create_engine

import copy

app = FastAPI(app="Recommender API")
db_path = "collab.db"
project_root = Path(".")

uri = f"sqlite:///{db_path}"

engine = create_engine(uri)

with engine.connect() as conn:
    df_movies = pl.read_database("SELECT * FROM movies", conn)

df_id_mappings = pl.read_parquet(project_root / "data/id_chId_mappings.parquet")

df_item_mappings = df_id_mappings.select("movieId", "movieChId").unique()

# Load main ml model
collab_nn = torch.load(Path(project_root) / "data/models/collab_nn.pth")

user_id = 610


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
    with engine.connect() as conn:
        list_reviews = pl.read_database(
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
            conn,
        ).to_dicts()

    return list_reviews


@app.post("/tune/{user_name}")
async def tune_user_model(user_name: str) -> str:
    with engine.connect() as conn:
        df_reviews = pl.read_database(
            f"""
            SELECT
                *
            FROM reviews r
            WHERE r.user = '{user_name}'
            """,
            conn,
        )

    df_reviews = df_reviews.rename({"movie_id": "movieId"})

    df_reviews = df_item_mappings.join(df_reviews, on="movieId", how="inner")

    item_chIds = df_reviews["movieChId"].to_numpy()

    item_scores = df_reviews["review_score"].to_numpy()

    user_nn = copy.deepcopy(collab_nn)

    nn_model.finetune_user(
        user_nn, 610, item_chIds, item_scores, learning_rate=1e-2, epochs=1000
    )

    torch.save(user_nn, Path(project_root) / f"data/models/nn_user_{user_name}.pth")

    return f"Fine-tuned model for {user_name}"


@app.get("/recommendations/{user_name}")
def get_recommendations(user_name: str) -> List[Recommendation]:
    # Use existing reviews to only make recs on unreviewed movies
    with engine.connect() as conn:
        df_reviews = pl.read_database(
            f"""
            SELECT
                *
            FROM reviews r
            WHERE r.user = '{user_name}'
            """,
            conn,
        )

    df_reviews = df_reviews.rename({"movie_id": "movieId"})

    df_item_exclusive = df_item_mappings.join(df_reviews, on="movieId", how="anti")

    item_ids = df_item_exclusive["movieChId"].to_numpy()
    item_ids = [item for item in item_ids if item is not None]

    user_nn = torch.load(Path(project_root) / f"data/models/nn_user_{user_name}.pth")

    recs_chid, scores = nn_model.get_recommendations(
        user_nn, user_id, item_ids, top_k=10
    )

    recs_chid = recs_chid.numpy()
    scores = scores.numpy()

    df_scores = pl.DataFrame({"movieChId": recs_chid, "predicted_score": scores})

    df_item_recs = df_item_mappings.join(df_scores, on="movieChId")

    df_movie_recs = df_movies.join(df_item_recs, on="movieId").drop("movieChId")

    return df_movie_recs.to_dicts()


@app.delete("/reviews/")
async def delete_review(review_delete: ReviewDelete) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    user = review_delete.user
    movie_id = review_delete.movie_id

    cursor.execute(
        f"""
        DELETE FROM reviews
        WHERE user = '{user}'
            AND movie_id = {movie_id}
        """
    )

    conn.commit()

    return f"Review of movie {movie_id} from user {user} succesfully deleted"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
