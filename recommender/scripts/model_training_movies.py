import polars as pl

from functions import PolarsCollabDataset
from functions import nn_model

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

import mlflow

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8078")
# mlflow.set_experiment("model training")

import sys

project_root = "/Users/marc/Documents/collab/"
sys.path.append(project_root)

# Data load
df_ratings = pl.read_csv(
    Path(project_root) / "data/ml-latest-small/ratings.csv"
)
df_movies = pl.read_csv(Path(project_root) / "data/ml-latest-small/movies.csv")

df_ratings = df_ratings.join(df_movies, on="movieId", how="inner")

# Create ordered ids for the users and movies
userId = df_ratings["userId"].unique().to_numpy()
movieId = df_ratings["movieId"].unique().to_numpy()

userId2userChId = {j: i for i, j in enumerate(userId)}
movieId2movieChId = {j: i for i, j in enumerate(movieId)}

userChId2userId = {v: k for k, v in userId2userChId.items()}
movieChId2movieId = {v: k for k, v in movieId2movieChId.items()}

q = [
    pl.col("userId").replace(userId2userChId).alias("userChId"),
    pl.col("movieId").replace(movieId2movieChId).alias("movieChId"),
]

df_ratings = df_ratings.with_columns(q)

df_id_mappings = df_ratings.select(
    ["userId", "movieId", "userChId", "movieChId"]
).unique()

df_id_mappings.write_parquet(
    Path(project_root) / "data/id_chId_mappings.parquet"
)

# Defining parameters for model
n_users = df_ratings.select("userId").unique().shape[0]
n_items = df_ratings.select("movieId").unique().shape[0]

min_review_score = df_ratings.select(pl.col("rating").min()).item()
max_review_score = df_ratings.select(pl.col("rating").max()).item()

# Creating dataset for batch loads
df_train, df_valid = train_test_split(df_ratings, test_size=0.3)

ds_train = PolarsCollabDataset(
    df_train,
    label_column="rating",
    user_column="userChId",
    item_column="movieChId",
)
ds_valid = PolarsCollabDataset(
    df_valid,
    label_column="rating",
    user_column="userChId",
    item_column="movieChId",
)

batch_size = 32
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dl_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)

collab_nn = nn_model.CollabNN(
    n_users, n_items, min_review_score, max_review_score
)

learning_rate = 1e-4
optimizer = torch.optim.Adam(collab_nn.parameters())

training_output = nn_model.train_model(
    collab_nn, dl_train, dl_valid, optimizer, epochs=10, patience=2
)

torch.save(
    collab_nn,
    Path(project_root) / "data/models/collab_nn.pth",
)

# Prep output
training_output["train_losses"] = training_output["train_losses"][-1]
training_output["valid_losses"] = training_output["valid_losses"][-1]

# # Tracking metrics and saving model
# with mlflow.start_run():
#     mlflow.log_metrics(training_output)
