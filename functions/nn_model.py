import torch
from torch import nn

from typing import List


class BoundedSigmoid(nn.Module):
    def __init__(self, min_x: float, max_x: float):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.max_x - self.min_x) + self.min_x


class CollabNN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        min_outcome: float,
        max_outcome: float,
        n_embeddings: int = 64,
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(n_users + 1, n_embeddings)
        self.item_embeddings = nn.Embedding(n_items, n_embeddings)

        self.fc_layers = nn.Sequential(
            nn.Linear(n_embeddings * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            BoundedSigmoid(min_outcome, max_outcome),
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)

        # That's fine, because we pick n from users and n form items
        # (repeating)
        x = torch.cat([user_embeds, item_embeds], dim=1)

        return self.fc_layers(x)


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    epochs=10,
    patience=3,
    min_delta=1e-3,
):
    criterion = nn.MSELoss()

    best_valid_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        for batch_user_ids, batch_item_ids, batch_ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_user_ids, batch_item_ids)
            loss = criterion(predictions, batch_ratings[:, None])
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_valid_loss = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch_user_ids, batch_item_ids, batch_ratings in valid_loader:
                predictions = model(batch_user_ids, batch_item_ids)
                loss = criterion(predictions, batch_ratings)

                epoch_valid_loss += loss.item()
                valid_batches += 1

        avg_valid_loss = epoch_valid_loss / valid_batches
        valid_losses.append(avg_valid_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_valid_loss:.4f}")

        # Early stopping on epoch
        if avg_valid_loss <= (best_valid_loss - min_delta):
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()

        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break

    return {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "best_valid_loss": best_valid_loss,
        "stopped_epoch": epoch + 1,
    }


def finetune_user(
    model,
    user_id: int,
    item_ids: List[int],
    item_ratings: List[float],
    learning_rate: float = 0.01,
    epochs: int = 100,
):
    criterion = nn.MSELoss()

    # I left the last user embedding of the model free, so can user
    # user_embedding - 1
    for param in model.parameters():
        param.require_grad = False

    # Keep item embeddings constant
    model.user_embeddings.weight.requires_grad = True

    optimizer = torch.optim.Adam(
        [model.user_embeddings.weight], lr=learning_rate
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(
            torch.tensor([user_id] * len(item_ids)), torch.tensor(item_ids)
        )
        loss = criterion(
            predictions, torch.tensor(item_ratings).float()[:, None]
        )
        loss.backward()
        optimizer.step()


def get_recommendations(model, user_id, item_ids, top_k=10):
    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor([user_id] * len(item_ids))
        item_ids = torch.tensor(item_ids)
        predictions = model(user_ids, item_ids).flatten()

        # Get top-k recommendations
        top_k_values, top_k_indices = torch.topk(predictions, k=top_k)
        recommended_items = item_ids[top_k_indices]

    return recommended_items, top_k_values