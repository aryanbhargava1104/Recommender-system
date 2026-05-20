from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import pickle

# ==========================================
# 1. Redefine the PyTorch Model Architecture
# PyTorch needs the exact class structure to load the weights
# ==========================================
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, mlp_layer_sizes=[128, 64, 32]):
        super(NeuMF, self).__init__()
        
        self.embedding_user_mf = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)
        
        mlp_emb_dim = mlp_layer_sizes[0] // 2 
        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_emb_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_emb_dim)
        
        mlp_modules = []
        for i in range(len(mlp_layer_sizes) - 1):
            mlp_modules.append(nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i+1]))
            mlp_modules.append(nn.ReLU())
            
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        predict_input_size = mf_dim + mlp_layer_sizes[-1]
        self.prediction_layer = nn.Linear(predict_input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf) 
        
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1) 
        mlp_vector = self.mlp_layers(mlp_vector)
        
        combined_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction_layer(combined_vector)
        return self.sigmoid(prediction).squeeze()

# ==========================================
# 2. Initialize FastAPI and Load Assets
# ==========================================
app = FastAPI(title="MovieLens Recommender API")

print("Loading artifacts...")
with open("ncf_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

user_mapping = artifacts['user_mapping']
movie_mapping = artifacts['movie_mapping']
reverse_movie_mapping = artifacts['reverse_movie_mapping']
movie_titles = artifacts['movie_titles']

num_users = len(user_mapping)
num_movies = len(movie_mapping)

print("Loading model weights...")
# We load the model onto the CPU for the API. 
# Inference for a single user is so fast that transferring to the GPU actually slows it down.
device = torch.device("cpu")
model = NeuMF(num_users=num_users, num_items=num_movies, mf_dim=32, mlp_layer_sizes=[128, 64, 32])
model.load_state_dict(torch.load("ncf_model_weights.pth", map_location=device))
model.eval()
print("✅ API is ready!")

# ==========================================
# 3. Define the Recommendation Endpoint
# ==========================================
@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_k: int = 10):
    # Check if user exists in our dataset
    if user_id not in user_mapping:
        raise HTTPException(status_code=404, detail="User not found in dataset.")
        
    user_idx = user_mapping[user_id]
    
    # Grab all unique movie indices the model knows about
    all_movie_indices = list(reverse_movie_mapping.keys())
    
    # Create tensors: A list of the same user duplicated, paired with every movie
    user_tensor = torch.tensor([user_idx] * len(all_movie_indices), dtype=torch.long)
    item_tensor = torch.tensor(all_movie_indices, dtype=torch.long)
    
    # Run the model predictions
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor).numpy()
        
    # Pair up the movie indices with their predicted score and sort descending
    movie_scores = list(zip(all_movie_indices, predictions))
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Format the top K results using our translation dictionaries
    recommendations = []
    for movie_idx, score in movie_scores[:top_k]:
        orig_id = int(reverse_movie_mapping[movie_idx])
        title = movie_titles.get(orig_id, "Unknown Title")
        recommendations.append({
            "movie_id": orig_id,
            "title": title,
            "match_score": round(float(score) * 100, 2) # Convert to a clean percentage
        })
        
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }