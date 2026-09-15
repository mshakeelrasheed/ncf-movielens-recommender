import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users=943, num_items=1682, latent_dim_gmf=32, latent_dim_mlp=32, mlp_hidden=[64, 32, 16]):
        super().__init__()
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        layers = []
        in_dim = latent_dim_mlp * 2
        for h_dim in mlp_hidden:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        self.prediction_layer = nn.Linear(latent_dim_gmf + mlp_hidden[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        p_gmf = self.user_embed_gmf(user_indices)
        q_gmf = self.item_embed_gmf(item_indices)
        phi_gmf = p_gmf * q_gmf

        p_mlp = self.user_embed_mlp(user_indices)
        q_mlp = self.item_embed_mlp(item_indices)
        phi_mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=-1))

        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        return self.sigmoid(self.prediction_layer(fusion)).squeeze()