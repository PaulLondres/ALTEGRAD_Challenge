import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, skip_connections=True):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.skip_connections = skip_connections
        if skip_connections:
            layernorm = [nn.LayerNorm(hidden_dim) for i in range(n_layers-1)]
            self.layernorm = nn.ModuleList(layernorm)

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
            if self.skip_connections:
                if i:
                    x = x + x_prev # Change : skip connections

                x = self.layernorm[i](x)
                x_prev = x.clone()

        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

import torch

def batch_trace(matrix_batch):
    return torch.einsum('bii->b', matrix_batch)  # Somme des diagonales

def batch_matrix_exp(matrix_batch):
    return torch.stack([torch.matrix_exp(matrix) for matrix in matrix_batch], dim=0)


def active_nodes_count(adj, threshold=0.01, sharpness=10):
    # Somme des lignes (ou colonnes, car adj est symétrique pour les graphes non dirigés)
    row_sums = adj.sum(dim=-1)  # Somme sur les colonnes, shape: (batch_size, n_nodes)

    # Sigmoïde pour approximer l'indicateur
    node_presence = torch.sigmoid(sharpness * (row_sums - threshold))  # Shape: (batch_size, n_nodes)

    # Nombre de nœuds actifs
    active_nodes = node_presence.sum(dim=-1)  # Shape: (batch_size,)
    return active_nodes


def graph_structure_loss(adj_recon, adj_target, coeff_node_count=1.0, coeff_density=1.0, coeff_triangles=1.0):
    batch_size, n_nodes, _ = adj_recon.size()

    # # 1. Connectivité approximée : trace(exp(A))
    # exp_recon = batch_matrix_exp(adj_recon)
    # exp_target = batch_matrix_exp(adj_target)
    # connectivity_recon = batch_trace(exp_recon) / n_nodes
    # connectivity_target = batch_trace(exp_target) / n_nodes
    # connectivity_loss = torch.abs(connectivity_recon - connectivity_target).mean()
    node_count_recon = active_nodes_count(adj_recon)
    node_count_target = active_nodes_count(adj_target)
    node_count_loss = torch.abs(node_count_recon - node_count_target).mean()

    # 2. Densité des arêtes : sum(A) / total_possible_edges
    total_possible_edges = n_nodes * (n_nodes - 1) / 2
    density_recon = adj_recon.sum(dim=(-2, -1)) / total_possible_edges
    density_target = adj_target.sum(dim=(-2, -1)) / total_possible_edges
    density_loss = torch.abs(density_recon - density_target).mean()

    # 3. Conservation des triangles : trace(A^3) / 6
    triangles_recon = batch_trace(torch.matrix_power(adj_recon, 3)) / 6
    triangles_target = batch_trace(torch.matrix_power(adj_target, 3)) / 6
    triangles_loss = torch.abs(triangles_recon - triangles_target).mean()

    # Perte totale combinée avec coefficients
    total_loss = (
        coeff_node_count * node_count_loss +
        coeff_density * density_loss +
        coeff_triangles * triangles_loss
    )

    # Détail des pertes individuelles
    loss_details = {
        "coeff_node_count": node_count_loss.item(),
        "density_loss": density_loss.item(),
        "triangles_loss": triangles_loss.item()
    }

    return total_loss, loss_details



# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, decoder_skip_connection=True, eps_scale = 0.02):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes, skip_connections=decoder_skip_connection)
        self.eps_scale = eps_scale

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * self.eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, coeff_loss_klv=5e-7, coeff_loss_recon = 1, coeffs_loss_struc = None):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        if coeffs_loss_struc is None:
            coeffs_loss_struc = {'coeff_global': 1, 'coeff_node_count': 2e-3, 'coeff_density': 0.3, 'coeff_triangles': 1e-2}
        
        # recon = F.l1_loss(adj, data.A, reduction='mean')
        recon = F.mse_loss(adj, data.A, reduction='mean')
        structural_loss = graph_structure_loss(adj, data.A, coeff_node_count=coeffs_loss_struc['coeff_node_count'],
                                               coeff_density=coeffs_loss_struc['coeff_density'],
                                               coeff_triangles=coeffs_loss_struc['coeff_triangles'])[0]
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = coeff_loss_recon * recon + coeff_loss_klv*kld + coeffs_loss_struc['coeff_global']*structural_loss

        return loss, recon, structural_loss, kld

