import argparse
import os
import random
import pickle
import csv
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset
from prompt_embedding import get_conditioning_vector
from evaluate_model import compute_prediction_mae

np.random.seed(13)

# Configurer le logger
logger = logging.getLogger("NeuralGraphGenerator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Création de gestionnaires pour les logs
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = None

# Fonction de training pour l'autoencoder
def train_autoencoder(autoencoder, train_loader, val_loader, loss_coeffs, args, device, writer, save_dir):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder + 1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_recon = 0
        train_loss_struct = 0
        train_loss_kld = 0
        cnt_train = 0
        iter_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, struct, kld = autoencoder.loss_function(data, coeff_loss_klv=loss_coeffs['coeff_loss_klv'],
                                                                 coeff_loss_recon=loss_coeffs['coeff_loss_recon'],
                                                                 coeffs_loss_struc=loss_coeffs['coeffs_loss_struc'])
            train_loss_recon += recon.item()
            train_loss_kld += kld.item()
            train_loss_struct += struct.item()
            cnt_train += 1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch) + 1
            optimizer.step()

            iter_loss += loss.item()

        writer.add_scalar('Autoencoder/Train_Loss', train_loss_all/cnt_train, epoch)
        writer.add_scalar('Autoencoder/Train_Loss_recon', train_loss_all/cnt_train, epoch)
        writer.add_scalar('Autoencoder/Train_Loss_struct', train_loss_all/cnt_train, epoch)
        writer.add_scalar('Autoencoder/Train_Loss_kld', train_loss_all/cnt_train, epoch)

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_recon = 0
        val_loss_struct = 0
        val_loss_kld = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                loss, recon, struct, kld = autoencoder.loss_function(data, coeff_loss_klv=loss_coeffs['coeff_loss_klv'],
                                                                 coeff_loss_recon=loss_coeffs['coeff_loss_recon'],
                                                                 coeffs_loss_struc=loss_coeffs['coeffs_loss_struc'])
                val_loss_recon += recon.item()
                val_loss_struct += struct.item()
                val_loss_kld += kld.item()
                val_loss_all += loss.item()
                cnt_val += 1
                val_count += torch.max(data.batch) + 1

                iter_loss += loss.item()
        val_loss_all /= len(val_loader)
        writer.add_scalar('Autoencoder/Val_Loss', val_loss_all / cnt_val, epoch)
        writer.add_scalar('Autoencoder/Val_Loss_recon', val_loss_all / cnt_val, epoch)
        writer.add_scalar('Autoencoder/Val_Loss_struct', val_loss_all / cnt_val, epoch)
        writer.add_scalar('Autoencoder/Val_Loss_kld', val_loss_all / cnt_val, epoch)

        if val_loss_all < best_val_loss:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_dir, 'autoencoder_best.pth.tar'))
        if epoch % 10 == 0:
            #dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_msg = (f'Epoch: {epoch:04d}, Train Loss: {train_loss_all / cnt_train:.5f}, '
                       f'Train Reconstruction Loss: {train_loss_recon / cnt_train:.5f}, '
                       f'Train Structural Loss: {train_loss_struct / cnt_train:.5f}, '
                       f'Train KLD Loss: {train_loss_kld / cnt_train:.5f}, '
                       f'Val Loss: {val_loss_all / cnt_val:.5f}, '
                       f'Val Reconstruction Loss: {val_loss_recon / cnt_val:.5f}, '
                       f'Val Structural Loss: {val_loss_struct / cnt_val:.5f}, '
                       f'Val KLD Loss: {val_loss_kld / cnt_val:.5f}')
            logger.info(log_msg)
        scheduler.step()

    return autoencoder

# Fonction de training pour le modèle de débruitage
def train_denoiser(denoise_model, autoencoder, train_loader, val_loader, args, device, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, writer, save_dir):
    optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise + 1):
        denoise_model.train()
        train_loss = 0
        train_count = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            cond_vector = get_conditioning_vector(data.stats, data.cond_text, args.conditioning_embedding).to(device)
            loss = p_losses(denoise_model, x_g, t, cond_vector, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            loss.backward()
            train_loss += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        writer.add_scalar('Denoiser/Train_Loss', train_loss/train_count, epoch)

        denoise_model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                cond_vector = get_conditioning_vector(data.stats, data.cond_text, args.conditioning_embedding).to(device)
                loss = p_losses(denoise_model, x_g, t, cond_vector, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
                val_loss += x_g.size(0) * loss.item()
                val_count += x_g.size(0)
        val_loss /= len(val_loader)

        writer.add_scalar('Denoiser/Val_Loss', val_loss/train_count, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join('denoise_model.pth.tar'))

        if epoch % 10 == 0:
            #dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log_msg = (f'Epoch: {epoch:04d}, Train Loss: {train_loss / train_count:.5f}, '
                       f'Val Loss: {val_loss / val_count:.5f}')
            logger.info(log_msg)

        scheduler.step()

    return denoise_model

# Fonction de prédiction et d'évaluation
def predict_and_evaluate(csv_path, autoencoder, denoise_model, loader, args, device, betas, save_dir):
    with open(os.path.join(save_dir, csv_path), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])
        for k, data in enumerate(tqdm(loader, desc=f"Processing {loader}", )):
            data = data.to(device)
            # stat = data.stats
            bs = data.stats.size(0)

            graph_ids = data.filename
            cond_vector = get_conditioning_vector(data.stats, data.cond_text, args.conditioning_embedding).to(device)
            samples = sample(denoise_model, cond_vector, latent_dim=args.latent_dim, timesteps=args.timesteps,
                             betas=betas, batch_size=bs)
            x_sample = samples[-1]
            adj = autoencoder.decode_mu(x_sample)

            for i in range(bs):

                Gs_generated = construct_nx_from_adj(adj[i, :, :].detach().cpu().numpy())
                graph_id = graph_ids[i]
                edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])
                writer.writerow([graph_id, edge_list_text])

# Fonction principale
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f"test/{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    log_file = os.path.join(save_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',  datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    logger.info(args)

    # Préparation des datasets
    trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
    validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
    testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Modèles et optimisateurs
    decoder_skip_connection = True
    eps_scale = 0.02
    loss_coeffs = {'coeff_loss_klv': 5e-7,
    'coeff_loss_recon': 1,
    'coeffs_loss_struc': {'coeff_global': 0.01,
                          'coeff_node_count': 2e-3,
                          'coeff_density': 0.3,
                          'coeff_triangles': 1e-2}}

    autoencoder = VariationalAutoEncoder(
        args.spectral_emb_dim + 1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim,
        args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, decoder_skip_connection, eps_scale).to(device)

    denoise_model = DenoiseNN(
        input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise,
        n_layers=args.n_layers_denoise, n_cond=7, d_cond=args.dim_condition).to(device)

    # Entraînement de l'autoencoder
    if args.train_autoencoder:
        logger.info("Starting autoencoder training")
        autoencoder = train_autoencoder(autoencoder, train_loader, val_loader, loss_coeffs, args, device, writer, save_dir)
    else:
        logger.info("Skipping autoencoder training")
        autoencoder.load_state_dict(torch.load('autoencoder_best.pth'))

    # Entraînement du dénoiseur
    betas = linear_beta_schedule(args.timesteps)
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1. - betas, axis=0))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - torch.cumprod(1. - betas, axis=0))

    if args.train_denoiser:
        logger.info("Starting denoiser training")
        denoise_model = train_denoiser(denoise_model, autoencoder, train_loader, val_loader, args, device, betas,
                                       sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, writer, save_dir)
    else:
        logger.info("Skipping denoiser training")
        denoise_model.load_state_dict(torch.load('denoise_model_best.pth'))

    val_pred_file = "pred_valid.csv"
    test_pred_file = "pred_test.csv"
    logger.info("Predict graphs on validation set")
    predict_and_evaluate(val_pred_file, autoencoder, denoise_model, val_loader, args, device, betas, save_dir)
    logger.info("Predict graphs on test set")
    predict_and_evaluate(test_pred_file, autoencoder, denoise_model, test_loader, args, device, betas, save_dir)

    final_mae, final_mae_per_metric = compute_prediction_mae(os.path.join(save_dir, val_pred_file))
    logger.info("Final MAE: {}".format(final_mae))
    logger.info("Final MAE per metric: {}".format(final_mae_per_metric))
    writer.add_scalar('Pred_metrics/MAE_tot', final_mae, 0)
    writer.add_scalar('Pred_metrics/MAE_nb_nodes', final_mae_per_metric[0])
    writer.add_scalar('Pred_metrics/MAE_nb_edges', final_mae_per_metric[1])
    writer.add_scalar('Pred_metrics/MAE_avg_degree', final_mae_per_metric[2])
    writer.add_scalar('Pred_metrics/MAE_nb_trg', final_mae_per_metric[3])
    writer.add_scalar('Pred_metrics/MAE_cluster_coeff', final_mae_per_metric[4])
    writer.add_scalar('Pred_metrics/MAE_max_kcore', final_mae_per_metric[5])
    writer.add_scalar('Pred_metrics/MAE_nb_communities', final_mae_per_metric[6])
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')
    # Learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for the optimizer, typically a small float value (default: 0.001)")  # Change : 0.001 to 0.01

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

    # Batch size for training
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

    # Number of epochs for the autoencoder training
    parser.add_argument('--epochs-autoencoder', type=int, default=200,
                        help="Number of training epochs for the autoencoder (default: 200)")

    # Hidden dimension size for the encoder network
    parser.add_argument('--hidden-dim-encoder', type=int, default=64,
                        help="Hidden dimension size for encoder layers (default: 64)")

    # Hidden dimension size for the decoder network
    parser.add_argument('--hidden-dim-decoder', type=int, default=256,
                        help="Hidden dimension size for decoder layers (default: 256)")

    # Dimensionality of the latent space
    parser.add_argument('--latent-dim', type=int, default=32,
                        help="Dimensionality of the latent space in the autoencoder (default: 32)")

    # Maximum number of nodes of graphs
    parser.add_argument('--n-max-nodes', type=int, default=50,
                        help="Possible maximum number of nodes in graphs (default: 50)")

    # Number of layers in the encoder network
    parser.add_argument('--n-layers-encoder', type=int, default=2,
                        help="Number of layers in the encoder network (default: 2)")

    # Number of layers in the decoder network
    parser.add_argument('--n-layers-decoder', type=int, default=3,
                        help="Number of layers in the decoder network (default: 3)")

    # Dimensionality of spectral embeddings for graph structure representation
    parser.add_argument('--spectral-emb-dim', type=int, default=10,
                        help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

    # Number of training epochs for the denoising model
    parser.add_argument('--epochs-denoise', type=int, default=600,
                        help="Number of training epochs for the denoising model (default: 100)")

    # Number of timesteps in the diffusion
    parser.add_argument('--timesteps', type=int, default=500,
                        help="Number of timesteps for the diffusion (default: 500)")

    # Hidden dimension size for the denoising model
    parser.add_argument('--hidden-dim-denoise', type=int, default=512,
                        help="Hidden dimension size for denoising model layers (default: 512)")

    # Number of layers in the denoising model
    parser.add_argument('--n-layers_denoise', type=int, default=3,
                        help="Number of layers in the denoising model (default: 3)")  # Change : increased n layers from 3

    # Flag to toggle training of the autoencoder (VGAE)
    parser.add_argument('--train-autoencoder', action='store_true', default=True,
                        help="Flag to enable/disable autoencoder (VGAE) training (default: disabled)")

    # Flag to toggle training of the diffusion-based denoising model
    parser.add_argument('--train-denoiser', action='store_true', default=True,
                        help="Flag to enable/disable denoiser training (default: enabled)")

    # Dimensionality of conditioning vectors for conditional generation
    parser.add_argument('--dim-condition', type=int, default=128,
                        help="Dimensionality of conditioning vectors for conditional generation (default: 128)")  # Change : increased dim-condition from 128

    # Number of conditions used in conditional vector (number of properties)
    parser.add_argument('--n-condition', type=int, default=7,
                        help="Number of distinct condition properties used in conditional vector (default: 7)")

    parser.add_argument('--conditioning-embedding', choices=['regex_parsing', 'SBERT', 'DistilGPT2', 'RoBERTa'],
                        default='regex_parsing')
    args = parser.parse_args()
    main(args)

