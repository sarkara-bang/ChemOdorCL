import time
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch import nn
import args
import torch.nn.functional as F
import os
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
# from fast_transformers.builders import TransformerEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from apex import optimizers
import subprocess
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
# from utils import normalize_smiles
from rdkit.Chem import MolFromSmiles, rdmolops, AllChem
from rdkit import Chem
from sklearn.preprocessing import MultiLabelBinarizer
import json
from copy import deepcopy
from pathlib import Path

FEATURE_EXTRACTION_DIR = Path(__file__).resolve().parent.parent / "feature_extraction"

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# ============ Added: truncated SVD fingerprint reducer ============
class SVDFingerprintReducer:
    """
    Truncated SVD reducer: map 2048-dimensional binary Morgan fingerprints to 768-dimensional continuous vectors.
    - Prefit the projection matrix using sklearn TruncatedSVD.
    - Apply a functional PyTorch transform with GPU support.
    - L2-normalize the output for direct use in cosine similarity calculations.
    """
    def __init__(self, out_dim: int = 768, random_state: int = 42):
        self.out_dim = out_dim
        self.random_state = random_state
        self.W = None  # Projection matrix: (2048, out_dim).
        self._fitted = False
        
    def fit(self, fp_np: np.ndarray):
        """
        Fit the SVD projection matrix on a reference dataset; only one fit is required.
        Args:
            fp_np: np.ndarray of shape (N, 2048), binary Morgan fingerprints
        """
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=self.out_dim, random_state=self.random_state)
        svd.fit(fp_np)
        # components_: (n_components, n_features); transpose to (2048, 768).
        self.W = torch.tensor(svd.components_.T, dtype=torch.float32)
        self._fitted = True
        print(f"✓ SVDFingerprintReducer fitted: {fp_np.shape} -> projection matrix {self.W.shape}")
        
    def transform(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """
        Functional transform: binary fingerprints -> L2-normalized continuous vectors.
        Args:
            fp_tensor: (n, 2048) float32 binary tensor
        Returns:
            (n, 768) L2-normalized continuous tensor for cosine similarity
        """
        assert self._fitted, "SVDFingerprintReducer must be fitted before transform"
        device = fp_tensor.device
        W = self.W.to(device)
        fp_float = fp_tensor.float()
        # Linear projection: (n, 2048) @ (2048, 768) -> (n, 768).
        proj = torch.matmul(fp_float, W)
        # L2 normalization: cosine similarity equals the dot product of normalized vectors.
        return F.normalize(proj, dim=-1, p=2)


class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer, svd_reducer=None):
        super(LightningModule, self).__init__()
        
        self.start = time.time()
        self.config = config
        self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        self.min_loss = {
            self.hparams.measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.measure_name + "min_epoch": 0,
        }
        # ============ Added: SVD reducer support ============
        self.svd_reducer = svd_reducer
        self.svd_fitted = (svd_reducer is not None) and getattr(svd_reducer, '_fitted', False)
        # ===========================================

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  
        self.net_for_fingerprint = self.fp_layer(config.n_embd)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.net = self.Net(
            config.n_embd, dims=config.dims, dropout=config.dropout,
        )

    class Net(nn.Module):
        dims = [150, 50, 50, 2]


        def __init__(self, smiles_embed_dim, dims=dims, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 1)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            return z

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
            # self.head = nn.Linear(n_embd, 8*(n_vocab//8+1), bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor
    
    class fp_layer(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, 2048, bias=False)
            # self.sigmoid = torch.nn.Sigmoid()
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor1 = self.ln_f(tensor)
            tensor = self.head(tensor1)
            # tensor = self.sigmoid(tensor)
            return tensor1, tensor

    def get_loss(self, smiles_emb, measures):

        z_pred = self.net.forward(smiles_emb).squeeze()
        measures = measures.float()

        return self.loss(z_pred, measures), z_pred, measures
    
    # def get_info_nce_loss(self, features):
    #     labels = torch.cat([torch.arange(features.shape[0] // 2) for i in range(2)], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(features.device)

    #     features = F.normalize(features, dim=1)

    #     similarity_matrix = torch.matmul(features, features.T)

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    #     logits = logits / 0.1 # temperature
        
    #     return logits, labels
    
    
    def get_model_distribution(self, features):# Compute the model output distribution for MSE.
        """
        Compute the model similarity distribution x.
        features: [2N, emb_dim] molecular vectors produced by the encoder.
        Returns: [2N, 2N] cosine similarity matrix.
        """
        features = F.normalize(features, dim=1)  # L2 normalization.
        similarity_matrix = torch.matmul(features, features.T)  # Cosine similarity.
        return similarity_matrix
    
    def get_model_distribution01(self, features, temperature=0.1): # Model output distribution for KL divergence.
        """
        Compute the model similarity probability distribution.
        features: [2N, emb_dim] molecular vectors produced by the encoder.
        temperature: controls the sharpness of the distribution.
        Returns: [2N, 2N] probability matrix with each row summing to 1.
        """
        features = F.normalize(features, dim=1)  # L2 normalization.
        similarity_matrix = torch.matmul(features, features.T)  # Cosine similarity.
        # Scale by temperature and apply softmax to obtain probabilities.
        logits = similarity_matrix / temperature
        model_prob = F.softmax(logits, dim=1)  # Each row sums to 1.
        return model_prob
    

    def get_target_distribution(self, fps, labels, lambdap=0.5): # MSE: Tanimoto similarity for fingerprints and odor labels.
        """
        Compute the target similarity distribution y = λ*sm + (1-λ)*sl.
        Args:
            fps: [2N, 2048] binary Morgan fingerprints.
            labels: [2N, 27] multi-hot odor labels.
            lambdap: fingerprint similarity weight λ.
        Returns: target_similarity: [2N, 2N] target similarity matrix.
        """
        batch_size = fps.shape[0]
        
        # 1. Morgan fingerprint Tanimoto similarity, computed in vectorized form.
        fps_bool = (fps > 0.5).float()  # [2N, 2048]
        
        fp_counts = fps_bool.sum(dim=1, keepdim=True)  # [2N, 1]: count the ones in each vector.
        intersection = torch.matmul(fps_bool, fps_bool.T)  # [2N, 2N]: intersection matrix, fps_bool @ fps_bool.T.
        union = fp_counts + fp_counts.T - intersection  # [2N, 2N]: union matrix, |A| + |B| - |A∩B|.
        
        sm = intersection / (union + 1e-8)  # [2N, 2N]: Tanimoto similarity.
        
        # 2. Odor-label Tanimoto similarity, computed in vectorized form.
        labels_float = labels.float()  # [2N, 27]
        
        label_counts = labels_float.sum(dim=1, keepdim=True)  # [2N, 1]: number of labels for each molecule.
        same_labels = torch.matmul(labels_float, labels_float.T)  # [2N, 2N]: matrix of shared label counts.
        total_labels = label_counts + label_counts.T  # [2N, 2N]: matrix of total label counts.
        
        sl = same_labels / (total_labels - same_labels + 1e-8) # [2N, 2N]: odor-label Tanimoto similarity.
        
        # 3. Weighted combination.
        target_similarity = lambdap * sm + (1 - lambdap) * sl
        
        # 4. Optionally force the diagonal to 1: a molecule is identical to itself.
        target_similarity.fill_diagonal_(1.0)
        
        return target_similarity

    def get_target_distribution01(self, fps, labels, lambdap=0.5): # MSE: cosine similarity for fingerprints and Tanimoto similarity for odor labels.
        """
        Compute the target similarity distribution y = λ*sm + (1-λ)*sl.
        Args:
            fps: [2N, 2048] binary Morgan fingerprints.
            labels: [2N, 27] multi-hot odor labels.
            lambdap: fingerprint similarity weight λ.
        Returns: target_similarity: [2N, 2N] target similarity matrix.
        """
        batch_size = fps.shape[0]
        
        # 1. Morgan fingerprint cosine similarity, computed in vectorized form.
        fps_float = fps.float()  # [2N, 2048]: floating-point values support gradient propagation.
        # F.normalize is equivalent to fps / (||fps||_2 + eps).
        fps_normalized = F.normalize(fps_float, p=2, dim=1, eps=1e-8)  # [2N, 2048]
        #  cos_sim(i,j) = (fps_i · fps_j) / (||fps_i|| * ||fps_j||)
        # Normalization reduces the calculation to fps_normalized @ fps_normalized.T.
        sm = torch.matmul(fps_normalized, fps_normalized.T)  # [2N, 2N]

        # 2. Odor-label Tanimoto similarity, computed in vectorized form.
        labels_float = labels.float()  # [2N, 27]
        
        label_counts = labels_float.sum(dim=1, keepdim=True)  # [2N, 1]: number of labels for each molecule.
        same_labels = torch.matmul(labels_float, labels_float.T)  # [2N, 2N]: matrix of shared label counts.
        total_labels = label_counts + label_counts.T  # [2N, 2N]: matrix of total label counts.
        
        sl = same_labels / (total_labels - same_labels + 1e-8) # [2N, 2N]: odor-label Tanimoto similarity.
        
        # 3. Weighted combination.
        target_similarity = lambdap * sm + (1 - lambdap) * sl
        
        # 4. Optionally force the diagonal to 1: a molecule is identical to itself.
        target_similarity.fill_diagonal_(1.0)
        
        return target_similarity
    
    def get_target_distribution02(self, fps, labels, lambdap=0.5): # KL divergence: Tanimoto similarity for fingerprints and odor labels.
        """
        Compute the target similarity distribution y = λ*sm + (1-λ)*sl.
        Args:
            fps: [2N, 2048] binary Morgan fingerprints.
            labels: [2N, 27] multi-hot odor labels.
            lambdap: fingerprint similarity weight λ.
        Returns: target_similarity: [2N, 2N] target similarity matrix.
        """
        batch_size = fps.shape[0]
        
        # 1. Morgan fingerprint Tanimoto similarity, computed in vectorized form.
        fps_bool = (fps > 0.5).float()  # [2N, 2048]
        
        fp_counts = fps_bool.sum(dim=1, keepdim=True)  # [2N, 1]: count the ones in each vector.
        intersection = torch.matmul(fps_bool, fps_bool.T)  # [2N, 2N]: intersection matrix, fps_bool @ fps_bool.T.
        union = fp_counts + fp_counts.T - intersection  # [2N, 2N]: union matrix, |A| + |B| - |A∩B|.
        
        sm = intersection / (union + 1e-8)  # [2N, 2N]: Tanimoto similarity.
        
        # 2. Odor-label Tanimoto similarity, computed in vectorized form.
        labels_float = labels.float()  # [2N, 27]
        
        label_counts = labels_float.sum(dim=1, keepdim=True)  # [2N, 1]: number of labels for each molecule.
        same_labels = torch.matmul(labels_float, labels_float.T)  # [2N, 2N]: matrix of shared label counts.
        total_labels = label_counts + label_counts.T  # [2N, 2N]: matrix of total label counts.
        
        sl = same_labels / (total_labels - same_labels + 1e-8) # [2N, 2N]: odor-label Tanimoto similarity.
        
        # 3. Weighted combination.
        target_similarity = lambdap * sm + (1 - lambdap) * sl
        
        # 4. Optionally force the diagonal to 1: a molecule is identical to itself.
        target_similarity.fill_diagonal_(1.0)

        # 5. Added: convert to a probability distribution to match model_distribution.
        temperature = 0.1  # Use the same temperature as model_distribution.
        target_prob = F.softmax(target_similarity / temperature, dim=1)
        
        return target_prob
    
    def get_target_distribution03(self, fps, labels, lambdap=0.5):# MSE: cosine similarity for fingerprints and Tanimoto similarity for odor labels.
        """
        Compute the target similarity distribution for MSE loss.
        Use truncated SVD to map binary Morgan fingerprints to 768-dimensional continuous vectors before computing cosine similarity.
        
        Args:
            fps: [2N, 2048] binary Morgan fingerprints stored as floating-point tensors.
            labels: [2N, 27] multi-hot odor labels.
            lambdap: fingerprint similarity weight λ (default: 0.5).
        Returns:
            target_similarity: [2N, 2N] target similarity matrix.
        """
        # 1. Fingerprint similarity: SVD reduction followed by cosine similarity.
        if self.svd_fitted:
            # Apply the prefitted SVD projection.
            fps_continuous = self.svd_reducer.transform(fps)  # [2N, 768]
            sm = torch.matmul(fps_continuous, fps_continuous.T)  # [2N, 2N]: cosine similarity.
        else:
            # Fall back to raw fingerprint cosine similarity if the reducer is not fitted.
            rank_zero_warn("SVDFingerprintReducer not fitted, using raw fingerprint cosine similarity as fallback")
            fps_float = fps.float()
            fps_normalized = F.normalize(fps_float, p=2, dim=1, eps=1e-8)
            sm = torch.matmul(fps_normalized, fps_normalized.T)
        
        # 2. Odor-label Tanimoto similarity, computed in vectorized form.
        labels_float = labels.float()  # [2N, 27]
        label_counts = labels_float.sum(dim=1, keepdim=True)  # [2N, 1]
        same_labels = torch.matmul(labels_float, labels_float.T)  # [2N, 2N]
        total_labels = label_counts + label_counts.T  # [2N, 2N]
        sl = same_labels / (total_labels - same_labels + 1e-8)  # [2N, 2N]
        
        # 3. Weighted combination.
        target_similarity = lambdap * sm + (1 - lambdap) * sl
        
        # 4. Force the diagonal to 1: a molecule is identical to itself.
        target_similarity.fill_diagonal_(1.0)
        
        return target_similarity
    
    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if self.hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-6)
        return optimizer

    # def training_step(self, batch, batch_idx):
    #     #if(batch_idx==0):
    #         #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',torch.cuda.memory_summary())
    #     idx = batch[0]
    #     mask = batch[1]
    #     odor_labels = batch[2]
    #     _ = batch[3]
    #     _ = batch[4]
    #     fps_labels = batch[5]
    #     fps_masks = batch[6] # Fingerprint prediction mask.
    #     fcl_mask = batch[7] # Contrastive learning mask.

    #     loss = 0

    #     token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
    #     x = self.drop(token_embeddings)
    #     x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        
    #     token_embeddings = x
    #     loss_input = token_embeddings[:, 0, :]
    #     ###
    #     source, all_logits = self.net_for_fingerprint(token_embeddings[:, 0, :])
    #     fps_logits = all_logits * fps_masks.unsqueeze(-1)
        
    #     logits, labels = self.get_info_nce_loss(source)
    #     fcl_mask = fcl_mask.to(x.device)
    #     labels = labels + fcl_mask # XCLR may not need to mask incidental similar molecules; they are not treated as negatives.
    #     contrastive_loss = self.criterion(logits, labels)
    #     del logits  # Release the (2N, 2N) computation graph to prevent GPU memory accumulation across batches.
    #     torch.cuda.empty_cache()  # Optional, but recommended.
        
    #     fps_labels = fps_labels.to(x.device)
    #     fps_pred_loss = self.bce_loss(fps_logits, fps_labels)
    #     print("fingerPrintLoss: ",fps_pred_loss," contrastive_loss: ",contrastive_loss) # Print both loss components per batch when testing their relative weights.
    #     loss = fps_pred_loss + contrastive_loss/2
         
    #     self.log('contrastive_loss', contrastive_loss, on_step=True)
    #     self.log('fps_prediction', fps_pred_loss, on_step=True)

    #     return {"loss": loss}
    

    def training_step(self, batch, batch_idx):
        idx = batch[0]
        mask = batch[1]
        odor_labels = batch[2]
        _ = batch[3]
        _ = batch[4]
        fps_labels = batch[5]
        fps_masks = batch[6] # Fingerprint prediction mask.
        fcl_mask = batch[7] # Contrastive learning mask.
        
        # Forward pass.
        token_embeddings = self.tok_emb(idx)
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        
        # Get molecular representations with shape [2N, emb_dim].
        mol_features = x[:, 0, :]
        
        # Compute the model distribution x.
        model_similarity = self.get_model_distribution(mol_features) # Compute the model output distribution for MSE.

        # The next two lines compute the model output distribution for KL divergence.
        temperature = 0.1  # Tunable temperature: smaller values produce a sharper distribution.
        model_prob = self.get_model_distribution01(mol_features, temperature=temperature)
        
        # Compute the target distribution y.
        lambdap = 0.8  # Can be a learnable parameter or a hyperparameter.
        target_similarity = self.get_target_distribution03( # Compute the target distribution for MSE.
            fps_labels, odor_labels, lambdap=lambdap
        )

        target_prob = self.get_target_distribution02( # Compute the target distribution for KL divergence.
            fps_labels, odor_labels, lambdap=lambdap
        )
        
        # Compute MSE loss.
        # Optionally mask invalid samples.
        valid_mask = (fcl_mask != -100).float().unsqueeze(0) * (fcl_mask != -100).float().unsqueeze(1)
        contrastive_loss = F.mse_loss( # Compute the MSE loss.
            model_similarity * valid_mask,
            target_similarity * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)

        # Build the valid [2N, 2N] mask; the following commented code computes KL divergence loss.
        # valid_mask = (fcl_mask != -100).float()
        # valid_mask_matrix = valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)
        # # Compute contrastive loss using KL divergence.
        # # F.kl_div expects log probabilities as input and probabilities as the target.
        # model_log_prob = (model_prob + 1e-8).log()  # Add epsilon to avoid log(0).
        # kl_loss = F.kl_div(
        #     model_log_prob,
        #     target_prob,
        #     reduction='none'  # Compute elementwise losses for masking.
        # )
        # contrastive_loss = (kl_loss * valid_mask_matrix).sum() / (valid_mask_matrix.sum() + 1e-8) # Apply the mask and average.
        
        # Fingerprint prediction loss, retaining the existing behavior.
        source, all_logits = self.net_for_fingerprint(mol_features)
        fps_logits = all_logits * fps_masks.unsqueeze(-1)
        fps_pred_loss = self.bce_loss(fps_logits, fps_labels)
        
        # Total loss.
        total_loss = fps_pred_loss + 8 * contrastive_loss  # Adjustable loss weight.
        
        self.log('contrastive_loss', contrastive_loss, on_step=True)
        self.log('fps_prediction', fps_pred_loss, on_step=True)
        
        return {"loss": total_loss}


    # def validation_step(self, val_batch, batch_idx, dataset_idx):
    #     idx =     val_batch[0]
    #     mask = val_batch[1]
    #     _ = val_batch[2]
    #     _ = val_batch[3]
    #     _ = val_batch[4]
    #     fps_labels = val_batch[5]
    #     fps_masks = val_batch[6]
    #     fcl_mask = val_batch[7]

    #     loss = 0
        
    #     token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
    #     x = self.drop(token_embeddings)
    #     x = self.blocks(x, length_mask=LM(mask.sum(-1)))
    #     token_embeddings = x
        
    #     ###
    #     source, all_logits = self.net_for_fingerprint(token_embeddings[:, 0, :])
    #     fps_logits = all_logits * fps_masks.unsqueeze(-1)
    #     fps_logits.shape
        
    #     logits, labels = self.get_info_nce_loss(source)
    #     fcl_mask = fcl_mask.to(x.device)
    #     labels = labels + fcl_mask # XCLR may not need to mask incidental similar molecules; they are not treated as negatives.
    #     contrastive_loss = self.criterion(logits, labels)
        
    #     fps_labels = fps_labels.to(x.device)
    #     fps_pred_loss = self.bce_loss(fps_logits, fps_labels)
        
    #     loss = fps_pred_loss + contrastive_loss * 0.5
          
    #     self.log('train_loss', loss, on_step=True)
    #     self.log('contrastive_loss', contrastive_loss, on_step=True)
    #     self.log('fps_prediction', fps_pred_loss, on_step=True)
    #     return {
    #         "val_loss": loss,
    #         "dataset_idx": dataset_idx,
    #     }
    
    def _evaluation_step(self, val_batch):
        idx = val_batch[0]
        mask = val_batch[1]
        odor_labels = val_batch[2]
        fps_labels = val_batch[5]
        fps_masks = val_batch[6]
        fcl_mask = val_batch[7]
        
        token_embeddings = self.tok_emb(idx)
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        
        mol_features = x[:, 0, :]
        
        model_similarity = self.get_model_distribution(mol_features)
        target_similarity = self.get_target_distribution01(
            fps_labels, odor_labels, lambdap=0.8
        )
        
        valid_mask = (fcl_mask != -100).float().unsqueeze(0) * (fcl_mask != -100).float().unsqueeze(1)
        contrastive_loss = F.mse_loss(
            model_similarity * valid_mask,
            target_similarity * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)
        
        source, all_logits = self.net_for_fingerprint(mol_features)
        fps_logits = all_logits * fps_masks.unsqueeze(-1)
        fps_pred_loss = self.bce_loss(fps_logits, fps_labels)
        
        total_loss = fps_pred_loss + 8 * contrastive_loss
        
        return {
            "loss": total_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "fps_prediction": fps_pred_loss.detach(),
            "batch_size": idx.size(0),
        }

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def _evaluation_epoch_end(self, outputs, split):
        if not outputs:
            raise ValueError(f"The {split} dataset produced no evaluation batches.")
        sample_count = sum(output["batch_size"] for output in outputs)
        metrics = {
            f"{self.hparams.measure_name}_{split}_{name}": sum(
                output[name] * output["batch_size"] for output in outputs
            ) / sample_count
            for name in ("loss", "contrastive_loss", "fps_prediction")
        }
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        print(f"{split}: {metrics}")
        append_to_file(
            os.path.join(self.hparams.results_dir, f"results_{split}.csv"),
            f"{self.hparams.measure_name},{self.current_epoch},"
            + ",".join(str(value.item()) for value in metrics.values()),
        )
        return metrics

    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "valid")

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "test")


# Convert semicolon-separated odor labels from the CSV into a multi-label binary matrix and PyTorch tensors.
def get_dataset(data_root, filename, dataset_len, aug, measure_name, normalize=False, tokenizer=None, similar_molecules=None, randomize_similar=True):
    df = pd.read_csv(os.path.join(data_root, filename))# 1. Load the dataset.
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    
    # 2. Split semicolon-separated odor labels into lists; use an empty list for missing values.
    df["odor_list"] = df["odor"].apply(
        lambda x: [tag.strip() for tag in x.split(";")] if isinstance(x, str) else []
    )
    # 3. Initialize the multi-label encoder with 27 explicit classes to avoid missing labels.
    target_labels = [
        "fruity", "alcoholic", "herbal", "vegetable", "woody", "lipidic", "fermented", 
        "baked-roasted", "sweet", "smoky", "spicy", "alliaceous", "stink", "sour", 
        "phenolic", "floral", "savory", "petroleum-solvent", "fresh", "briny", "animalic", 
        "resinous-balsamic", "metallic", "musty-earthy", "dusty", "nutty", "odorless"
    ]  # 27 label classes.
    mlb = MultiLabelBinarizer(classes=target_labels)
    label_matrix = mlb.fit_transform(df["odor_list"])# A 27-dimensional binary vector per sample indicates which odor labels are present.

    # 4. Append the encoded label columns to the original dataframe.
    label_df = pd.DataFrame(label_matrix, columns=mlb.classes_)
    df = pd.concat([df, label_df], axis=1)

    # 5. Convert to PyTorch tensors for the model.
    # Final labels: floating-point binary tensor with shape (number of samples, 27).
    label_tensor = torch.tensor(label_matrix, dtype=torch.float32)
    print("标签张量形状：", label_tensor.shape)  # For example, (1000, 27).

    label_matrix = mlb.fit_transform(df["odor_list"])
    
    # Store the 27-dimensional labels in a dedicated column without overwriting original columns.
    df["odormultilabel"] = list(label_matrix)  # Each row contains a 27-dimensional list.
    
    # Disable normalization; multi-label targets do not require it.
    normalize = False  # Force normalization off.

    smiles_to_label = {} # Map canonical SMILES to odor-label vectors.
    for idx, row in df.iterrows():
        smi = row['smiles']
        canonical_smi = normalize_smiles(smi, canonical=True, isomeric=False)
        if canonical_smi:
            label_vec = row["odormultilabel"]  # 27-dimensional list.
            smiles_to_label[canonical_smi] = label_vec
    
    dataset = PropertyPredictionDataset(
        df, 
        measure_name="odormultilabel",  # Use the newly added label column.
        tokenizer=tokenizer,
        aug=aug, 
        normalize=normalize,
        smiles_to_label=smiles_to_label,
        similar_molecules=similar_molecules,
        randomize_similar=randomize_similar,
    )
    return dataset

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df,  measure_name, tokenizer, aug=True, normalize=False, smiles_to_label=None, similar_molecules=None, randomize_similar=True):
        df = df.dropna()  # TODO - Check why some rows are na
        self.df = df
        all_smiles = df["smiles"].tolist()
        self.original_smiles = []
        self.original_canonical_map = {
            smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
        }

        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.smiles_to_label = smiles_to_label

        self.is_multilabel = measure_name == "odormultilabel"
        if self.is_multilabel:
            # Use the 27-dimensional multi-label vector directly and disable normalization.
            self.measure_mean = 0
            self.measure_std = 1
            # Index labels directly instead of building measure_map.
        else:
            if normalize:
                self.measure_mean = df[measure_name].mean()
                self.measure_std = df[measure_name].std()
                df[measure_name] = (df[measure_name] - self.measure_mean)/self.measure_std
            else:
                self.measure_mean = 0
                self.measure_std = 1
            # df[measure_name] = (df[measure_name]-df[measure_name].min())/(df[measure_name].max()-df[measure_name].min())
            all_measures = df[measure_name].tolist()
            self.measure_map = {all_smiles[i]: all_measures[i] for i in range(len(all_smiles))}

        # Get the canonical smiles
        # Convert the keys to canonical smiles if not already

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")

        self.aug = aug
        self.is_measure_available = "measure" in df.columns

        self.similar_molecules = similar_molecules or {}
        self.randomize_similar = randomize_similar

    def __getitem__(self, index): # Retrieve a molecular sample by index.
        original_smiles = self.original_smiles[index]
        canonical_smiles = self.original_canonical_map[original_smiles]
        # Candidates must belong to the current split; always use the top-ranked candidate during evaluation.
        candidates = [
            smi for smi in self.similar_molecules.get(original_smiles, {}).get('smiles', [])
            if smi in self.original_canonical_map
        ]
        if candidates:
            sim_smiles = random.choice(candidates) if self.randomize_similar else candidates[0]
        else:
            # Retain samples with empty pools; collate masks duplicate molecules in the contrastive loss.
            sim_smiles = original_smiles
        sim_canonical_smiles = self.original_canonical_map[sim_smiles]

        candidates = [smi for smi in candidates if smi != sim_smiles]
        
        # Return 27-dimensional label vectors.
        label_vec_orig = self.df.iloc[index]["odormultilabel"]  # Existing 27-dimensional odor-label vector of the original molecule.
        if self.smiles_to_label and sim_canonical_smiles in self.smiles_to_label:
            label_vec_sim = self.smiles_to_label[sim_canonical_smiles] # Odor-label vector of the similar molecule.
        else:
            # If unavailable, fall back to the original molecule's labels or an all-zero vector.
            label_vec_sim = label_vec_orig  # Alternatively, use [0] * 27.
        # print("Label vector length and shape in __getitem__: ", len(label_vec), torch.tensor(label_vec).shape) # Print the label vector shape.
        return (canonical_smiles, # Canonical SMILES.
                label_vec_orig, # Odor-label vector of the original molecule.
                sim_canonical_smiles, # Canonical SMILES of the similar molecule.
                original_smiles, # Original SMILES of the original molecule.
                sim_smiles, # Original SMILES of the similar molecule.
                label_vec_sim, # Odor-label vector of the similar molecule.
                candidates, #
                )
        
    def __len__(self):
        return len(self.original_smiles)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        #self.smiles_emb_size = hparams.n_embd
        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.dataset_name = hparams.dataset_name
        
        self.fpgen = AllChem.GetMorganGenerator(radius=2)


    @staticmethod
    def get_split_dataset_filename(dataset_name, split):
        if dataset_name in ['qm9', 'qm8', 'esol', 'freesolv', 'lipo', 'mop_base', 'mop_xclr', 'mop', 'mop_cleanbase', 'mop_cleanxclr']:
            return dataset_name + "_" + split + ".csv"
        else:
            return split + '.csv'
        
    def _load_split_dataset(self, split):
        is_train = split == "train"
        pool_split = "val" if split == "valid" else split
        pool_path = (
            FEATURE_EXTRACTION_DIR / self.dataset_name
            / f'similar_molecules_top5_{pool_split}.json'
        )
        with open(pool_path) as pool_file:
            similar_molecules = json.load(pool_file)
        return get_dataset(
            self.hparams.data_root,
            self.get_split_dataset_filename(self.dataset_name, split),
            self.hparams.train_dataset_length if is_train else self.hparams.eval_dataset_length,
            aug=self.hparams.aug if is_train else False,
            measure_name=self.hparams.measure_name,
            normalize=False,
            tokenizer=self.tokenizer,
            similar_molecules=similar_molecules,
            randomize_similar=is_train,
        )

    def setup(self, stage=None):
        # Load only training and validation data during fit; load test data and its positive pool only for final testing.
        if stage in (None, "fit"):
            self.train_ds = self._load_split_dataset("train")
            self.val_ds = self._load_split_dataset("valid")
        if stage in (None, "test"):
            self.test_ds = self._load_split_dataset("test")


    def collate(self, batch):
        tmp1 = []
        tmp2 = []
        all_labels_orig = []  # Labels of the original molecules.
        all_labels_sim = []   # Labels of the similar molecules.
        fps_labels = []
        fps_masks = []
        
        jointed_candidates = []
        origin_tmp1 = []
        origin_tmp2 = []
        for smile in batch:
            origin_tmp1.append(smile[3])
            origin_tmp2.append(smile[4])
            jointed_candidates += smile[-1]
            
            tmp1.append(smile[0])
            tmp2.append(smile[2])
            all_labels_orig.append(smile[1])   # Labels of the original molecules.
            all_labels_sim.append(smile[5])    # Labels of the similar molecules.

        all_smiles = tmp1 + tmp2 # Canonical SMILES of the original and similar molecules.
        original_smiles = origin_tmp1 + origin_tmp2 # Original SMILES of the original and similar molecules.
        all_labels = all_labels_orig + all_labels_sim # Label vectors of the original and similar molecules.
        all_labels = torch.tensor(all_labels,  dtype=torch.float32) # Convert to a PyTorch tensor.
        jointed_candidates = list(set(jointed_candidates))
        
        for smile in all_smiles:
            try:
                mol = Chem.MolFromSmiles(smile)
                fp = self.fpgen.GetFingerprint(mol)
                # fp = AllChem.GetMorganFingerprint(mol, radius=2)
                fps_labels.append(torch.tensor(fp).float())
                fps_masks.append(1.)
            except Exception as e:
                print(f"Failed to process smile: {smile}, error: {e}")
                fps_labels.append(torch.zeros(2048).float())
                fps_masks.append(0.)
        
        fcl_mask = torch.zeros(len(all_smiles))
        counted = {x:all_smiles.count(x) for x in all_smiles}
        
        for i, smi in enumerate(all_smiles):
            if (original_smiles[i] in jointed_candidates) or (counted[smi] != 1):
                fcl_mask[i] = -100
                
        #all_labels = torch.tensor(all_labels)
        tokens = self.tokenizer.batch_encode_plus(all_smiles, padding=True, add_special_tokens=True)
        
        fps_labels = torch.stack(fps_labels)
        fps_masks = torch.tensor(fps_masks)

        return (torch.tensor(tokens['input_ids']), 
                torch.tensor(tokens['attention_mask']), 
                all_labels, # Tensor of odor-label vectors.
                0,  # Multi-label targets are not normalized; evaluation does not depend on the training dataset object.
                1,
                fps_labels, 
                fps_masks, 
                fcl_mask.long()
                )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )



class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
                #filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")

def main():
    
    margs = args.parse_args()

    margs.dataset_name = margs.dataset_name.strip() #clean up any extra spaces
    margs.measure_name = margs.measure_name.strip()

    # ============ Prefit SVD ============
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fpgen = AllChem.GetMorganGenerator(radius=2)
    train_filename = PropertyPredictionDataModule.get_split_dataset_filename(margs.dataset_name, "train")
    train_df = pd.read_csv(os.path.join(margs.data_root, train_filename))
    if margs.train_dataset_length:
        train_df = train_df.head(margs.train_dataset_length)
    
    fps_list = []
    for smi in train_df["smiles"].sample(n=min(10000, len(train_df)), random_state=42):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = fpgen.GetFingerprint(mol)
                fps_list.append(np.array(fp, dtype=np.float32))
        except:
            continue
    
    svd_reducer = None
    if len(fps_list) >= 100:
        fps_array = np.stack(fps_list)
        svd_reducer = SVDFingerprintReducer(out_dim=768, random_state=42)
        svd_reducer.fit(fps_array)
        # Optionally save the matrix for later loading.
        # torch.save(svd_reducer.W, f"svd_W_{margs.dataset_name}.pt")
    # ===================================

    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    print('pos_emb_type is {}'.format(pos_emb_type))

    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))

    print(run_name)
    datamodule = PropertyPredictionDataModule(margs)
    margs.dataset_names = ["valid"]
    margs.run_name = run_name

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.measure_name)
    margs.checkpoint_root = checkpoint_root
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models")
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"{margs.measure_name}_valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        period=1,
        dirpath=checkpoint_dir,
        filename="best",
        verbose=True,
    )

    print(margs)

    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        #version=run_name,
        name="lightning_logs",
        default_hp_metric=False,
    )

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    seed.seed_everything(margs.seed)

    if margs.seed_path == '':
        print("# training from scratch")
        model = LightningModule(margs, tokenizer, svd_reducer=svd_reducer)
    else:
        print("# loaded pre-trained model from {args.seed_path}")
        model = LightningModule.load_from_checkpoint(
            margs.seed_path, strict=False, config=margs, tokenizer=tokenizer,
            svd_reducer=svd_reducer,
        )


    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from_checkpoint = None
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from scratch")

    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
        accelerator="gpu"
    )

    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    best_checkpoint_path = checkpoint_callback.best_model_path
    if not best_checkpoint_path or checkpoint_callback.best_model_score is None:
        raise RuntimeError("No best checkpoint was selected from the validation set.")
    # Explicitly restore the weights selected on validation data before final testing.
    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["state_dict"])
    print(f"Final test using best validation checkpoint: {best_checkpoint_path}")
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    with open(os.path.join(results_dir, "final_test_metrics.json"), "w", encoding="utf-8") as result_file:
        json.dump({
            "best_checkpoint": best_checkpoint_path,
            "best_epoch": best_checkpoint.get("epoch"),
            "monitor": checkpoint_callback.monitor,
            "best_validation_score": float(checkpoint_callback.best_model_score),
            "test_metrics": test_results,
        }, result_file, ensure_ascii=False, indent=2)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()
