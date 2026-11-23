import time
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch import nn
import args
import torch.nn.functional as F
import os
import json
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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score, # Import the mAP calculation function.
) # Multi-label evaluation metrics.
from utils import normalize_smiles

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# ==========================================
# Global configuration: label order, loss weights, and inference thresholds.
# ==========================================
# 1. Define the global order of 21 labels to keep preprocessing and model outputs aligned.
TARGET_LABELS = [
    "fruity", "alcoholic", "herbal", "vegetable", "woody", "lipidic",  
    "baked-roasted", "sweet", "smoky", "spicy", "alliaceous", "stink",  
    "phenolic", "floral", "fresh", "animalic", 
    "resinous-balsamic", "musty-earthy", "dusty", "nutty", "odorless"
]

# 2. Class weights for the pos_weight argument of BCEWithLogitsLoss.
# Tuning: larger positive-class weights favor predicting the odor as present, increasing recall.
# Use 1.0 for frequent classes and larger weights, such as 3.0-15.0, for rarer classes.
# If loss becomes NaN early in training, reduce large weights that may cause exploding gradients.
CLASS_WEIGHTS_CONFIG = {
    "fruity": 1.0,             # 3757 samples (frequent class).
    "alcoholic": 4.0,          # 512
    "herbal": 1.0,             # 3778 samples (frequent class).
    "vegetable": 4.0,          # 508
    "woody": 1.35,              # 1474
    "lipidic": 1.0,            # 2004
    "baked-roasted": 3.35,      # 617
    "sweet": 1.0,              # 2296 samples (frequent class).
    "smoky": 3.35,              # 639
    "spicy": 2.0,              # 1154
    "alliaceous": 8.7,        # 230 samples (rarest class, assigned the highest weight).
    "stink": 5.0,              # 423
    "phenolic": 6.45,           # 310
    "floral": 1.0,             # 2494 samples (frequent class).
    "fresh":2.0,              # 1000
    "animalic": 2.75,           # 731
    "resinous-balsamic": 2.0,  # 980
    "musty-earthy": 2.25,       # 787
    "dusty": 5.0,              # 425
    "nutty": 2.5,              # 804
    "odorless": 2.5,           # 870
} 
# In the formulas below, max is fruity's sample count, 3757; cur is the current class count, e.g. 804 for nutty.
# Method 1 (low): weight = ln[(max/cur)*e].
# Method 2 (high): if cur > 1000, weight = (max-cur)/10000 + 1; if 500 < cur <= 1000, weight = ln[(max/cur)*e]; if cur <= 500, multiply ln[(max/cur)*e] by [(max-cur)/10000 + 1].
# Method 3 (medium): for cur < 800, weight = 0.8 * the weight from the high method.

# CLASS_WEIGHTS_CONFIG = {
#     "fruity": 1.0,             # 3757 samples (frequent class).
#     "alcoholic": 3.11,          # 512
#     "herbal": 1.0,             # 3778 samples (frequent class).
#     "vegetable": 3.11,          # 508
#     "woody": 2.0,              # 1474
#     "lipidic": 1.0,            # 2004
#     "baked-roasted": 2.93,      # 617
#     "sweet": 1.0,              # 2296 samples (frequent class).
#     "smoky": 2.93,              # 639
#     "spicy": 2.23,              # 1154
#     "alliaceous": 3.74,        # 230 samples (rarest class, assigned the highest weight).
#     "stink": 3.33,              # 423
#     "phenolic": 3.57,           # 310
#     "floral": 1.0,             # 2494 samples (frequent class).
#     "fresh": 2.23,              # 1000
#     "animalic": 2.72,           # 731
#     "resinous-balsamic": 2.23,  # 980
#     "musty-earthy": 2.56,       # 787
#     "dusty": 3.33,              # 425
#     "nutty": 2.56,              # 804
#     "odorless": 2.56,           # 870
# } 

# 3. Inference/validation thresholds for converting probabilities to binary labels.
# Tuning: introducing pos_weight tends to increase predicted probabilities.
# To balance precision, consider raising thresholds for rare classes from 0.5 to 0.6 or 0.7.
CLASS_THRESHOLDS_CONFIG = {label: 0.5 for label in TARGET_LABELS}
# Example: adjust the threshold for "alliaceous" separately if its metrics are poor.
# CLASS_THRESHOLDS_CONFIG["alliaceous"] = 0.65 

class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
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

        # ==========================================
        # Configure the loss, class weights, and thresholds.
        # ==========================================
        # Extract weights and thresholds in the global label order and convert them to tensors.
        pos_weight_list = [CLASS_WEIGHTS_CONFIG.get(label, 1.0) for label in TARGET_LABELS]
        thresholds_list = [CLASS_THRESHOLDS_CONFIG.get(label, 0.5) for label in TARGET_LABELS]
        
        pos_weight = torch.tensor(pos_weight_list, dtype=torch.float32)
        thresholds = torch.tensor(thresholds_list, dtype=torch.float32)
        
        # Register buffers so they move with the model between GPU/CPU without receiving gradients.
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('thresholds', thresholds)

         # Initialize weighted BCEWithLogitsLoss.
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.net = self.Net(
            config.n_embd, dims=config.dims, dropout=config.dropout,
        )


    # Fully connected layers after the molecular encoder produce label predictions.
    class Net(nn.Module):
        dims = [150, 50, 50, 2]


        def __init__(self, smiles_embed_dim, dims=dims, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)#768,768
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 21)  # Predict values for 21 label classes.

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

        z_pred = self.net.forward(smiles_emb)# z_pred has shape [batch_size, 27] and contains the predicted label values.
        measures = measures.float()

        return self.loss(z_pred, measures), z_pred, measures
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
        #optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        # return optimizer # Replace the Apex optimizer requiring CUDA extensions with a native PyTorch optimizer.
        import torch.optim as optim
        optimizer = optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=0.0  # Keep the original weight decay setting.
        )
        return optimizer

    # Core training step.
    def training_step(self, batch, batch_idx):
        idx = batch[0]
        mask = batch[1]
        targets = batch[2]
        measure_mean = batch[3]
        measure_std = batch[4]

        loss = 0
        loss_tmp = 0

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        
        token_embeddings = x
        loss_input = token_embeddings[:, 0, :]
        loss, pred, actual = self.get_loss(loss_input, targets)

        self.log('train_loss', loss, on_step=True)

        logs = {"train_loss": loss}

        return {"loss": loss}

    def _evaluation_step(self, val_batch):
        idx =     val_batch[0]
        mask = val_batch[1]
        targets = val_batch[2]
        measure_mean = val_batch[3]
        measure_std = val_batch[4]

        loss = 0
        loss_tmp = 0
        b, t = idx.size()
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        loss_input = token_embeddings[:, 0, :]
        loss, pred_logits, actual = self.get_loss(loss_input, targets)
        ### 
        # pred_logits = (pred * measure_std) + measure_mean # Do not denormalize pred_logits.
        
        # Compute sigmoid probabilities for evaluation metrics.
        pred_probs = torch.sigmoid(pred_logits)
        
        # Apply per-class thresholds from self.thresholds to produce predicted labels.
        # Per-class thresholds can balance precision and recall better than a fixed global threshold of 0.5.
        pred_labels = (pred_probs > self.thresholds).float() 
        return {
            "loss": loss.detach(),
            "pred_probs": pred_probs.detach(), # Return probabilities for AUC and related metrics.
            "pred_labels": pred_labels.detach(), # Return predicted labels.
            "actual_labels": actual.detach(), # Return ground-truth labels.
            "batch_size": targets.size(0),
        }

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch)

    def _evaluation_epoch_end(self, batch_outputs, dataset):
        if not batch_outputs:
            raise ValueError(f"The {dataset} dataset produced no evaluation batches.")
        sample_count = sum(output["batch_size"] for output in batch_outputs)
        avg_loss = sum(
            output["loss"] * output["batch_size"] for output in batch_outputs
        ) / sample_count

        # Aggregate predictions and ground-truth labels across all batches.
        all_pred_labels = torch.cat([x["pred_labels"] for x in batch_outputs], dim=0) # [N_total, 27]
        all_actual_labels = torch.cat([x["actual_labels"] for x in batch_outputs], dim=0) # [N_total, 27]
        all_pred_probs = torch.cat([x["pred_probs"] for x in batch_outputs], dim=0) # [N_total, 27]: probabilities used for mAP and AUC.

        # Move tensors to CPU and convert them to NumPy arrays for sklearn.
        all_pred_labels_np = all_pred_labels.cpu().numpy()
        all_actual_labels_np = all_actual_labels.cpu().numpy()
        all_pred_probs_np = all_pred_probs.cpu().numpy()

        # Compute multi-label metrics.
        # 1. mAP (Mean Average Precision)
        # average_precision_score expects probability scores or decision scores for positive class
        ap_scores = []
        n_labels = all_actual_labels_np.shape[1] # 27
        for i in range(n_labels):
            y_true = all_actual_labels_np[:, i]
            y_scores = all_pred_probs_np[:, i]
            # Check for single-class labels, which may cause unreliable AP values or warnings.
            unique_labels = np.unique(y_true)
            if len(unique_labels) == 1:
                # This label has either all positive or all negative samples in the validation set.
                if unique_labels[0] == 1: # All positive samples.
                    ap = 1.0
                elif unique_labels[0] == 0: # All negative samples.
                    # For cases where there are no positive samples, AP is undefined.
                    # sklearn might return 0 or raise depending on implementation/version.
                    # We might want to skip these or handle them differently.
                    # For now, setting to 0 is common practice when no positives exist.
                    ap = 0.0
                    # print(f"Warning: Label {i} has only negative samples. AP set to 0.0.")
                else:
                    # Should not happen if labels are binary (0, 1)
                    ap = float('nan')
            else:
                # Calculate AP normally if both classes are present
                ap = average_precision_score(y_true, y_scores)
            ap_scores.append(ap)
        # Filter out NaN values if any were added due to potential errors, though unlikely with our check above
        # ap_scores = [score for score in ap_scores if not np.isnan(score)]
        # For robustness against potential NaN (though prevented by checks above), use nanmean if necessary
        # However, standard mean is fine here as we handle single-class cases explicitly.
        map_score = np.mean(ap_scores)

        # ========== Updated micro-mAP calculation ==========
        y_true_flat = all_actual_labels_np.ravel()
        y_scores_flat = all_pred_probs_np.ravel()

        # Support older sklearn versions by explicitly handling the absence of positive samples.
        try:
            # Check whether positive samples are present.
            if np.sum(y_true_flat) == 0:
                # With no positive samples, AP is undefined; use 0 or 1 by convention.
                micro_map = 0.0  # Alternatively, use 1.0 depending on the task.
                print(f"Warning: No positive samples in {dataset} set, micro-mAP set to 0.0")
            else:
                # Compute AP normally without the zero_division argument.
                micro_map = average_precision_score(y_true_flat, y_scores_flat)
        except ValueError as e:
            # Handle other edge cases.
            print(f"Warning: AP calculation failed: {e}")
            micro_map = 0.0

        # Macro-AUROC
        auc_scores = []
        for i in range(all_actual_labels_np.shape[1]):
            y_true = all_actual_labels_np[:, i]
            y_scores = all_pred_probs_np[:, i]
            if len(np.unique(y_true)) < 2:
                auc_scores.append(0.5)  # Handle the single-class edge case.
            else:
                auc_scores.append(roc_auc_score(y_true, y_scores))
        macro_auroc = np.mean(auc_scores)
        
        # Micro-AUROC
        y_true_flat = all_actual_labels_np.ravel()
        y_scores_flat = all_pred_probs_np.ravel()
        if len(np.unique(y_true_flat)) < 2:
            micro_auroc = 0.5
        else:
            micro_auroc = roc_auc_score(y_true_flat, y_scores_flat)
        

        # 2. CP (Class-wise Precision - macro average)
        cp = precision_score(all_actual_labels_np, all_pred_labels_np, average='macro', zero_division=0)

        # 3. CR (Class-wise Recall - macro average)
        cr = recall_score(all_actual_labels_np, all_pred_labels_np, average='macro', zero_division=0)

        # 4. CF1 (Class-wise F1 - macro average)
        cf1 = f1_score(all_actual_labels_np, all_pred_labels_np, average='macro', zero_division=0)

        # 5. OP (Overall Precision - micro average)
        op = precision_score(all_actual_labels_np, all_pred_labels_np, average='micro', zero_division=0)

        # 6. OR (Overall Recall - micro average)
        or_score = recall_score(all_actual_labels_np, all_pred_labels_np, average='micro', zero_division=0)

        # 7. OF1 (Overall F1 - micro average)
        of1 = f1_score(all_actual_labels_np, all_pred_labels_np, average='micro', zero_division=0)

        # Optional: retain the previous metrics for comparison.
        # hamming = hamming_loss(all_actual_labels_np, all_pred_labels_np)
        # subset_acc = accuracy_score(all_actual_labels_np, all_pred_labels_np)
        # jaccard_macro = jaccard_score(all_actual_labels_np, all_pred_labels_np, average='macro', zero_division=0)
        # jaccard_micro = jaccard_score(all_actual_labels_np, all_pred_labels_np, average='micro', zero_division=0)
        # avg_auc = calculate_avg_auc(all_actual_labels_np, all_pred_probs_np) # You'd need to define this helper if keeping AUC

        # Include the new metrics in tensorboard_logs.
        tensorboard_logs = {
            f"{self.hparams.measure_name}_{dataset}_loss": avg_loss,
            f"{self.hparams.measure_name}_{dataset}_map": map_score,         # 1. mAP
            f"{self.hparams.measure_name}_{dataset}_micro_map": micro_map,   # Added: micro-mAP.
            f"{self.hparams.measure_name}_{dataset}_macro_auroc": macro_auroc,
            f"{self.hparams.measure_name}_{dataset}_micro_auroc": micro_auroc,         
            f"{self.hparams.measure_name}_{dataset}_cp": cp,                 # 2. CP (Macro Precision)
            f"{self.hparams.measure_name}_{dataset}_cr": cr,                 # 3. CR (Macro Recall)
            f"{self.hparams.measure_name}_{dataset}_cf1": cf1,               # 4. CF1 (Macro F1)
            f"{self.hparams.measure_name}_{dataset}_op": op,                 # 5. OP (Micro Precision)
            f"{self.hparams.measure_name}_{dataset}_or": or_score,           # 6. OR (Micro Recall)
            f"{self.hparams.measure_name}_{dataset}_of1": of1,               # 7. OF1 (Micro F1)
            # f"{self.hparams.measure_name}_{dataset}_hamming_loss": hamming, # Optionally retain.
            # f"{self.hparams.measure_name}_{dataset}_subset_accuracy": subset_acc, # Optionally retain.
        }

        # Log metrics.
        # self.logger.log_metrics(tensorboard_logs, self.global_step) # self.log normally handles metric logging.
        for k, v in tensorboard_logs.items():
            self.log(k, v, on_epoch=True, prog_bar=False, logger=True) # Record the metric with self.log.

        print(f"Evaluation Epoch {self.current_epoch}: Dataset {dataset}")
        print(f"  Macro mAP (Mean Average Precision): {map_score:.4f}")
        print(f"  Micro-mAP (global AP): {micro_map:.4f}")
        print(f"  Macro-AUROC: {macro_auroc:.4f}")
        print(f"  Micro-AUROC: {micro_auroc:.4f}") 
        print(f"  CP (Class-wise Precision - macro): {cp:.4f}")
        print(f"  CR (Class-wise Recall - macro): {cr:.4f}")
        print(f"  CF1 (Class-wise F1 - macro): {cf1:.4f}")
        print(f"  OP (Overall Precision - micro): {op:.4f}")
        print(f"  OR (Overall Recall - micro): {or_score:.4f}")
        print(f"  OF1 (Overall F1 - micro): {of1:.4f}")
        # print(f"  Hamming Loss: {hamming:.4f}") # Optional print
        # print(f"  Subset Accuracy: {subset_acc:.4f}") # Optional print

        # Build a row for the CSV results file.
        # Format: measure_name, epoch, loss, map, cp, cr, cf1, op, or, of1.
        # Separate values with commas and use suitable decimal precision.
        result_row = ( # All metrics except validation loss are higher-is-better, with an optimum of 1.0.
            f"{self.hparams.measure_name}, "  # Task name (measure_name).
            f"{self.current_epoch}, "        # Current epoch.
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_loss']:.6f}, "  # Validation loss (_loss).
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_map']:.6f}, "    # mAP
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_micro_map']:.6f}, "  # Micro-mAP
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_macro_auroc']:.6f}, "      # Macro-AUROC
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_micro_auroc']:.6f}, "      # Micro-AUROC
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_cp']:.6f}, "    # CP (Class-wise Precision - macro)
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_cr']:.6f}, "    # CR (Class-wise Recall - macro)
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_cf1']:.6f}, "   # CF1 (Class-wise F1 - macro)
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_op']:.6f}, "    # OP (Overall Precision - micro)
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_or']:.6f}, "    # OR (Overall Recall - micro)
            f"{tensorboard_logs[f'{self.hparams.measure_name}_{dataset}_of1']:.6f}"     # OF1 (Overall F1 - micro)
            # Do not append a comma after the last value.
        )

        # Write the results row using append_to_file.
        # Join the results directory and filename with os.path.join.
        append_to_file(
            filename=os.path.join(self.hparams.results_dir, f"results_{dataset}.csv"),
            line=result_row                                                   # Content to write.
        )
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "valid")

    def test_epoch_end(self, outputs):
        self._evaluation_epoch_end(outputs, "test")

    
# Convert semicolon-separated odor labels from the CSV into a multi-label binary matrix and PyTorch tensors.
def get_dataset(data_root, filename, dataset_len, aug, measure_name, normalize=False, tokenizer=None):
    df = pd.read_csv(os.path.join(data_root, filename))# 1. Load the dataset.
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    
    # 2. Split semicolon-separated odor labels into lists; use an empty list for missing values.
    df["odor_list"] = df["odor"].apply(
        lambda x: [tag.strip() for tag in x.split(";")] if isinstance(x, str) else []
    )
    # 3. Initialize the multi-label encoder with 21 explicit classes to avoid missing labels.
    target_labels = [
        "fruity", "alcoholic", "herbal", "vegetable", "woody", "lipidic",  
        "baked-roasted", "sweet", "smoky", "spicy", "alliaceous", "stink",  
        "phenolic", "floral", "fresh", "animalic", 
        "resinous-balsamic", "musty-earthy", "dusty", "nutty", "odorless"
    ]  # 21 label classes.

    # Use the global TARGET_LABELS to ensure a consistent class order.
    mlb = MultiLabelBinarizer(classes=TARGET_LABELS)
    label_matrix = mlb.fit_transform(df["odor_list"])# A 21-dimensional binary vector per sample indicates which odor labels are present.

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
    
    dataset = PropertyPredictionDataset(
        df, 
        measure_name="odormultilabel",  # Use the newly added label column.
        tokenizer=tokenizer,
        aug=aug, 
        normalize=normalize
    )
    return dataset

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df,  measure_name, tokenizer, aug=True, normalize=False):
        df = df.dropna()  # TODO - Check why some rows are na
        self.df = df
        all_smiles = df["smiles"].tolist()
        self.original_smiles = []
        self.original_canonical_map = {
            smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
        }

        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')

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

    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        canonical_smiles = self.original_canonical_map[original_smiles]
        
        if self.is_multilabel:
            # Return a 27-dimensional label vector.
            label_vec = self.df.iloc[index]["odormultilabel"]  # Existing 27-dimensional list.
            # print("Label vector length and shape in __getitem__: ", len(label_vec), torch.tensor(label_vec).shape) # Print the label vector shape.
            return canonical_smiles, label_vec
        else:
            return canonical_smiles, self.measure_map[original_smiles]

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

    @staticmethod
    def get_split_dataset_filename(dataset_name, split):
        return dataset_name + "_" + split + ".csv"

    def _load_split_dataset(self, split):
        is_train = split == "train"
        return get_dataset(
            self.hparams.data_root,
            self.get_split_dataset_filename(self.dataset_name, split),
            self.hparams.train_dataset_length if is_train else self.hparams.eval_dataset_length,
            aug=self.hparams.aug if is_train else False,
            measure_name=self.hparams.measure_name,
            normalize=False,
            tokenizer=self.tokenizer,
        )

    def setup(self, stage=None):
        # Load test data only for final testing; keep it out of validation during fit.
        if stage in (None, "fit"):
            self.train_ds = self._load_split_dataset("train")
            self.val_ds = self._load_split_dataset("valid")
        if stage in (None, "test"):
            self.test_ds = self._load_split_dataset("test")

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus(
            [ smile[0] for smile in batch], 
            padding=True, 
            add_special_tokens=True
        )
        # Check the label dimensions.
        first_label = batch[0][1]
        if isinstance(first_label, (list, np.ndarray)) and len(first_label) == 21:
            # Multi-label case: combine labels into a [B, 21] tensor.
            labels = torch.tensor([smile[1] for smile in batch], dtype=torch.float32)
        else:
            # Scalar case: retain the existing behavior.
            labels = torch.tensor([smile[1] for smile in batch], dtype=torch.float32)
        
        return (
            torch.tensor(tokens['input_ids']), 
            torch.tensor(tokens['attention_mask']), 
            labels,  # training_step uses batch[2] as labels with shape [batch_size, 27].
            0,  # Multi-label targets are not normalized; evaluation does not depend on the training dataset object.
            1,
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
    #margs.smiles_emb_size = datamodule.get_embedding_sizes()
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
        monitor=f"{margs.measure_name}_valid_map",
        mode="max",
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
        model = LightningModule(margs, tokenizer)
    else:
        print("# loaded pre-trained model from {args.seed_path}")
        # ori model laod way:  model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))
        model = LightningModule(margs, tokenizer)# Create a new model instance; net.final maps 768 features to 27 labels.
        # Load the pretrained checkpoint.
        checkpoint = torch.load(margs.seed_path, map_location=lambda storage, loc: storage)
        # Remove checkpoint entries belonging to the final layer.
        keys_to_remove = [key for key in checkpoint['state_dict'].keys() if key.startswith('net.final.')]
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]
            print(f"Removed {key} from checkpoint (will be randomly initialized)")

        # Load the modified state_dict; strict=False allows the final layer to remain unloaded.
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # The final layer intentionally retains its random initialization.


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
    # Explicitly restore the weights selected on validation data and evaluate the test set once.
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
