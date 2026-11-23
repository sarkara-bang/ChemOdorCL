from rdkit.Chem import MolFromSmiles, rdmolops
from rdkit.Chem import AllChem, AddHs
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm
from rdkit import DataStructs
import json
import torch
import math
import random
import copy
import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / 'data'

parser = argparse.ArgumentParser(description='Find similar molecules for FCL')
parser.add_argument('--task_name', type=str, default='bbbp')

args = parser.parse_args()

def get_data(task_name):    
    if task_name == 'bace':
        return get_bace()
    elif task_name == 'bbbp':
        return get_bbbp()
    elif task_name == 'hiv':
        return get_hiv()
    elif task_name == 'tox21':
        return get_tox21()
    elif task_name == 'sider':
        return get_sider()
    elif task_name == 'clintox':
        return get_clintox()
    elif task_name == 'esol':
        return get_esol()
    elif task_name == 'freesolv':
        return get_freesolv()
    elif task_name == 'lipo':
        return get_lipo()
    elif task_name == 'qm9':
        return get_qm9()
    elif task_name == 'mop':
        return get_mop()
    elif task_name == 'mop_aug':
        return get_mop_aug()
    elif task_name == 'mop_xclr':
        return get_mop_xclr()
    elif task_name == 'mop_cleanxclr':  # Added support for the mop_base dataset.
        return get_mop_cleanxclr()
    else:
        print(f'[{task_name}] is not supported!!')
        raise

# Keep the existing get_xxx logic; ensure each CSV contains an odor column.
def get_bace():
    train = pd.read_csv(DATA_ROOT / 'bace/train.csv')
    val = pd.read_csv(DATA_ROOT / 'bace/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'bace/test.csv')
    return train, val, test

def get_bbbp():
    train = pd.read_csv(DATA_ROOT / 'bbbp/train.csv')
    val = pd.read_csv(DATA_ROOT / 'bbbp/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'bbbp/test.csv')
    return train, val, test

def get_hiv():
    train = pd.read_csv(DATA_ROOT / 'hiv/train.csv')
    val = pd.read_csv(DATA_ROOT / 'hiv/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'hiv/test.csv')
    return train, val, test

def get_tox21():
    train = pd.read_csv(DATA_ROOT / 'tox21/train.csv')
    val = pd.read_csv(DATA_ROOT / 'tox21/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'tox21/test.csv')
    return train, val, test

def get_sider():
    train = pd.read_csv(DATA_ROOT / 'sider/train.csv')
    val = pd.read_csv(DATA_ROOT / 'sider/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'sider/test.csv')
    return train, val, test

def get_clintox():
    train = pd.read_csv(DATA_ROOT / 'clintox/train.csv')
    val = pd.read_csv(DATA_ROOT / 'clintox/valid.csv')
    test = pd.read_csv(DATA_ROOT / 'clintox/test.csv')
    return train, val, test

def get_esol():
    train = pd.read_csv(DATA_ROOT / 'esol/esol_train.csv')
    val = pd.read_csv(DATA_ROOT / 'esol/esol_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'esol/esol_test.csv')
    return train, val, test

def get_lipo():
    train = pd.read_csv(DATA_ROOT / 'lipo/lipo_train.csv')
    val = pd.read_csv(DATA_ROOT / 'lipo/lipo_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'lipo/lipo_test.csv')
    return train, val, test

def get_freesolv():
    train = pd.read_csv(DATA_ROOT / 'freesolv/freesolv_train.csv')
    val = pd.read_csv(DATA_ROOT / 'freesolv/freesolv_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'freesolv/freesolv_test.csv')
    return train, val, test

def get_qm9():
    train = pd.read_csv(DATA_ROOT / 'qm9/qm9_train.csv')
    val = pd.read_csv(DATA_ROOT / 'qm9/qm9_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'qm9/qm9_test.csv')
    return train, val, test

def get_mop():
    train = pd.read_csv(DATA_ROOT / 'mop/mop_small_train.csv')
    val = pd.read_csv(DATA_ROOT / 'mop/mop_small_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'mop/mop_small_test.CSV')
    return train, val, test

def get_mop_aug():
    train = pd.read_csv(DATA_ROOT / 'mop_aug/mop_train.csv')
    val = pd.read_csv(DATA_ROOT / 'mop_aug/mop_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'mop_aug/mop_test.csv')
    return train, val, test

def get_mop_xclr():
    train = pd.read_csv(DATA_ROOT / 'mop_xclr/mop_xclr_train.csv')
    val = pd.read_csv(DATA_ROOT / 'mop_xclr/mop_xclr_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'mop_xclr/mop_xclr_test.csv')
    return train, val, test

def get_mop_cleanxclr():
    train = pd.read_csv(DATA_ROOT / 'mop_cleanxclr/mop_cleanxclr_train.csv')
    val = pd.read_csv(DATA_ROOT / 'mop_cleanxclr/mop_cleanxclr_valid.csv')
    test = pd.read_csv(DATA_ROOT / 'mop_cleanxclr/mop_cleanxclr_test.csv')
    return train, val, test

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def cos_sim_tensor(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.matmul(a_norm, b_norm.t())
        return sim_mt

# Load the data and preprocess the odor column.
train, val, test = get_data(args.task_name)

# Deduplicate the training data and extract odor labels as sets.
train_smiles = train['smiles'].to_list()
train_odor = train['odor'].fillna('').to_list()  # Fill missing values to avoid errors.
train_canonical_smiles = train['canonical_smiles'].to_list()  # Added: canonical SMILES.
# Deduplicate SMILES while keeping the corresponding odor labels aligned.
train_unique = pd.DataFrame({'smiles': train_smiles, 'odor': train_odor, 'canonical_smiles': train_canonical_smiles}).drop_duplicates(subset=['smiles'])
train_smiles = train_unique['smiles'].to_list()
# Split odor strings into sets for subsequent intersection checks.
train_odor_list = [set(odor.split(';')) if odor else set() for odor in train_unique['odor'].to_list()]
train_canonical_smiles_list = train_unique['canonical_smiles'].to_list()  # Added: list of canonical SMILES.

# Apply the same preprocessing to validation/test data if needed.
val_smiles = val['smiles'].to_list()
val_odor = val['odor'].fillna('').to_list()
val_canonical_smiles = val['canonical_smiles'].to_list()  # Added: canonical SMILES.
val_unique = pd.DataFrame({'smiles': val_smiles, 'odor': val_odor, 'canonical_smiles': val_canonical_smiles}).drop_duplicates(subset=['smiles'])
val_smiles = val_unique['smiles'].to_list()
val_odor_list = [set(odor.split(';')) if odor else set() for odor in val_unique['odor'].to_list()]
val_canonical_smiles_list = val_unique['canonical_smiles'].to_list()  # Added: list of canonical SMILES.

test_smiles = test['smiles'].to_list()
test_odor = test['odor'].fillna('').to_list()
test_canonical_smiles = test['canonical_smiles'].to_list()  # Added: canonical SMILES.
test_unique = pd.DataFrame({'smiles': test_smiles, 'odor': test_odor, 'canonical_smiles': test_canonical_smiles}).drop_duplicates(subset=['smiles'])
test_smiles = test_unique['smiles'].to_list()
test_odor_list = [set(odor.split(';')) if odor else set() for odor in test_unique['odor'].to_list()]
test_canonical_smiles_list = test_unique['canonical_smiles'].to_list()  # Added: list of canonical SMILES.

def filter_unique_canonical(smiles_list, sim_list, canonical_smiles_list, bucket_canonical_list):
    """
    Deduplicate molecules by canonical SMILES, retaining the highest-similarity molecule for each.
    """
    seen_canonical = {}
    filtered_smiles = []
    filtered_sim = []
    
    for smiles, sim, canonical_smiles in zip(smiles_list, sim_list, canonical_smiles_list):
        # Retain only the molecule with the highest similarity.
        if canonical_smiles not in seen_canonical or sim > seen_canonical[canonical_smiles][1]:
            seen_canonical[canonical_smiles] = (smiles, sim)
    
    # Sort by similarity and extract the results.
    sorted_items = sorted(seen_canonical.items(), key=lambda x: x[1][1], reverse=True)
    for canonical_smiles, (smiles, sim) in sorted_items:
        filtered_smiles.append(smiles)
        filtered_sim.append(sim)
    
    return filtered_smiles, filtered_sim

def get_similar(bucket, odor_list, canonical_smiles_list, split='train'):
    """
    Updated function with an additional canonical_smiles_list parameter:
    - bucket: list of molecular SMILES.
    - odor_list: list of odor-label sets, one per molecule.
    - canonical_smiles_list: list of canonical SMILES, one per molecule.
    - split: dataset split name used in output filenames.
    
    Function logic:
    1. Compute similarity only for molecules sharing odor labels with the anchor.
    2. Exclude molecules with the same canonical SMILES as the anchor, including augmented variants.
    3. Also deduplicate the similar-molecule list by canonical SMILES.
    """
    fpgen = AllChem.GetMorganGenerator(radius=2)

    # Generate fingerprints.
    fps = []
    fps_tensors = []
    for i, smiles in enumerate(tqdm(bucket, desc=f"Generating fingerprints for {split}")):
        try:
            mol = MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES detected: {smiles}")
                fp = None
                ftensor = torch.zeros(2048).float()
            else:
                fp = fpgen.GetFingerprint(mol)
                ftensor = torch.Tensor(fp.ToList()).float()
            fps.append(fp)
            fps_tensors.append(ftensor)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            fps.append(None)
            ftensor = torch.zeros(2048).float()
            fps_tensors.append(ftensor)

    fps_tensors = torch.stack(fps_tensors).cuda()

    # Search among molecules sharing odor labels and having different canonical SMILES.
    most_sims = {}
    for i, fp in enumerate(tqdm(fps, desc=f"Finding similar molecules for {split}")):
        if fp is None:
            most_sims[bucket[i]] = {'smiles': [], "similarity": []}
            continue
            
        # Step 1: collect eligible molecule indices.
        anchor_odor = odor_list[i]
        anchor_canonical_smiles = canonical_smiles_list[i]
        valid_indices = []
        
        for j in range(len(bucket)):
            if i == j:  # Skip the anchor itself.
                continue
                
            # Exclusion 1: no odor labels in common with the anchor.
            if not (anchor_odor & odor_list[j]):
                continue
                
            # Exclusion 2: the same canonical SMILES as the anchor, including augmented variants.
            if canonical_smiles_list[j] == anchor_canonical_smiles:
                continue
                
            valid_indices.append(j)
        
        # Keep all candidates if fewer than 50 are available; record empty lists if none exist.
        if not valid_indices:
            most_sims[bucket[i]] = {'smiles': [], "similarity": []}
            continue
        
        # Step 2: gather eligible fingerprint tensors and compute similarities.
        anchor_tensor = copy.deepcopy(fps_tensors[i]).reshape(1, -1)
        valid_tensors = fps_tensors[valid_indices]
        # Compute similarities between the anchor and all eligible molecules.
        sims = cos_sim_tensor(anchor_tensor, valid_tensors).squeeze()
        # Select the top k candidates, up to 50.
        topk_num = min(50, len(valid_indices))
        topk_sims, topk_valid_idx = torch.topk(sims, topk_num)
        
        # Step 3: map back to the original indices and collect the results.
        topk_sims = topk_sims.tolist()
        topk_valid_idx = topk_valid_idx.tolist()
        
        top_smiles = []
        top_sim = []
        for sim_val, valid_idx in zip(topk_sims, topk_valid_idx):
            original_idx = valid_indices[valid_idx]
            top_smiles.append(bucket[original_idx])
            top_sim.append(sim_val)
        
        # Step 4: deduplicate similar molecules by canonical SMILES.
        # Get the canonical SMILES of these similar molecules.
        similar_canonical_smiles = [canonical_smiles_list[valid_indices[valid_idx]] for valid_idx in topk_valid_idx]
        
        # Retain the highest-similarity molecule for each canonical SMILES.
        filtered_smiles, filtered_sim = filter_unique_canonical(
            top_smiles, 
            top_sim, 
            similar_canonical_smiles,
            canonical_smiles_list
        )
        
        most_sims[bucket[i]] = {'smiles': filtered_smiles, "similarity": filtered_sim}
    
    # Extract the top 5/10, allowing for fewer available candidates.
    top1 = {}
    top5 = {}
    top10 = {}
    for key, value in tqdm(most_sims.items(), desc=f"Extracting top 5/10 for {split}"):
        top1[key] = {
            'smiles': value['smiles'][:1],
            'similarity': value['similarity'][:1]
        }
        top5[key] = {
            'smiles': value['smiles'][:5],
            'similarity': value['similarity'][:5]
        }
        top10[key] = {
            'smiles': value['smiles'][:10],
            'similarity': value['similarity'][:10]
        }

    # Save the results.
    output_dir = SCRIPT_DIR / args.task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json.dump(most_sims, open(output_dir / f'similar_molecules_{split}.json', 'w'), ensure_ascii=False)
    json.dump(top1, open(output_dir / f'similar_molecules_top1_{split}.json', 'w'), ensure_ascii=False)
    json.dump(top5, open(output_dir / f'similar_molecules_top5_{split}.json', 'w'), ensure_ascii=False)
    json.dump(top10, open(output_dir / f'similar_molecules_top10_{split}.json', 'w'), ensure_ascii=False)
    
    return most_sims, top1, top5, top10  # Return the results for subsequent use.

# Call the function with the additional canonical_smiles_list argument.
print("Processing training set...")
train_most_sims, train_top1, train_top5, train_top10 = get_similar(train_smiles, train_odor_list, train_canonical_smiles_list, 'train')
print("Processing validation set...")
val_most_sims, val_top1, val_top5, val_top10 = get_similar(val_smiles, val_odor_list, val_canonical_smiles_list, 'val')
print("Processing test set...")
test_most_sims, test_top1, test_top5, test_top10 = get_similar(test_smiles, test_odor_list, test_canonical_smiles_list, 'test')
