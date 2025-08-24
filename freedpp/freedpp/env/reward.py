from functools import partial
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch
from rdkit.Chem import rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED as qed_module
import os
from typing import List
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings('ignore')
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from scipy.spatial.distance import euclidean, cosine
from cats2d.rd_cats2d import CATS2D
from rdkit.DataStructs import BulkTanimotoSimilarity
import tmap
from map4.map4 import MAP4

dim = 1024

MAP4 = MAP4(dimensions=dim)
ENC = tmap.Minhash(dim)

class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.weight * self.reward(property)
        return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]

def SA(m):
    return sascorer.calculateScore(m)

def qed_mol(m): 
    return qed_module.qed(m)

def Brenk(m):
    params_brenk = FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog(params_brenk)
    return 1*catalog_brenk.HasMatch(m)

predefined_smiles = pd.read_csv('ER_antagonists_degraders.csv')['SMILES'].to_list()

cats_generator = CATS2D()

predefined_cats = []

for smi in predefined_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Invalid SMILES: {smi}, skipping")
        continue
    try:
        # Generate descriptors using the CATS2D instance
        cats_desc = cats_generator.getCATs2D(mol)
        predefined_cats.append(cats_desc)
    except AttributeError as e:
        print(f"Critical error: {e}. Check if 'cats_generator' is a CATS2D instance.")
        raise  # Halt execution to fix initialization
    except Exception as e:
        print(f"Error generating CATS for {smi}: {e}, skipping")
        continue

if not predefined_cats:
    raise ValueError("No valid predefined CATS descriptors were generated.")

predefined_cats = np.array(predefined_cats)

def CATS_Euclid(mol):
    """Calculate the minimum Euclidean distance between the generated molecule's CATS descriptor
    and all valid predefined CATS descriptors.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Generated molecule

    Returns:
        float: Minimum Euclidean distance (or infinity if generation fails)
    """
    try:
        generated_cats = cats_generator.getCATs2D(mol)
    except Exception as e:
        print(f"Error computing CATS for the molecule: {e}")
        return np.inf  # Return infinity to indicate failure

    if not isinstance(generated_cats, (list, np.ndarray)) or len(generated_cats) != 210:
        print("Invalid CATS descriptor for the generated molecule.")
        return np.inf

    generated_cats = np.array(generated_cats)
    distances = np.linalg.norm(predefined_cats - generated_cats, axis=1)
    return np.min(distances)

def CATS_Cosine(mol):
    """Calculate the maximum cosine similarity between the generated molecule's CATS descriptor
    and all valid predefined CATS descriptors.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Generated molecule

    Returns:
        float: Maximum cosine similarity (range: -1 to 1), or -inf if generation fails
    """
    try:
        generated_cats = cats_generator.getCATs2D(mol)
    except Exception as e:
        print(f"Error computing CATS for the molecule: {e}")
        return -np.inf  # Return -infinity to indicate failure

    if not isinstance(generated_cats, (list, np.ndarray)) or len(generated_cats) != 210:
        print("Invalid CATS descriptor for the generated molecule.")
        return -np.inf

    generated_cats = np.array(generated_cats)
    
    # Normalize both the predefined and generated CATS vectors
    predefined_norm = predefined_cats / np.linalg.norm(predefined_cats, axis=1, keepdims=True)
    generated_norm = generated_cats / np.linalg.norm(generated_cats)
    
    # Calculate cosine similarities (1 - cosine distance)
    similarities = 1 - np.array([cosine(predefined_norm[i], generated_norm) 
                               for i in range(len(predefined_norm))])
    
    return np.max(similarities)  # Return maximum similarity

predefined_fps = []

for smi in predefined_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Invalid SMILES: {smi}, skipping")
        continue
    try:
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        predefined_fps.append(fp)
    except Exception as e:
        print(f"Error generating MACCS fingerprint for {smi}: {e}, skipping")
        continue

if not predefined_fps:
    raise ValueError("No valid predefined MACCS fingerprints were generated.")

def MACCS_Tanimoto(mol):
    """Calculate the minimum Tanimoto similarity between the generated molecule's MACCS fingerprint
    and all valid predefined MACCS fingerprints.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Generated molecule

    Returns:
        float: Minimum Tanimoto similarity (or 0 if generation fails)
    """
    try:
        gen_fp = AllChem.GetMACCSKeysFingerprint(mol)
    except Exception as e:
        print(f"Error computing MACCS fingerprint for the molecule: {e}")
        return 0.0  

    if not predefined_fps:
        return 0.0

    try:
        similarities = BulkTanimotoSimilarity(gen_fp, predefined_fps)
        min_similarity = min(similarities)
        return min_similarity
    except Exception as e:
        print(f"Error calculating Tanimoto similarity: {e}")
        return 0.0
    
def to_vector_uint(arr):
    return tmap.VectorUint(arr.tolist())

predefined_map4 = []

for smi in predefined_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Invalid SMILES: {smi}, skipping")
        #continue
    try:
        map4_a = MAP4.calculate(mol)  # numpy array or list
        vec = to_vector_uint(np.array(map4_a))
        predefined_map4.append(vec)
    except Exception as e:
        print(f"Error generating map4 for {smi}: {e}, skipping")
        #continue

if not predefined_map4:
    raise ValueError("No valid predefined map4 descriptors were generated.")

def map4(mol):
    try:
        generated_map4 = MAP4.calculate(mol)
        generated_vec = to_vector_uint(np.array(generated_map4))
    except Exception as e:
        print(f"Error computing MAP4 for the molecule: {e}")
        return np.inf

    distances = []
    for predefined_vec in predefined_map4:
        dist = ENC.get_distance(generated_vec, predefined_vec)
        distances.append(dist)

    if distances:
        return min(distances)
    else:
        return np.inf
    


