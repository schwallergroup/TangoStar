from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import torch
import numpy as np
from desp.inference.utils import smiles_to_fp

class TangoValue():
    def __init__(self, tango_weight=25.0, tanimoto_weight=1.0, mcs_weight=0.0, root_smiles=None):
        self.weight = tango_weight
        self.tanimoto_weight = tanimoto_weight
        self.mcs_weight = mcs_weight
        if root_smiles is not None:
            root_mol = Chem.MolFromSmiles(root_smiles)
            self.root_fp = GetMorganFingerprintAsBitVect(root_mol, radius=3, nBits=2048)
        else:
            self.root_fp = None  # Handle the case where root_fp is not provided

    def tanimoto(self, query_fp, precursor_fp):
        """
        Computes the weighted Tanimoto similarity between two fingerprints.
        
        Args:
            query_fp: Fingerprint of the query molecule.
            precursor_fp: Fingerprint of the precursor molecule.
        
        Returns:
            float: Weighted Tanimoto similarity score.
        """
        return self.tanimoto_weight * TanimotoSimilarity(query_fp, precursor_fp)
    
    def mcs(self, query_mol, precursor_mol):
        """
        Computes the weighted Maximum Common Substructure (MCS) score between two molecules.
        
        Args:
            query_mol: RDKit Mol object of the query molecule.
            precursor_mol: RDKit Mol object of the precursor molecule.
        
        Returns:
            float: Weighted MCS score.
        """
        # Find the Maximum Common Substructure (MCS) between the query and precursor molecules
        mcs_result = rdFMCS.FindMCS(
            mols=[query_mol, precursor_mol],
            matchChiralTag=True,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringCompare=rdFMCS.RingCompare.StrictRingFusion,
            completeRingsOnly=True
        )
        # Compute the weighted MCS score based on the fraction of matching atoms
        return self.mcs_weight * max(0, (mcs_result.numAtoms / precursor_mol.GetNumAtoms()))

    def dissimilar_pred(self,query):
        """
        Predicts the dissimilarity score between the query molecule and a root molecule.
        
        Args:
            query (str): SMILES string of the query molecule.
        
        Returns:
            float: Dissimilarity score.
        
        Note:
            This method uses 'self.root_fp', which should be defined elsewhere in the class.
        """
        # Convert the query SMILES string to an RDKit Mol object
        query_mol = Chem.MolFromSmiles(query)
        # Generate the fingerprint for the query molecule
        query_fp = GetMorganFingerprintAsBitVect(query_mol, radius=3, nBits=2048)
        # Compute the dissimilarity score using the Tanimoto similarity to the root fingerprint
        return self.weight * (
            self.tanimoto(query_fp, self.root_fp)
        )
    
    def predict(self, sm, query):
        """
        Predicts the synthetic cost (distance) between two molecules.
        
        Args:
            sm (str): SMILES string of the precursor molecule.
            query (str): SMILES string of the target molecule.
        
        Returns:
            float: Synthetic distance between the precursor and target molecules.
        """
        # Convert SMILES strings to RDKit Mol objects
        sm_mol = Chem.MolFromSmiles(sm)
        query_mol = Chem.MolFromSmiles(query)
        
        # Generate Morgan fingerprints for both molecules
        sm_fp = GetMorganFingerprintAsBitVect(sm_mol, radius=3, nBits=2048)
        query_fp = GetMorganFingerprintAsBitVect(query_mol, radius=3, nBits=2048)
        
        # Compute the weighted Tanimoto similarity between the query and precursor fingerprints
        return self.weight * (
            self.tanimoto(query_fp, sm_fp)
        )

    
    def predict_batch(self, starts, targets):
        """
        Computes the tango distances between lists of start and target molecules in batch.
        
        Args:
            starts (list of str): List of SMILES strings for start (precursor) molecules.
            targets (list of str): List of SMILES strings for target molecules.
        
        Returns:
            torch.Tensor: 2D tensor of synthetic distances (dissimilarities).
        """

        starts = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=3, nBits=2048) for smi in starts]
        targets = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=3, nBits=2048) for smi in targets]
        num_starts = len(starts)
        num_targets = len(targets)
        sim_matrix = np.zeros((num_starts, num_targets))
        
        for i, start_fp in enumerate(starts):
            # Compute similarities to all target fingerprints
            sims = BulkTanimotoSimilarity(start_fp, targets)
            transformed_sims = self.weight * (1 - np.array(sims))
            sim_matrix[i, :] = transformed_sims
        
        # Convert the similarity matrix to a Torch tensor if needed
        sim_tensor = torch.from_numpy(sim_matrix)
        return sim_tensor
