"""
Phase 1: Data Ingestion & Chemoinformatics Setup for GeoBind

This module handles parsing and sanitization of receptor (PDB) and ligand (SDF/MOL2) files.
It extracts 3D atomic coordinates and provides cleaned molecular structures for downstream
physics calculations.

Dependencies:
    - Biopython (for PDB parsing)
    - RDKit (for ligand processing)
    - NumPy (for coordinate arrays)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AtomicCoordinates:
    """
    Container for atomic coordinates and metadata.
    
    Attributes:
        atom_names: List of atom names (e.g., ['CA', 'CB', 'N'])
        coordinates: numpy array of shape (N, 3) with x, y, z positions
        residues: List of residue identifiers (e.g., ['ALA-1', 'GLY-2'])
        elements: List of element symbols (e.g., ['C', 'N', 'O'])
    """
    
    def __init__(self, atom_names: List[str], coordinates: np.ndarray, 
                 residues: List[str] = None, elements: List[str] = None):
        """
        Initialize AtomicCoordinates.
        
        Args:
            atom_names: List of atom names
            coordinates: (N, 3) numpy array of atomic positions
            residues: List of residue identifiers (optional)
            elements: List of element symbols (optional)
        """
        self.atom_names = atom_names
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.residues = residues if residues is not None else ['UNK'] * len(atom_names)
        self.elements = elements if elements is not None else ['X'] * len(atom_names)
        
        if len(self.coordinates) != len(self.atom_names):
            raise ValueError("Mismatch between number of atoms and coordinates.")
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atom_names)
    
    def __repr__(self) -> str:
        return f"AtomicCoordinates(n_atoms={len(self)}, shape={self.coordinates.shape})"


class ReceptorParser:
    """
    Parse and sanitize PDB files for receptor structures.
    """
    
    @staticmethod
    def parse_pdb(pdb_file_path: str, chain_id: Optional[str] = None) -> Tuple[AtomicCoordinates, Dict]:
        """
        Parse a PDB file and extract atomic coordinates.
        
        Args:
            pdb_file_path: Path to the PDB file
            chain_id: Specific chain to extract (e.g., 'A'). If None, uses the first chain.
        
        Returns:
            Tuple of (AtomicCoordinates object, metadata dict)
        
        Raises:
            FileNotFoundError: If PDB file does not exist
            ValueError: If no valid atoms found in the structure
        """
        pdb_file_path = Path(pdb_file_path)
        if not pdb_file_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
        
        logger.info(f"Parsing PDB file: {pdb_file_path}")
        
        # Initialize PDB parser (permissive mode for flexibility)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_file_path))
        
        # Extract protein chain
        atom_names = []
        coordinates = []
        residues = []
        elements = []
        
        # Determine which chain to use
        model = structure[0]
        chains = list(model.get_chains())
        
        if not chains:
            raise ValueError("No chains found in PDB structure.")
        
        if chain_id is None:
            selected_chain = chains[0]
            chain_id = selected_chain.id
            logger.info(f"No chain specified; using first chain: {chain_id}")
        else:
            selected_chain = model[chain_id]
        
        # Extract coordinates from atoms, excluding water (HOH, WAT)
        for residue in selected_chain.get_residues():
            res_name = residue.get_resname().strip()
            
            # Skip water molecules
            if res_name in ['HOH', 'WAT', 'H2O']:
                continue
            
            res_id = residue.get_id()[1]
            
            for atom in residue.get_atoms():
                atom_name = atom.get_name().strip()
                coord = atom.get_vector().get_array()
                # Get element symbol from atom (handle both old and new BioPython versions)
                try:
                    element = atom.element.strip() if atom.element else 'X'
                except (AttributeError, TypeError):
                    element = 'X'  # Default if element info unavailable
                
                atom_names.append(atom_name)
                coordinates.append(coord)
                residues.append(f"{res_name}-{res_id}")
                elements.append(element)
        
        if not coordinates:
            raise ValueError("No valid atoms found after filtering.")
        
        coordinates = np.array(coordinates, dtype=np.float32)
        
        metadata = {
            'pdb_file': str(pdb_file_path),
            'chain_id': chain_id,
            'n_atoms': len(atom_names),
            'center_of_mass': np.mean(coordinates, axis=0),
            'bounding_box_min': np.min(coordinates, axis=0),
            'bounding_box_max': np.max(coordinates, axis=0),
        }
        
        logger.info(f"Extracted {len(atom_names)} atoms from chain {chain_id}")
        
        return AtomicCoordinates(atom_names, coordinates, residues, elements), metadata
    
    @staticmethod
    def sanitize_receptor(atomic_coords: AtomicCoordinates) -> AtomicCoordinates:
        """
        Sanitize receptor coordinates (remove ions, non-standard residues if needed).
        
        Args:
            atomic_coords: AtomicCoordinates object
        
        Returns:
            Cleaned AtomicCoordinates object
        """
        # For now, keep all heavy atoms (basic sanitization already done in parse_pdb)
        logger.info(f"Sanitization complete: {len(atomic_coords)} atoms retained")
        return atomic_coords


class LigandParser:
    """
    Parse and sanitize ligand files (SDF, MOL2).
    """
    
    @staticmethod
    def parse_sdf(sdf_file_path: str, sanitize: bool = True) -> Tuple[AtomicCoordinates, Dict]:
        """
        Parse an SDF (Structure Data Format) file and extract ligand coordinates.
        
        Args:
            sdf_file_path: Path to the SDF file
            sanitize: Whether to sanitize the molecule (default: True)
        
        Returns:
            Tuple of (AtomicCoordinates object, metadata dict)
        
        Raises:
            FileNotFoundError: If SDF file does not exist
            ValueError: If molecule cannot be parsed
        """
        sdf_file_path = Path(sdf_file_path)
        if not sdf_file_path.exists():
            raise FileNotFoundError(f"SDF file not found: {sdf_file_path}")
        
        logger.info(f"Parsing SDF file: {sdf_file_path}")
        
        # Read molecule from SDF
        suppl = Chem.SDMolSupplier(str(sdf_file_path), removeHs=False, sanitize=sanitize)
        
        if not suppl or len(suppl) == 0:
            raise ValueError("No valid molecules found in SDF file.")
        
        mol = suppl[0]
        if mol is None:
            raise ValueError("Failed to parse molecule from SDF file.")
        
        # Ensure 3D coordinates are present
        if not mol.GetNumConformers():
            logger.warning("No 3D coordinates found in SDF; generating them.")
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        conf = mol.GetConformer(0)
        
        # Extract coordinates
        atom_names = []
        coordinates = []
        elements = []
        
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            symbol = atom.GetSymbol()
            
            atom_names.append(f"{symbol}{atom_idx}")
            coordinates.append([pos.x, pos.y, pos.z])
            elements.append(symbol)
        
        coordinates = np.array(coordinates, dtype=np.float32)
        
        # Get molecule properties
        mol_weight = Descriptors.MolWt(mol)
        mol_formula = rdMolDescriptors.CalcMolFormula(mol)
        
        metadata = {
            'sdf_file': str(sdf_file_path),
            'n_atoms': len(atom_names),
            'molecular_weight': mol_weight,
            'molecular_formula': mol_formula,
            'center_of_mass': np.mean(coordinates, axis=0),
            'bounding_box_min': np.min(coordinates, axis=0),
            'bounding_box_max': np.max(coordinates, axis=0),
        }
        
        logger.info(f"Extracted {len(atom_names)} atoms from ligand (MW: {mol_weight:.2f})")
        
        return AtomicCoordinates(atom_names, coordinates, elements=elements), metadata
    
    @staticmethod
    def parse_mol2(mol2_file_path: str) -> Tuple[AtomicCoordinates, Dict]:
        """
        Parse a MOL2 (Tripos MOL2) file and extract ligand coordinates.
        
        Args:
            mol2_file_path: Path to the MOL2 file
        
        Returns:
            Tuple of (AtomicCoordinates object, metadata dict)
        
        Raises:
            FileNotFoundError: If MOL2 file does not exist
            ValueError: If file format is invalid
        """
        mol2_file_path = Path(mol2_file_path)
        if not mol2_file_path.exists():
            raise FileNotFoundError(f"MOL2 file not found: {mol2_file_path}")
        
        logger.info(f"Parsing MOL2 file: {mol2_file_path}")
        
        # Read MOL2 file
        mol = Chem.MolFromMol2File(str(mol2_file_path), removeHs=False)
        
        if mol is None:
            raise ValueError("Failed to parse molecule from MOL2 file.")
        
        # Ensure 3D coordinates
        if not mol.GetNumConformers():
            logger.warning("No 3D coordinates found in MOL2; generating them.")
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        conf = mol.GetConformer(0)
        
        # Extract coordinates
        atom_names = []
        coordinates = []
        elements = []
        
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            symbol = atom.GetSymbol()
            
            atom_names.append(f"{symbol}{atom_idx}")
            coordinates.append([pos.x, pos.y, pos.z])
            elements.append(symbol)
        
        coordinates = np.array(coordinates, dtype=np.float32)
        
        mol_weight = Descriptors.MolWt(mol)
        mol_formula = rdMolDescriptors.CalcMolFormula(mol)
        
        metadata = {
            'mol2_file': str(mol2_file_path),
            'n_atoms': len(atom_names),
            'molecular_weight': mol_weight,
            'molecular_formula': mol_formula,
            'center_of_mass': np.mean(coordinates, axis=0),
            'bounding_box_min': np.min(coordinates, axis=0),
            'bounding_box_max': np.max(coordinates, axis=0),
        }
        
        logger.info(f"Extracted {len(atom_names)} atoms from ligand (MW: {mol_weight:.2f})")
        
        return AtomicCoordinates(atom_names, coordinates, elements=elements), metadata


class DataIngestionPipeline:
    """
    Unified interface for Phase 1 data ingestion.
    """
    
    def __init__(self):
        self.receptor = None
        self.ligand = None
        self.receptor_metadata = None
        self.ligand_metadata = None
    
    def load_receptor(self, pdb_file_path: str, chain_id: Optional[str] = None) -> Dict:
        """
        Load and parse a receptor PDB file.
        
        Args:
            pdb_file_path: Path to PDB file
            chain_id: Optional specific chain ID
        
        Returns:
            Metadata dictionary
        """
        self.receptor, self.receptor_metadata = ReceptorParser.parse_pdb(pdb_file_path, chain_id)
        return self.receptor_metadata
    
    def load_ligand(self, ligand_file_path: str) -> Dict:
        """
        Load and parse a ligand file (SDF or MOL2).
        
        Args:
            ligand_file_path: Path to SDF or MOL2 file
        
        Returns:
            Metadata dictionary
        
        Raises:
            ValueError: If file format is not recognized
        """
        file_ext = Path(ligand_file_path).suffix.lower()
        
        if file_ext == '.sdf':
            self.ligand, self.ligand_metadata = LigandParser.parse_sdf(ligand_file_path)
        elif file_ext == '.mol2':
            self.ligand, self.ligand_metadata = LigandParser.parse_mol2(ligand_file_path)
        else:
            raise ValueError(f"Unsupported ligand file format: {file_ext}. Use .sdf or .mol2")
        
        return self.ligand_metadata
    
    def get_summary(self) -> Dict:
        """
        Get a summary of loaded structures.
        
        Returns:
            Dictionary with structure summaries
        """
        return {
            'receptor': {
                'loaded': self.receptor is not None,
                'n_atoms': len(self.receptor) if self.receptor else 0,
                'metadata': self.receptor_metadata
            },
            'ligand': {
                'loaded': self.ligand is not None,
                'n_atoms': len(self.ligand) if self.ligand else 0,
                'metadata': self.ligand_metadata
            }
        }


if __name__ == '__main__':
    logger.info("Phase 1 Data Ingestion Module - Ready for import")
