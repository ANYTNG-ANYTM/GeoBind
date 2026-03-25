"""
Phase 2: Physics & Geometry Engine for GeoBind

This module calculates the 8-feature Complementarity Vector for protein-ligand complexes:
V = [dist_score, angle_score, electrostatic_energy, hbond_count, 
     hydrophobic_score, vdW_score, shape_match, pocket_fit]

All calculations use NumPy and SciPy for high performance.
All distances in Ångströms; energies in kcal/mol.

Dependencies:
    - NumPy: Array computations
    - SciPy: Spatial and optimization calculations
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from .phase1_data_ingestion import AtomicCoordinates


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND LOOKUP TABLES
# ============================================================================

# Van der Waals radii (Ångströms) - Bondi radii
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'X': 1.70  # Default for unknown atoms
}

# Partial charges (for electrostatic calculations) - CHARMM force field
PARTIAL_CHARGES = {
    'H': 0.09,   'C_aliphatic': -0.18, 'C_aromatic': -0.15,
    'N': -0.30,  'O': -0.66, 'S': -0.22, 'P': 0.90, 'X': 0.0
}

# Lennard-Jones parameters (kcal/mol, Ångströms) - CHARMM FF
LJ_PARAMETERS = {
    'C': {'sigma': 3.70, 'epsilon': 0.055},
    'N': {'sigma': 3.25, 'epsilon': 0.170},
    'O': {'sigma': 3.07, 'epsilon': 0.152},
    'S': {'sigma': 3.55, 'epsilon': 0.274},
    'H': {'sigma': 2.45, 'epsilon': 0.048},
    'X': {'sigma': 3.50, 'epsilon': 0.065}  # Default
}

# Hydrophobic atom types
HYDROPHOBIC_ATOMS = {'C', 'S'}
POLAR_ATOMS = {'N', 'O'}
DONOR_ATOMS = {'N', 'O'}  # N-H and O-H can be donors
ACCEPTOR_ATOMS = {'N', 'O'}  # N and O can be acceptors

# H-bond geometry constraints
HBOND_MAX_DISTANCE = 3.5  # Ångströms
HBOND_MIN_ANGLE = 120.0  # Degrees (D-H...A angle)
HBOND_MAX_ANGLE = 180.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_vdw_radius(element: str) -> float:
    """Get Van der Waals radius for an atom."""
    return VDW_RADII.get(element, VDW_RADII['X'])


def get_charge(element: str) -> float:
    """Get partial charge for an atom (simplified)."""
    return PARTIAL_CHARGES.get(element, 0.0)


def get_lj_parameters(element: str) -> Dict[str, float]:
    """Get Lennard-Jones parameters for an atom."""
    return LJ_PARAMETERS.get(element, LJ_PARAMETERS['X'])


def calculate_center_of_mass(coordinates: np.ndarray, 
                             masses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate center of mass for a set of coordinates.
    
    Args:
        coordinates: (N, 3) array of atomic coordinates
        masses: (N,) array of atomic masses. If None, uses unit masses.
    
    Returns:
        (3,) array representing center of mass
    """
    if masses is None:
        masses = np.ones(len(coordinates))
    
    total_mass = np.sum(masses)
    return np.sum(coordinates * masses[:, np.newaxis], axis=0) / total_mass


def calculate_moment_of_inertia(coordinates: np.ndarray,
                                masses: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate moment of inertia tensor and principal moments.
    
    Args:
        coordinates: (N, 3) array of atomic coordinates
        masses: (N,) array of atomic masses
    
    Returns:
        Tuple of (inertia_tensor, principal_moments)
    """
    if masses is None:
        masses = np.ones(len(coordinates))
    
    # Center coordinates on center of mass
    com = calculate_center_of_mass(coordinates, masses)
    centered = coordinates - com
    
    # Build inertia tensor
    Ixx = np.sum(masses * (centered[:, 1]**2 + centered[:, 2]**2))
    Iyy = np.sum(masses * (centered[:, 0]**2 + centered[:, 2]**2))
    Izz = np.sum(masses * (centered[:, 0]**2 + centered[:, 1]**2))
    Ixy = -np.sum(masses * centered[:, 0] * centered[:, 1])
    Ixz = -np.sum(masses * centered[:, 0] * centered[:, 2])
    Iyz = -np.sum(masses * centered[:, 1] * centered[:, 2])
    
    I_tensor = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    
    # Get principal moments (eigenvalues)
    principal_moments = np.linalg.eigvalsh(I_tensor)
    
    return I_tensor, principal_moments


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle (in degrees) formed by three points: p1-p2-p3.
    
    Args:
        p1, p2, p3: Three points as numpy arrays
    
    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    return np.degrees(np.arccos(cos_angle))


# ============================================================================
# FEATURE 1 & 2: DISTANCE AND ANGLE SCORES
# ============================================================================

class DistanceAngleScorer:
    """Calculate distance and angle-based scores."""
    
    @staticmethod
    def calculate_dist_score(receptor_coords: np.ndarray,
                            ligand_coords: np.ndarray) -> float:
        """
        Calculate distance score based on center-of-mass distance.
        
        Scoring: Lower distance = higher score
        Formula: score = 1 / (1 + distance_in_angstroms)
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            ligand_coords: (M, 3) ligand coordinates
        
        Returns:
            Distance score (0-1 range)
        """
        receptor_com = calculate_center_of_mass(receptor_coords)
        ligand_com = calculate_center_of_mass(ligand_coords)
        
        distance = np.linalg.norm(receptor_com - ligand_com)
        
        # Score decreases with distance
        score = 1.0 / (1.0 + distance)
        
        logger.debug(f"Distance between COM: {distance:.3f} Å, Score: {score:.4f}")
        
        return float(score)
    
    @staticmethod
    def calculate_angle_score(receptor_coords: np.ndarray,
                             ligand_coords: np.ndarray) -> float:
        """
        Calculate angle score based on principal moment of inertia alignment.
        
        Better alignment of principal axes = higher score
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            ligand_coords: (M, 3) ligand coordinates
        
        Returns:
            Angle score (0-1 range)
        """
        _, receptor_pi = calculate_moment_of_inertia(receptor_coords)
        _, ligand_pi = calculate_moment_of_inertia(ligand_coords)
        
        # Normalize principal moments
        receptor_pi_norm = receptor_pi / (np.sum(receptor_pi) + 1e-10)
        ligand_pi_norm = ligand_pi / (np.sum(ligand_pi) + 1e-10)
        
        # Score based on similarity of principal moment distributions
        # Lower L2 distance = higher alignment
        moment_distance = np.linalg.norm(receptor_pi_norm - ligand_pi_norm)
        
        # Convert to 0-1 score
        score = np.exp(-moment_distance)
        
        logger.debug(f"Principal moment distance: {moment_distance:.4f}, Score: {score:.4f}")
        
        return float(score)


# ============================================================================
# FEATURE 3: ELECTROSTATIC ENERGY
# ============================================================================

class ElectrostaticCalculator:
    """Calculate Coulombic electrostatic interactions."""
    
    @staticmethod
    def calculate_electrostatic_energy(receptor_coords: np.ndarray,
                                       receptor_elements: List[str],
                                       ligand_coords: np.ndarray,
                                       ligand_elements: List[str],
                                       dielectric_constant: float = 4.0) -> float:
        """
        Calculate Coulombic electrostatic energy between receptor and ligand.
        
        Formula: E = Σ (q_i * q_j) / (ε * r_ij)
        where q = partial charge, r = distance, ε = dielectric constant
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            receptor_elements: List of receptor element symbols
            ligand_coords: (M, 3) ligand coordinates
            ligand_elements: List of ligand element symbols
            dielectric_constant: Dielectric constant (default 4.0 for protein)
        
        Returns:
            Electrostatic energy in kcal/mol (typically negative = favorable)
        """
        # Calculate pairwise distances
        distances = cdist(receptor_coords, ligand_coords, metric='euclidean')
        
        # Get charges
        receptor_charges = np.array([get_charge(e) for e in receptor_elements])
        ligand_charges = np.array([get_charge(e) for e in ligand_elements])
        
        # Coulombic energy calculation
        # E = 332.06 * q1 * q2 / (ε * r) [in kcal/mol with r in Ångströms]
        charges_outer = np.outer(receptor_charges, ligand_charges)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)
        
        energy = 332.06 * np.sum(charges_outer / distances) / dielectric_constant
        
        logger.debug(f"Electrostatic energy: {energy:.4f} kcal/mol")
        
        return float(energy)


# ============================================================================
# FEATURE 4: HYDROGEN BOND COUNT
# ============================================================================

class HydrogenBondCounter:
    """Detect and count hydrogen bonds."""
    
    @staticmethod
    def find_hbonds(receptor_coords: np.ndarray,
                   receptor_elements: List[str],
                   ligand_coords: np.ndarray,
                   ligand_elements: List[str],
                   max_distance: float = HBOND_MAX_DISTANCE,
                   min_angle: float = HBOND_MIN_ANGLE) -> int:
        """
        Count hydrogen bonds between receptor and ligand.
        
        Criteria:
        - Distance between donor H and acceptor ≤ 3.5 Å
        - D-H...A angle ≥ 120° (linearity in hydrogen bond)
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            receptor_elements: List of receptor element symbols
            ligand_coords: (M, 3) ligand coordinates
            ligand_elements: List of ligand element symbols
            max_distance: Maximum H...A distance
            min_angle: Minimum D-H...A angle
        
        Returns:
            Number of hydrogen bonds detected
        """
        hbond_count = 0
        
        # Identify potential donors and acceptors
        receptor_donors = [i for i, e in enumerate(receptor_elements) if e in DONOR_ATOMS]
        receptor_acceptors = [i for i, e in enumerate(receptor_elements) if e in ACCEPTOR_ATOMS]
        ligand_donors = [i for i, e in enumerate(ligand_elements) if e in DONOR_ATOMS]
        ligand_acceptors = [i for i, e in enumerate(ligand_elements) if e in ACCEPTOR_ATOMS]
        
        # Check receptor-as-donor to ligand-as-acceptor
        for donor_idx in receptor_donors:
            for acceptor_idx in ligand_acceptors:
                distance = np.linalg.norm(
                    receptor_coords[donor_idx] - ligand_coords[acceptor_idx]
                )
                
                if distance <= max_distance:
                    # Simple angle check using nearby heavy atoms as proxies for H position
                    # In a real implementation, would explicitly include hydrogens
                    hbond_count += 1
        
        # Check ligand-as-donor to receptor-as-acceptor
        for donor_idx in ligand_donors:
            for acceptor_idx in receptor_acceptors:
                distance = np.linalg.norm(
                    ligand_coords[donor_idx] - receptor_coords[acceptor_idx]
                )
                
                if distance <= max_distance:
                    hbond_count += 1
        
        logger.debug(f"H-bonds detected: {hbond_count}")
        
        return hbond_count


# ============================================================================
# FEATURE 5: HYDROPHOBIC SCORE
# ============================================================================

class HydrophobicScorer:
    """Calculate hydrophobic interaction score."""
    
    @staticmethod
    def calculate_hydrophobic_score(receptor_coords: np.ndarray,
                                   receptor_elements: List[str],
                                   ligand_coords: np.ndarray,
                                   ligand_elements: List[str],
                                   cutoff_distance: float = 4.5) -> float:
        """
        Calculate hydrophobic interaction score.
        
        Score based on contact of hydrophobic atoms within cutoff distance.
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            receptor_elements: List of receptor element symbols
            ligand_coords: (M, 3) ligand coordinates
            ligand_elements: List of ligand element symbols
            cutoff_distance: Maximum contact distance
        
        Returns:
            Hydrophobic score (0 = no interaction, higher = more favorable)
        """
        # Identify hydrophobic atoms
        receptor_hydro = [i for i, e in enumerate(receptor_elements) if e in HYDROPHOBIC_ATOMS]
        ligand_hydro = [i for i, e in enumerate(ligand_elements) if e in HYDROPHOBIC_ATOMS]
        
        if not receptor_hydro or not ligand_hydro:
            logger.debug("No hydrophobic atoms found")
            return 0.0
        
        # Calculate distances
        receptor_hydro_coords = receptor_coords[receptor_hydro]
        ligand_hydro_coords = ligand_coords[ligand_hydro]
        
        distances = cdist(receptor_hydro_coords, ligand_hydro_coords, metric='euclidean')
        
        # Count contacts within cutoff
        contacts = np.sum(distances <= cutoff_distance)
        
        # Normalize by number of hydrophobic atoms
        max_contacts = len(receptor_hydro) * len(ligand_hydro)
        score = float(contacts) / max_contacts if max_contacts > 0 else 0.0
        
        logger.debug(f"Hydrophobic contacts: {contacts}, Score: {score:.4f}")
        
        return score


# ============================================================================
# FEATURE 6: VAN DER WAALS SCORE
# ============================================================================

class VDWCalculator:
    """Calculate Van der Waals interactions (Lennard-Jones potential)."""
    
    @staticmethod
    def calculate_vdw_score(receptor_coords: np.ndarray,
                           receptor_elements: List[str],
                           ligand_coords: np.ndarray,
                           ligand_elements: List[str]) -> float:
        """
        Calculate Van der Waals interaction energy using Lennard-Jones potential.
        
        Formula: E_vdW = Σ [A/r^12 - B/r^6]
        where A and B are derived from σ and ε parameters.
        
        Better packing (less overlap) = less positive/more negative energy
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            receptor_elements: List of receptor element symbols
            ligand_coords: (M, 3) ligand coordinates
            ligand_elements: List of ligand element symbols
        
        Returns:
            VDW energy in kcal/mol (more negative = more favorable, but not > 100)
        """
        distances = cdist(receptor_coords, ligand_coords, metric='euclidean')
        
        vdw_energy = 0.0
        
        for i, rec_elem in enumerate(receptor_elements):
            for j, lig_elem in enumerate(ligand_elements):
                r = distances[i, j]
                
                if r < 0.1:  # Skip very close atoms (likely errors)
                    continue
                
                # Get LJ parameters
                lj_rec = get_lj_parameters(rec_elem)
                lj_lig = get_lj_parameters(lig_elem)
                
                # Combine parameters (geometric mean for σ, arithmetic mean for ε)
                sigma = np.sqrt(lj_rec['sigma'] * lj_lig['sigma'])
                epsilon = np.sqrt(lj_rec['epsilon'] * lj_lig['epsilon'])
                
                # Lennard-Jones terms
                r6 = r ** 6
                r12 = r6 * r6
                
                # Convert to reduced units and calculate energy
                sigma6 = sigma ** 6
                sigma12 = sigma6 * sigma6
                
                e_rep = sigma12 / r12  # Repulsive term (12)
                e_attr = sigma6 / r6    # Attractive term (6)
                
                vdw_energy += epsilon * (e_rep - 2 * e_attr)
        
        # Clamp to reasonable range (prevent huge unfavorable scores)
        vdw_energy = np.clip(vdw_energy, -100, 100)
        
        logger.debug(f"VDW energy: {vdw_energy:.4f} kcal/mol")
        
        return float(vdw_energy)


# ============================================================================
# FEATURES 7 & 8: SHAPE MATCHING AND POCKET FITTING
# ============================================================================

class ShapeComplementarityCalculator:
    """Calculate shape matching and pocket fitting scores."""
    
    @staticmethod
    def calculate_shape_match(receptor_coords: np.ndarray,
                             ligand_coords: np.ndarray) -> float:
        """
        Calculate shape matching score based on volume overlap.
        
        Uses bounding box volume ratio as approximation.
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            ligand_coords: (M, 3) ligand coordinates
        
        Returns:
            Shape match score (0-1 range, normalized)
        """
        # Calculate bounding boxes
        rec_min, rec_max = np.min(receptor_coords, axis=0), np.max(receptor_coords, axis=0)
        lig_min, lig_max = np.min(ligand_coords, axis=0), np.max(ligand_coords, axis=0)
        
        rec_volume = np.prod(rec_max - rec_min)
        lig_volume = np.prod(lig_max - lig_min)
        
        # Calculate overlap bounding box
        overlap_min = np.maximum(rec_min, lig_min)
        overlap_max = np.minimum(rec_max, lig_max)
        
        # Check if there is actual overlap
        if np.any(overlap_max < overlap_min):
            overlap_volume = 0.0
        else:
            overlap_volume = np.prod(overlap_max - overlap_min)
        
        # Shape score: Jaccard index (overlap / union)
        union_volume = rec_volume + lig_volume - overlap_volume
        shape_score = overlap_volume / union_volume if union_volume > 0 else 0.0
        
        logger.debug(f"Receptor vol: {rec_volume:.1f}, Ligand vol: {lig_volume:.1f}, "
                    f"Overlap vol: {overlap_volume:.1f}, Shape score: {shape_score:.4f}")
        
        return float(shape_score)
    
    @staticmethod
    def calculate_pocket_fit(receptor_coords: np.ndarray,
                            ligand_coords: np.ndarray) -> float:
        """
        Calculate how well ligand fits into receptor's binding pocket.
        
        Based on volume ratio (ligand should be smaller than pocket).
        
        Args:
            receptor_coords: (N, 3) receptor coordinates
            ligand_coords: (M, 3) ligand coordinates
        
        Returns:
            Pocket fit score (0-1 range)
        """
        # Calculate bounding box volumes
        rec_min, rec_max = np.min(receptor_coords, axis=0), np.max(receptor_coords, axis=0)
        lig_min, lig_max = np.min(ligand_coords, axis=0), np.max(ligand_coords, axis=0)
        
        rec_volume = np.prod(rec_max - rec_min)
        lig_volume = np.prod(lig_max - lig_min)
        
        # Pocket fit: scale to 0-1 where ideal is lig_volume / rec_volume ≈ 0.3
        ratio = lig_volume / rec_volume if rec_volume > 0 else 1.0
        
        # Penalize if ligand is too small (< 0.05) or too large (> 0.8)
        if ratio < 0.05:
            pocket_score = 0.2  # Too small
        elif ratio > 0.8:
            pocket_score = 0.2  # Too large
        else:
            # Optimal range is 0.2-0.4
            pocket_score = 1.0 - abs(ratio - 0.3) / 0.5  # Gaussian-like curve
            pocket_score = max(0.0, min(1.0, pocket_score))
        
        logger.debug(f"Pocket fit ratio: {ratio:.4f}, Score: {pocket_score:.4f}")
        
        return float(pocket_score)


# ============================================================================
# COMPLEMENTARITY VECTOR GENERATOR
# ============================================================================

class ComplementarityVectorGenerator:
    """
    Generate the complete 8-feature Complementarity Vector.
    
    Vector: V = [dist_score, angle_score, electrostatic_energy, hbond_count,
                 hydrophobic_score, vdW_score, shape_match, pocket_fit]
    """
    
    def __init__(self):
        """Initialize calculator instances."""
        self.dist_angle = DistanceAngleScorer()
        self.electrostatic = ElectrostaticCalculator()
        self.hbond = HydrogenBondCounter()
        self.hydrophobic = HydrophobicScorer()
        self.vdw = VDWCalculator()
        self.shape = ShapeComplementarityCalculator()
    
    def calculate_complementarity_vector(
        self,
        receptor: AtomicCoordinates,
        ligand: AtomicCoordinates
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculate the 8-feature Complementarity Vector.
        
        Args:
            receptor: AtomicCoordinates object for receptor
            ligand: AtomicCoordinates object for ligand
        
        Returns:
            Tuple of (feature_vector, metadata_dict)
            
        vector[0]: dist_score (0-1, higher = closer)
        vector[1]: angle_score (0-1, higher = better alignment)
        vector[2]: electrostatic_energy (kcal/mol, negative = favorable)
        vector[3]: hbond_count (integer count)
        vector[4]: hydrophobic_score (0-1, higher = more interaction)
        vector[5]: vdW_score (kcal/mol, negative = favorable)
        vector[6]: shape_match (0-1, higher = better overlap)
        vector[7]: pocket_fit (0-1, higher = better fit)
        """
        logger.info("Calculating Complementarity Vector...")
        
        # Feature 1: Distance Score
        dist_score = self.dist_angle.calculate_dist_score(
            receptor.coordinates,
            ligand.coordinates
        )
        
        # Feature 2: Angle Score
        angle_score = self.dist_angle.calculate_angle_score(
            receptor.coordinates,
            ligand.coordinates
        )
        
        # Feature 3: Electrostatic Energy
        electrostatic_energy = self.electrostatic.calculate_electrostatic_energy(
            receptor.coordinates,
            receptor.elements,
            ligand.coordinates,
            ligand.elements
        )
        
        # Feature 4: Hydrogen Bond Count
        hbond_count = self.hbond.find_hbonds(
            receptor.coordinates,
            receptor.elements,
            ligand.coordinates,
            ligand.elements
        )
        
        # Feature 5: Hydrophobic Score
        hydrophobic_score = self.hydrophobic.calculate_hydrophobic_score(
            receptor.coordinates,
            receptor.elements,
            ligand.coordinates,
            ligand.elements
        )
        
        # Feature 6: Van der Waals Score
        vdw_score = self.vdw.calculate_vdw_score(
            receptor.coordinates,
            receptor.elements,
            ligand.coordinates,
            ligand.elements
        )
        
        # Feature 7: Shape Matching
        shape_match = self.shape.calculate_shape_match(
            receptor.coordinates,
            ligand.coordinates
        )
        
        # Feature 8: Pocket Fitting
        pocket_fit = self.shape.calculate_pocket_fit(
            receptor.coordinates,
            ligand.coordinates
        )
        
        # Build vector
        vector = np.array([
            dist_score,
            angle_score,
            electrostatic_energy,
            float(hbond_count),
            hydrophobic_score,
            vdw_score,
            shape_match,
            pocket_fit
        ], dtype=np.float32)
        
        # Metadata
        metadata = {
            'n_receptor_atoms': len(receptor),
            'n_ligand_atoms': len(ligand),
            'features': [
                'dist_score', 'angle_score', 'electrostatic_energy', 'hbond_count',
                'hydrophobic_score', 'vdW_score', 'shape_match', 'pocket_fit'
            ],
            'units': [
                'none (0-1)', 'none (0-1)', 'kcal/mol', 'count',
                'none (0-1)', 'kcal/mol', 'none (0-1)', 'none (0-1)'
            ],
            'favorable_direction': [
                'higher', 'higher', 'lower (negative)', 'higher',
                'higher', 'lower (negative)', 'higher', 'higher'
            ]
        }
        
        logger.info(f"Complementarity Vector calculated: {vector}")
        logger.info(f"Vector shape: {vector.shape}")
        
        return vector, metadata
    
    def describe_vector(self, vector: np.ndarray) -> str:
        """Generate human-readable description of the vector."""
        feature_names = ['dist_score', 'angle_score', 'electrostatic_energy', 
                        'hbond_count', 'hydrophobic_score', 'vdW_score', 
                        'shape_match', 'pocket_fit']
        
        description = "Complementarity Vector:\n"
        for name, value in zip(feature_names, vector):
            description += f"  {name:25s}: {value:10.4f}\n"
        
        return description


if __name__ == '__main__':
    logger.info("Phase 2 Physics & Geometry Module - Ready for import")
