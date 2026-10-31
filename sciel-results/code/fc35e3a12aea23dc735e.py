"""LJ38 cluster global minimum search using FCC truncated octahedron initialization + local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _lj_energy(coords):
    """Compute total Lennard-Jones energy for a set of coordinates."""
    n = len(coords)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            r2 = np.dot(diff, diff)
            r6 = r2 ** 3
            energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return energy


def _lj_force(coords):
    """Compute total Lennard-Jones force for a set of coordinates."""
    n = len(coords)
    forces = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            r2 = np.dot(diff, diff)
            r6 = r2 ** 3
            r12 = r6 * r6
            # Force magnitude factor: 48 * (r^-14 - 0.5*r^-8) * diff
            # F_ij = 48 * (r^-14 - 0.5 * r^-8) * (r_i - r_j)
            factor = 48.0 * (1.0 / (r12 * r2) - 0.5 / (r6 * r2))
            f = factor * diff
            forces[i] += f
            forces[j] -= f
    return forces


def _fcc_truncated_octahedron(n_atoms=38):
    """
    Generate coordinates for an FCC truncated octahedron cluster.
    
    The ideal FCC truncated octahedron for n=38 consists of:
    - 1 atom at center
    - 12 atoms at first shell (cubic positions)
    - 6 atoms at second shell (face centers)
    - 12 atoms at third shell (edge centers)
    - 7 atoms at fourth shell (corner positions)
    
    Actually, let's use a more systematic approach:
    FCC lattice with basis vectors, take atoms within a cutoff radius.
    """
    # FCC lattice: conventional cubic cell with lattice constant a
    # Basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # For LJ, the equilibrium spacing is at r = 2^(1/6) * sigma
    # In reduced units sigma=1, so nearest neighbor distance should be 2^(1/6) ≈ 1.12246
    
    # Let's generate FCC lattice points and select the 38 closest to origin
    # FCC primitive vectors:
    # a1 = (0, 0.5, 0.5) * a
    # a2 = (0.5, 0, 0.5) * a  
    # a3 = (0.5, 0.5, 0) * a
    # where a is the cubic lattice constant
    
    # Nearest neighbor distance in FCC = a / sqrt(2)
    # We want nearest neighbor = 2^(1/6), so a = 2^(1/6) * sqrt(2) = 2^(2/3)
    
    a = 2.0 ** (2.0 / 3.0)  # ≈ 1.5874
    
    # Generate FCC lattice points
    points = []
    # Search range: need enough to get 38 atoms
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # Four basis points per cubic cell
                for bx, by, bz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append((x, y, z))
    
    # Convert to numpy array
    points = np.array(points)
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take the 38 closest
    selected = points[sorted_idx[:n_atoms]]
    
    return selected


def construct():
    """Construct initial positions for LJ38 cluster, then relax locally."""
    # Start with FCC truncated octahedron configuration
    coords = _fcc_truncated_octahedron(N)
    
    # Ensure minimum distance constraint: atoms must be at least 0.5 apart
    # Check and adjust if needed
    min_dist = 0.5
    for _ in range(100):
        min_pair_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        if min_pair_dist >= min_dist:
            break
        # If too close, slightly expand the configuration
        coords *= 1.01
    
    # Local relaxation using L-BFGS
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def force_func(x):
        return -_lj_force(x.reshape(-1, 3)).flatten()
    
    x0 = coords.flatten()
    
    # Try to use L-BFGS-B with gradient
    result = minimize(
        energy_func,
        x0,
        jac=force_func,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or result.fun < _lj_energy(coords):
        coords = result.x.reshape(-1, 3)
    
    # Final check: ensure minimum distance constraint
    for _ in range(1000):
        min_pair_dist = float('inf')
        min_pair_idx = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair_idx = (i, j)
        if min_pair_dist >= min_dist:
            break
        # Nudge atoms apart
        i, j = min_pair_idx
        diff = coords[i] - coords[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            diff = diff / norm
        else:
            diff = np.array([1.0, 0.0, 0.0])
        coords[i] += diff * 0.01
        coords[j] -= diff * 0.01
    
    return coords
# EVOLVE-BLOCK-END
