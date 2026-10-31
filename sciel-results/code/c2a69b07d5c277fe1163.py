import numpy as np
import time
from scipy.optimize import minimize

N = 38
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def lj_energy(coords):
    """Compute total Lennard-Jones energy."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 * r6
    energy_matrix = 4.0 * (r12 - r6)
    # Sum only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    return np.sum(energy_matrix[iu])


def lj_energy_and_grad(coords):
    """Compute Lennard-Jones energy and gradient."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 * r6
    energy_matrix = 4.0 * (r12 - r6)
    iu = np.triu_indices(N, k=1)
    energy = np.sum(energy_matrix[iu])
    
    # Gradient: dE/dr_i = -12*4*r_i*(r^-14 - r^-8) for each pair
    # Using the formula: F_ij = 24*(2*r^-14 - r^-8) * r_vec
    # grad_i = -sum_j F_ij for i != j
    # Let's compute it efficiently
    inv_r6 = 1.0 / r6
    inv_r12 = inv_r6 * inv_r6
    # Force magnitude factor for pair (i,j): 24*(2*inv_r12 - inv_r6)
    force_factor = 24.0 * (2.0 * inv_r12 - inv_r6)
    # Zero out diagonal
    np.fill_diagonal(force_factor, 0.0)
    # grad_i = -sum_j force_factor[i,j] * diff[i,j]
    # diff[i,j] = coords[i] - coords[j]
    grad = -np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def relax(positions, max_time=10.0):
    """Local relaxation using L-BFGS."""
    if _remaining_time() < 1.0:
        return positions
    
    def fun_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = lj_energy_and_grad(x)
        return e, g.reshape(-1)
    
    try:
        res = minimize(fun_and_grad, positions.reshape(-1), method='L-BFGS-B',
                       jac=True, options={'maxiter': 500, 'maxfun': 2000, 'gtol': 1e-8})
        if res.success or res.status == 0:
            return res.x.reshape(-1, 3)
        else:
            return res.x.reshape(-1, 3)
    except Exception:
        return positions


def construct():
    """Construct 38 LJ atoms near the FCC truncated octahedron global minimum."""
    best_coords = None
    best_energy = np.inf
    
    # Generate multiple random FCC-based seeds and relax them
    n_attempts = 200
    
    for attempt in range(n_attempts):
        if _remaining_time() < 5.0:
            break
        
        # Create FCC lattice points
        a = 1.12  # lattice constant (approximate minimum distance)
        # FCC points in unit cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
        fcc_basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        
        # Generate points in a cubic region
        n_per_side = 4  # 4x4x4 grid of FCC unit cells
        points = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                for k in range(n_per_side):
                    for b in fcc_basis:
                        points.append([i + b[0], j + b[1], k + b[2]])
        
        points = np.array(points)
        
        # Add small random perturbation
        points = points * a + np.random.randn(len(points), 3) * 0.05
        
        # Select N closest to center
        center = points.mean(axis=0)
        dists = np.sum((points - center) ** 2, axis=1)
        idx = np.argsort(dists)[:N]
        coords = points[idx].copy()
        
        # Relax
        relaxed = relax(coords)
        e = lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_coords = relaxed.copy()
    
    if best_coords is not None:
        # Final high-precision polish
        if _remaining_time() > 3.0:
            best_coords = relax(best_coords, max_time=15.0)
        return best_coords
    
    # Fallback: simple cubic lattice
    side = int(np.ceil(N ** (1 / 3)))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(pts[:N], dtype=float) * 1.12
