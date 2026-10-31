"""LJ38 global minimum search via FCC truncated octahedron parameterization + L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # Sum over upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    e = 4.0 * np.sum(r12[iu] - r6[iu])
    return e


def _get_fcc_truncated_octahedron_coords():
    """
    Generate the 38-atom FCC truncated octahedron.
    
    The FCC truncated octahedron for n=38 consists of:
    - 1 atom at center (0,0,0)
    - 12 atoms at FCC nearest neighbors: (±1,±1,0), (±1,0,±1), (0,±1,±1) * (1/√2)
    - 24 atoms at FCC second-nearest neighbors: (±2,0,0), (0,±2,0), (0,0,±2) type positions
    
    Actually, let me think more carefully. The standard LJ38 global minimum is a 
    truncated octahedron from FCC. Let me use the known structure.
    
    For FCC, the lattice points in a truncated octahedron shell structure:
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms (FCC 1st coordination) at distance a/√2 where a is lattice constant
    Shell 2: 24 atoms? No, that's too many. 1+12+24=37, need 38.
    
    Actually, let me reconsider. The LJ38 global minimum has 38 atoms.
    A common representation:
    - 1 center
    - 12 at first FCC shell (distance d1)
    - 24 at second FCC shell? That gives 37.
    - Plus 1 more? 
    
    Let me use a different approach. I'll generate FCC lattice points and select
    the 38 closest to center, then optimize.
    
    Actually, the known structure for LJ38:
    It's described as a truncated octahedron. Let me construct it properly.
    
    FCC lattice with basis at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell.
    
    Let me just generate many FCC points and pick the 38 nearest to center.
    """
    # Generate FCC lattice points in a box
    # FCC: points at integer (i,j,k) where i+j+k is even, plus face centers
    # Simpler: generate all (i,j,k) with i+j+k even (BCC-like) -- no, that's not FCC.
    
    # FCC in cubic cell of side a: corners + face centers
    # Let's use lattice constant = 1 for now, we'll scale later
    a = 1.0
    pts = []
    rng = range(-3, 4)
    for i in rng:
        for j in rng:
            for k in rng:
                # FCC: (i,j,k) where i,j,k all integers and i+j+k even
                # OR face-centered: (i+0.5, j+0.5, k), (i+0.5, j, k+0.5), (i, j+0.5, k+0.5)
                if (i + j + k) % 2 == 0:
                    pts.append((i * a, j * a, k * a))
                pts.append(((i + 0.5) * a, (j + 0.5) * a, k * a))
                pts.append(((i + 0.5) * a, j * a, (k + 0.5) * a))
                pts.append((i * a, (j + 0.5) * a, (k + 0.5) * a))
    
    pts = np.array(pts)
    # Center the cloud
    center = np.mean(pts, axis=0)
    pts = pts - center
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx][:N]
    return pts


def _lj_energy_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    iu = np.triu_indices(n, k=1)
    e = 4.0 * np.sum(r12[iu] - r6[iu])
    
    # Gradient: dE/dri = 4 * sum_j (12*ri^-13 - 6*ri^-7) * (ri/r)
    # More carefully: E = 4*sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4*sum_{j!=i} (-12*r_ij^-14 + 6*r_ij^-8) * (x_i - x_j)
    #         = 4*sum_{j!=i} (6*r_ij^-8 - 12*r_ij^-14) * (x_i - x_j)
    #         = 24*sum_{j!=i} (r_ij^-8 - 2*r_ij^-14) * (x_i - x_j)
    
    r8 = r6 * r2 ** (-1)  # r^-8 = r^-6 / r^2
    r14 = r12 * r2 ** (-1)  # r^-14 = r^-12 / r^2
    
    # For each pair (i,j), contribution to grad_i: 24*(r^-8 - 2*r^-14) * (pos_i - pos_j)
    # Use the full matrix
    coeff = 24.0 * (r8 - 2.0 * r14)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return e, grad


def construct():
    start_time = time.time()
    
    # Get initial FCC-based configuration
    pos0 = _get_fcc_truncated_octahedron_coords()
    
    # Scale to near the LJ minimum distance (~1.1225)
    # The FCC lattice constant should give nearest-neighbor distance ~ 1.1225
    # In FCC, nearest neighbor distance = a/√2
    # So a = 1.1225 * √2 ≈ 1.5874
    # But our generated points are already in some scale. Let's rescale.
    # Find the typical nearest-neighbor distance and scale
    diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn_dists = np.sqrt(np.min(r2, axis=1))
    median_nn = np.median(nn_dists)
    target_nn = 1.1225  # LJ minimum distance
    scale = target_nn / median_nn
    pos0 = pos0 * scale
    
    # Check minimum separation
    diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(np.min(r2))
    if min_dist < 0.5:
        # Expand slightly
        scale2 = 0.51 / min_dist
        pos0 = pos0 * scale2
    
    best_pos = pos0.copy()
    best_energy = _lj_energy(pos0)
    
    # Try L-BFGS-B relaxation
    try:
        result = minimize(
            fun=lambda x: _lj_energy_grad(x.reshape(N, 3))[0],
            x0=pos0.flatten(),
            jac=lambda x: _lj_energy_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 20}
        )
        if result.success or result.fun < best_energy:
            pos_relaxed = result.x.reshape(N, 3)
            # Check validity
            diff = pos_relaxed[:, np.newaxis, :] - pos_relaxed[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, np.inf)
            min_dist = np.sqrt(np.min(r2))
            if min_dist >= 0.5:
                best_pos = pos_relaxed
                best_energy = result.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        # Try a few more restarts with small perturbations
        np.random.seed(42)
        for restart in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            # Add small noise
            noise = np.random.randn(N, 3) * 0.05
            pos_try = best_pos + noise
            try:
                result = minimize(
                    fun=lambda x: _lj_energy_grad(x.reshape(N, 3))[0],
                    x0=pos_try.flatten(),
                    jac=lambda x: _lj_energy_grad(x.reshape(N, 3))[1].flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-10, 'maxls': 20}
                )
                if result.fun < best_energy:
                    pos_check = result.x.reshape(N, 3)
                    diff = pos_check[:, np.newaxis, :] - pos_check[np.newaxis, :, :]
                    r2 = np.sum(diff ** 2, axis=2)
                    np.fill_diagonal(r2, np.inf)
                    min_dist = np.sqrt(np.min(r2))
                    if min_dist >= 0.5:
                        best_pos = pos_check
                        best_energy = result.fun
            except Exception:
                pass
    
    return best_pos


# Pre-compute for speed: the energy function used by evaluator
def energy_fn(pos):
    return _lj_energy(pos)
