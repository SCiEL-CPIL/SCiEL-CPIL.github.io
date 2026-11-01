"""Lennard-Jones cluster global minimum, 55 atoms – basin-hopping with L-BFGS relaxation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 55
SIGMA = 1.0
EPSILON = 1.0
MIN_DIST = 0.5
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0  # seconds before limit to stop


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms.
    
    pos: (N, 3) array
    Returns: (energy, gradient) where gradient is (N, 3)
    """
    n = pos.shape[0]
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (n, n)
    # Avoid division by zero for self-interactions
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** (-3)  # 1/r^6
    r12 = r6 ** 2    # 1/r^12
    
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # Using the full matrix and dividing by 2
    pair_energy = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_i = 4 * sum_{j != i} (-12*r^-14 + 6*r^-8) * (r_i - r_j)
    # = 24 * sum_{j != i} (2*r^-14 - r^-8) * (r_i - r_j)
    # Let's compute more carefully.
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} d/dx_i (r_ij^-12 - r_ij^-6)
    # d/dr_ij (r^-12) = -12 r^-13, d/dr_ij (r^-6) = -6 r^-7
    # dE/dx_i = 4 * sum_{j != i} (-12 r^-13 + 6 r^-7) * d(r_ij)/dx_i
    # d(r_ij)/dx_i = (x_i - x_j) / r_ij
    # So dE/dx_i = 4 * sum_{j != i} (-12 r^-13 + 6 r^-7) * (x_i - x_j) / r
    # = 4 * sum_{j != i} (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # = 24 * sum_{j != i} (2 r^-14 - r^-8) * (x_i - x_j)
    
    # Using r2:
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    r14 = r2 ** (-7)
    r8 = r2 ** (-4)
    
    # Factor for each pair
    # dE/dx_i = 24 * sum_j (2*r14 - r8) * (x_i - x_j)
    # We can compute this as: for each i, sum over j of (2*r14[i,j] - r8[i,j]) * (pos[i] - pos[j])
    # Let's build a matrix F where F[i,j] = 2*r14[i,j] - r8[i,j]
    F = 2.0 * r14 - r8
    np.fill_diagonal(F, 0.0)
    
    # grad[i] = 24 * sum_j F[i,j] * (pos[i] - pos[j])
    # = 24 * (pos[i] * sum_j F[i,j] - sum_j F[i,j] * pos[j])
    # = 24 * (pos[i] * row_sum_F[i] - F @ pos[:, i])  -- but F is 2D, need to be careful
    
    # Let's do it vectorized:
    # grad = 24 * (F.sum(axis=1)[:, np.newaxis] * pos - F @ pos)
    row_sum_F = F.sum(axis=1)  # (n,)
    F_pos = F @ pos  # (n, 3)
    grad = 24.0 * (row_sum_F[:, np.newaxis] * pos - F_pos)
    
    return energy, grad


def _relax(pos):
    """Relax positions using L-BFGS."""
    def func_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.ravel()
    
    x0 = pos.ravel()
    result = minimize(func_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12})
    return result.x.reshape(-1, 3)


def _fbb_seed():
    """Face-centered cubic fragment seed for 55 atoms."""
    # FCC lattice: points at integer coordinates where all coords are integers and sum is even
    # Or use the standard FCC basis
    # Generate FCC points within a sphere
    points = []
    max_coord = 4
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (i,j,k) all integers, and i+j+k even
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    # Take N points
    pts = np.array(points[:N], dtype=float)
    # Scale: FCC nearest neighbor distance is sqrt(2) in these units, LJ min is 2^(1/6) ≈ 1.122
    # So scale factor = 2^(1/6) / sqrt(2)
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    return pts * scale


def _random_seed(rng):
    """Random seed within a sphere."""
    # Generate random points in a sphere of radius ~5
    radius = 5.0
    while True:
        pts = rng.uniform(-radius, radius, size=(N, 3))
        # Check minimum distance
        min_d = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
        if min_d >= MIN_DIST:
            return pts
        # If not, try again (should be rare for N=55 in radius 5)


def _basin_hopping(seed, rng, max_iters=50, step_size=0.5):
    """Basin hopping: perturb, relax, accept/reject."""
    current_pos = seed.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for it in range(max_iters):
        # Perturb
        perturbation = rng.normal(0, step_size, size=(N, 3))
        new_pos = current_pos + perturbation
        
        # Relax
        relaxed_pos = _relax(new_pos)
        relaxed_energy, _ = _lj_energy_and_grad(relaxed_pos)
        
        # Accept/reject (always accept for simplicity, or use Metropolis)
        # For simplicity, always accept (this is effectively steepest descent with random kicks)
        current_pos = relaxed_pos
        current_energy = relaxed_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def construct():
    """Construct 55-atom LJ cluster by basin-hopping from diverse seeds."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    rng = np.random.RandomState(42)
    
    # Best result across all runs
    best_pos = None
    best_energy = np.inf
    
    # Seed 1: FCC fragment
    seed_fcc = _fbb_seed()
    pos, energy = _basin_hopping(seed_fcc, rng, max_iters=30, step_size=0.3)
    if energy < best_energy:
        best_energy = energy
        best_pos = pos.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_pos
    
    # Seed 2-5: Random seeds
    for i in range(5):
        if time.time() > deadline - 15:
            break
        
        seed_rand = _random_seed(rng)
        pos, energy = _basin_hopping(seed_rand, rng, max_iters=20, step_size=0.4)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_pos
    
    # Final refinement: more iterations from the best position
    if time.time() < deadline - 5:
        pos, energy = _basin_hopping(best_pos, rng, max_iters=30, step_size=0.2)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    return best_pos
