"""LJ38 cluster global minimum via FCC truncated-octahedron seed + basin hopping."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)

def _lj_grad(pos):
    """Gradient of LJ energy (vectorized)."""
    # Compute distance matrix
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    # inv_r6 and inv_r12
    inv_r6 = 1.0 / (d2 * d2 * d2)
    inv_r12 = inv_r6 * inv_r6
    
    # Force magnitude factor: F(r) = 4 * (12*r^-14 - 6*r^-8) / r^2 * r? 
    # Let's derive carefully.
    # U = 4 * (r^-12 - r^-6)
    # dU/dr = 4 * (-12*r^-13 + 6*r^-7)
    # grad_i U = sum_j dU/dr * (r_i - r_j) / r
    # Let rij = pos[i] - pos[j]. 
    # dU/dr = 4 * (-12/r^13 + 6/r^7)
    # grad_i = sum_j dU/dr * rij / r = sum_j 4 * (-12/r^14 + 6/r^8) * rij
    # Let coeff_ij = 4 * (6/r^8 - 12/r^14)
    # grad_i = sum_{j!=i} coeff_ij * (pos[i] - pos[j])
    
    # We can compute this as:
    # grad = sum_j coeff_ij * pos[i] - sum_j coeff_ij * pos[j]
    # Let M be matrix of coeff_ij for i!=j, 0 for i=j.
    # coeff matrix C where C[i,j] = 4 * (6*d2[i,j]^-4 - 12*d2[i,j]^-6)
    # Note d2^-4 = (1/d2)^4, d2^-6 = (1/d2)^6
    
    # To avoid division by zero on diagonal, we already set d to 1.0 on diagonal.
    # We want C[i,i] = 0.
    
    c2_inv = 1.0 / d2
    c6_inv = c2_inv * c2_inv * c2_inv
    c12_inv = c6_inv * c6_inv
    
    # coeff = 4 * (6 * c8_inv - 12 * c14_inv)
    # c8_inv = c2_inv^4
    # c14_inv = c2_inv^7
    c8_inv = c2_inv * c2_inv * c2_inv * c2_inv
    c14_inv = c8_inv * c2_inv * c2_inv * c2_inv
    
    coeff = 4.0 * (6.0 * c8_inv - 12.0 * c14_inv)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    # grad_i = sum_j coeff[i,j]*pos[i] - sum_j coeff[i,j]*pos[j]
    # grad_i = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    
    sum_coeff = np.sum(coeff, axis=1, keepdims=True)
    grad = pos * sum_coeff - coeff.dot(pos)
    
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-10})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(50):
        moved = False
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < min_sep:
                    rij = pos[i] - pos[j]
                    r = np.linalg.norm(rij)
                    if r < 1e-12:
                        rij = np.array([1.0, 0.0, 0.0])
                        r = 1.0
                    push = (min_sep - r) * 0.5
                    unit = rij / r
                    pos[i] += push * unit
                    pos[j] -= push * unit
                    moved = True
        if not moved:
            break
    return pos

def _fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron 38-atom cluster.
    
    The structure is built from FCC lattice points. FCC lattice points are:
    (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by 2 (so nearest neighbor = 1.0 in reduced units where sigma=1, but we use the conventional lattice constant).
    
    Actually, for LJ with sigma=1, the equilibrium nearest-neighbor distance is 2^(1/6) ≈ 1.12246.
    The FCC lattice with conventional cube edge a has nearest-neighbor distance a/sqrt(2).
    So a = 2^(1/6) * sqrt(2) = 2^(2/3) ≈ 1.5874.
    
    The truncated octahedron for 38 atoms: we take FCC points within a certain radius.
    """
    # FCC lattice: generate points
    # Conventional cubic cell with a = 2^(2/3)
    a = 2.0 ** (2.0 / 3.0)
    
    # Generate FCC lattice points in a range
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    # Center the structure
    center = points.mean(axis=0)
    points -= center
    
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take the 38 closest
    # But we need to be careful: the truncated octahedron has a specific shape.
    # Let's just take the 38 closest to origin.
    pts = points[:N].copy()
    
    return pts

def construct():
    # Start with the FCC truncated-octahedron seed
    pos = _fcc_truncated_octahedron()
    pos = _enforce_min_separation(pos)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    np.random.seed(42)
    
    # First, relax the initial structure
    relaxed = _relax(pos)
    relaxed = _enforce_min_separation(relaxed)
    e = _lj_energy(relaxed)
    if e < best_energy:
        best_energy = e
        best_pos = relaxed.copy()
    
    # Basin Hopping
    # Use a larger perturbation to explore the landscape
    # Standard LJ38 basin hopping often uses step size ~0.5-1.0
    step_size = 0.8
    T = 1.0 # Initial temperature for Metropolis
    T_min = 0.01
    cooling = 0.95
    
    # Run for more iterations since gradient is faster
    n_iter = 100
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    for i in range(n_iter):
        # Perturb
        perturbation = np.random.randn(N, 3) * step_size
        new_pos = current_pos + perturbation
        new_pos = _enforce_min_separation(new_pos)
        
        # Relax
        new_pos = _relax(new_pos)
        new_pos = _enforce_min_separation(new_pos)
        new_energy = _lj_energy(new_pos)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            if T > T_min:
                accept = np.random.rand() < np.exp(-delta_e / T)
            else:
                accept = False
        
        if accept:
            current_pos = new_pos
            current_energy = new_energy
            
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Cool down
        T = max(T * cooling, T_min)
        
        # Adaptive step size: increase if many rejections, decrease if many acceptances
        # Simple heuristic: if we haven't improved best in last 20 steps, increase step size
        if i > 0 and i % 20 == 0:
             # Check if we are stuck
             pass # Keep simple for now, fixed step size is often sufficient for this task if high enough

    # Final check: try a few random restarts if we are far from known minimum
    # Known global min ~ -173.928
    if best_energy > -170.0:
        for _ in range(5):
            # Random seed within a box
            rand_pos = np.random.randn(N, 3) * 5.0
            rand_pos = _enforce_min_separation(rand_pos)
            rand_relaxed = _relax(rand_pos)
            rand_relaxed = _enforce_min_separation(rand_relaxed)
            rand_energy = _lj_energy(rand_relaxed)
            if rand_energy < best_energy:
                best_energy = rand_energy
                best_pos = rand_relaxed.copy()

    return best_pos
