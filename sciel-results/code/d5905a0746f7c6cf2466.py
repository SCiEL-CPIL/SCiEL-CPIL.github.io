import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 135.0  # Safety margin before 150s kill

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # dE/dpos_i = sum_{j!=i} 4 * (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    # = sum_{j!=i} 4 * (-12*r6/r2 + 6*r2^(-4)) * diff
    # r2^(-4) = r6 * r2^(-1)
    coeff = 4.0 * (-12.0 * r6 / r2 + 6.0 * r6 / r2)
    # Actually: d/dr (r^-12) = -12 r^-13, d/dr (r^-6) = -6 r^-7
    # dE/dri = 4*(-12*ri^-13 + 6*ri^-7)
    # dE/dpos_i = sum_j dE/dri * dri/dpos_i = sum_j 4*(-12*ri^-13 + 6*ri^-7) * (pos_i - pos_j)/ri
    # = sum_j 4*(-12*ri^-12 + 6*ri^-6) * (pos_i - pos_j)
    # = sum_j 4*(-12*r12 + 6*r6) * diff
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _min_distance(pos):
    """Compute minimum pairwise distance."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    return np.sqrt(r2.min())

def _get_mackay_directions():
    """Get unit vectors for Mackay icosahedron (1 center + 12 inner + 42 outer)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (adjacent vertices)
    dots = verts_unit @ verts_unit.T
    edges = []
    for i in range(12):
        d = dots[i].copy()
        d[i] = -np.inf
        neighbor_indices = np.argsort(d)[-5:]
        for j in neighbor_indices:
            if i < j:
                edges.append((i, j))
    
    # Outer shell: 12 at 2*v_i, 30 at v_i + v_j (adjacent)
    outer = []
    for i in range(12):
        v = 2 * verts_unit[i]
        outer.append(v / np.linalg.norm(v))
    for i, j in edges:
        v = verts_unit[i] + verts_unit[j]
        outer.append(v / np.linalg.norm(v))
    
    inner = verts_unit
    return inner, np.array(outer)

def _make_mackay_from_radii(R_inner, R_outer, center_offset=None):
    """Build Mackay icosahedron with given radii."""
    inner_dirs, outer_dirs = _get_mackay_directions()
    
    pos = np.zeros((1, 3))
    if center_offset is not None:
        pos[0] = center_offset
    
    pos_in = R_inner * inner_dirs
    pos_out = R_outer * outer_dirs
    
    positions = np.vstack([pos, pos_in, pos_out])
    return positions

def _relax_lbfgs(pos, maxiter=200, ftol=1e-10, gtol=1e-8):
    """Relax using L-BFGS-B with analytical gradient."""
    result = minimize(
        _lj_energy, pos, jac=_lj_energy_grad,
        method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x, result.fun

def construct():
    start_time = time.time()
    
    # Initial Mackay icosahedron
    R_inner = 1.12
    R_outer = 2.2
    pos0 = _make_mackay_from_radii(R_inner, R_outer)
    
    # Relax initial structure
    pos_best, e_best = _relax_lbfgs(pos0)
    
    # Check minimum distance
    if _min_distance(pos_best) < 0.5:
        # Scale up if too close
        scale = 0.5 / _min_distance(pos_best) * 1.01
        pos_best = pos_best * scale
        pos_best, e_best = _relax_lbfgs(pos_best)
    
    # Differential evolution population
    pop_size = 8
    population = [pos_best.copy()]
    
    # Initialize population with diverse structures
    rng = np.random.default_rng(42)
    for i in range(pop_size - 1):
        # Perturb Mackay with random noise
        noise = rng.normal(0, 0.1, pos0.shape)
        pos_i = pos0 + noise
        # Relax
        pos_i, e_i = _relax_lbfgs(pos_i, maxiter=100)
        if _min_distance(pos_i) >= 0.5:
            population.append(pos_i)
        else:
            # Try with larger noise
            noise = rng.normal(0, 0.2, pos0.shape)
            pos_i = pos0 + noise
            pos_i, e_i = _relax_lbfgs(pos_i, maxiter=100)
            if _min_distance(pos_i) >= 0.5:
                population.append(pos_i)
            else:
                population.append(pos_best.copy())
    
    # Track best
    energies = [_lj_energy(p) for p in population]
    best_idx = np.argmin(energies)
    pos_best = population[best_idx].copy()
    e_best = energies[best_idx]
    
    # Differential evolution loop
    max_generations = 20
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover probability
    
    for gen in range(max_generations):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for i in range(pop_size):
            if time.time() - start_time > DEADLINE - 5:
                    break
            
            # Select three distinct random individuals
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            
            # Mutation: v = a + F * (b - c)
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover: binomial crossover
            trial = np.zeros_like(population[i])
            j_rand = rng.integers(0, N * 3)
            for j in range(N * 3):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            # Reshape and relax
            trial_pos = trial.reshape(N, 3)
            
            # Check minimum distance before relaxation
            if _min_distance(trial_pos) < 0.3:
                # Too close, try with scaled version
                trial_pos = trial_pos * 1.1
                if _min_distance(trial_pos) < 0.5:
                    continue
            
            # Relax
            trial_pos, e_trial = _relax_lbfgs(trial_pos, maxiter=150)
            
            if _min_distance(trial_pos) < 0.5:
                continue
            
            # Selection
            if e_trial < energies[i]:
                population[i] = trial_pos
                energies[i] = e_trial
                if e_trial < e_best:
                    pos_best = trial_pos.copy()
                    e_best = e_trial
                best_idx = i
    
    # Final polish with tighter tolerance
    if time.time() - start_time < DEADLINE - 5:
        pos_best, e_best = _relax_lbfgs(pos_best, maxiter=500, ftol=1e-12, gtol=1e-10)
    
    # Ensure minimum distance constraint
    if _min_distance(pos_best) < 0.5:
        # Scale up
        scale = 0.5 / _min_distance(pos_best) * 1.02
        pos_best = pos_best * scale
        pos_best, e_best = _relax_lbfgs(pos_best, maxiter=100)
    
    return pos_best
