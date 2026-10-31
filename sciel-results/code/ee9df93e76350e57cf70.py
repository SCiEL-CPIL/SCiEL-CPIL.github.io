"""Lennard-Jones 38-atom cluster global minimum search."""
import numpy as np
import time

N = 38

def _lj_energy(pos):
    """Compute total LJ energy for positions (N,3)."""
    n = len(pos)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Sum over all pairs, divide by 2 since we count each pair twice
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    return energy

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # grad_i = sum_{j != i} 4 * (-12*r^-13 * r_ij + 6*r^-7 * r_ij)
    # = sum_{j != i} 4 * r_ij * (-12*r12 + 6*r6) / r2
    # factor[i,j] = 4 * (-12*r12[i,j] + 6*r6[i,j]) / r2[i,j]
    factor = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * rij, axis=1)
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    return energy, grad

def _relax(pos, max_iter=5000):
    """Local relaxation using L-BFGS via scipy."""
    from scipy.optimize import minimize
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    try:
        res = minimize(fun, pos.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-10})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy(pos)

def _basin_hopping_step(pos, temp, rng):
    """One basin hopping step: perturb, relax, accept/reject."""
    # Perturb each atom by random Gaussian displacement
    delta = rng.normal(0, 0.1, size=pos.shape)
    new_pos = pos + delta
    # Ensure minimum distance constraint
    for _ in range(10):
        rij = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        r2 = np.sum(rij * rij, axis=2)
        np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2)
        if np.min(r[r < 0.5]) >= 0.5:
            break
        # Push apart atoms that are too close
        mask = r < 0.5
        if not np.any(mask):
            break
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    d = new_pos[i] - new_pos[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-10:
                        push = (0.5 - dist) * d / dist
                        new_pos[i] += push
                        new_pos[j] -= push
    new_pos, new_e = _relax(new_pos, max_iter=2000)
    return new_pos, new_e

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Initial guess: truncated octahedron-like FCC structure
    # Start with a simple cubic and relax
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    cx, cy, cz = (side-1)/2.0, (side-1)/2.0, (side-1)/2.0
    points.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
    pos = np.array(points[:N], dtype=float) * 1.12
    
    # Initial relaxation
    pos, best_e = _relax(pos, max_iter=5000)
    best_pos = pos.copy()
    
    # Basin hopping
    current_pos = pos.copy()
    current_e = best_e
    temperature = 0.5
    min_temp = 0.01
    
    while time.time() < deadline:
        new_pos, new_e = _basin_hopping_step(current_pos, temperature, rng)
        
        # Metropolis acceptance
        delta_e = new_e - current_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temperature):
            current_pos = new_pos
            current_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos.copy()
        
        # Cool down
        temperature *= 0.995
        if temperature < min_temp:
            temperature = min_temp
    
    # Final relaxation of best
    best_pos, final_e = _relax(best_pos, max_iter=10000)
    if final_e < best_e:
        best_e = final_e
        best_pos = best_pos
    
    return best_pos
