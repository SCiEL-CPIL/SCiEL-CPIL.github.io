"""Morse cluster global minima search: deterministic Newton-Raphson folding on dense grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global deadline: evaluator kills after 330s wall-clock.
_DEADLINE = time.time() + 310.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    pair_e = e * (e - 2.0)
    return 0.5 * np.sum(pair_e)

def _morse_grad(pos, rho):
    """Gradient of Morse energy."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    # dE/dr_ij = rho * e * (2*e - 2) = 2*rho*e*(e-1)
    # dE/dx_i = -sum_{j!=i} (dE/dr_ij) * (x_i - x_j) / r_ij
    grad_pair = 2.0 * rho * e * (e - 1.0) / dist  # shape (n,n)
    np.fill_diagonal(grad_pair, 0.0)
    grad = -np.sum(diff * grad_pair[:, :, np.newaxis], axis=1)
    return grad

def _morse_hessian_diag(pos, rho):
    """Approximate diagonal of Hessian via finite differences of gradient."""
    n = len(pos)
    if n < 2:
        return np.ones(n * 3)
    h = 1e-4
    grad0 = _morse_grad(pos, rho).flatten()
    hess_diag = np.zeros(n * 3)
    for i in range(n * 3):
        pos_p = pos.copy()
        pos_p[i // 3, i % 3] += h
        pos_m = pos.copy()
        pos_m[i // 3, i % 3] -= h
        grad_p = _morse_grad(pos_p, rho).flatten()
        grad_m = _morse_grad(pos_m, rho).flatten()
        hess_diag[i] = (grad_p[i] - grad_m[i]) / (2 * h)
    return hess_diag

def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_d

def _generate_dense_bcc_grid(n, spacing=0.55, rng=None):
    """Generate n points on a dense BCC grid, sorted by distance from center."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Estimate needed grid size
    # BCC has 2 atoms per unit cell
    side = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    
    coords = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # Corner atoms
                coords.append([i * spacing, j * spacing, k * spacing])
                # Body center atom
                coords.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    coords = np.array(coords)
    # Sort by distance from center and take n closest
    center = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx].copy()
    points -= np.mean(points, axis=0)
    return points

def _newton_raphson_fold(pos, rho, max_iter=200, damping=0.5, min_step=1e-8):
    """
    Deterministic Newton-Raphson with adaptive damping.
    Uses gradient and approximate Hessian diagonal for step direction.
    """
    n = len(pos)
    if n < 2:
        return pos
    
    current_pos = pos.copy()
    current_e = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_e = current_e
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 2.0:
            break
        
        # Compute gradient
        grad = _morse_grad(current_pos, rho)
        grad_norm = np.linalg.norm(grad)
        
        # Convergence check
        if grad_norm < 1e-10:
            break
        
        # Approximate Hessian diagonal for scaling
        hess_diag = _morse_hessian_diag(current_pos, rho).reshape(-1, 3)
        # Ensure positive definite diagonal
        hess_diag = np.maximum(hess_diag, 1e-6)
        
        # Newton step: -H^{-1} g, approximated as -g / H_diag
        step = -grad / hess_diag
        step_norm = np.linalg.norm(step)
        
        # Adaptive damping
        if step_norm > 0.5:
            step *= 0.5 / step_norm
            step_norm = 0.5
        if step_norm < min_step:
            break
        
        # Try full step
        candidate = current_pos + step
        candidate_e = _morse_energy(candidate, rho)
        
        # If energy increases, reduce step
        if candidate_e > current_e:
            # Backtracking line search
            alpha = 0.5
            for _ in range(10):
                candidate = current_pos + alpha * step
                candidate_e = _morse_energy(candidate, rho)
                if candidate_e < current_e:
                    break
                alpha *= 0.5
            
            if candidate_e > current_e:
                # Step was bad, try smaller step
                candidate = current_pos + 0.1 * step
                candidate_e = _morse_energy(candidate, rho)
                if candidate_e > current_e:
                    break  # Stuck
        
        # Update
        current_pos = candidate
        current_e = candidate_e
        
        if current_e < best_e:
            best_e = current_e
            best_pos = current_pos.copy()
    
    return best_pos

def _lbfgs_polish(pos, rho, max_iter=200):
    """High-precision polish using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()
    
    def fun(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def grad(x):
        return _morse_grad(x.reshape(-1, 3), rho).flatten()
    
    try:
        res = minimize(fun, x0, jac=grad, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success or res.fun < fun(x0):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return pos

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 100 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # --- Phase 1: Initialize on dense BCC grid ---
    # Start with a compact BCC arrangement
    pos0 = _generate_dense_bcc_grid(size, spacing=0.55, rng=rng)
    
    # --- Phase 2: Deterministic Newton-Raphson folding ---
    # Apply NR folding to converge to a local minimum
    nr_pos = _newton_raphson_fold(pos0, parameter, max_iter=150, damping=0.5)
    
    if _check_min_dist(nr_pos):
        nr_e = _morse_energy(nr_pos, parameter)
        if nr_e < best_energy:
            best_energy = nr_e
            best_positions = nr_pos.copy()
    
    # --- Phase 3: L-BFGS-B polish for high precision ---
    if best_positions is not None and time.time() < _DEADLINE - 5.0:
        try:
            polished = _lbfgs_polish(best_positions, parameter, max_iter=300)
            if _check_min_dist(polished):
                pol_e = _morse_energy(polished, parameter)
                if pol_e < best_energy:
                    best_energy = pol_e
                    best_positions = polished
        except Exception:
            pass
    
    # --- Phase 4: If time remains, try a few structured seeds with NR ---
    if time.time() < _DEADLINE - 10.0:
        # Try FCC-like start
        side = int(np.ceil((size / 4.0) ** (1.0/3.0))) + 2
        fcc_coords = []
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                for k in range(-side, side + 1):
                    for (dx, dy, dz) in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                        fcc_coords.append([(i+dx)*0.55, (j+dy)*0.55, (k+dz)*0.55])
        fcc_coords = np.array(fcc_coords)
        center = np.mean(fcc_coords, axis=0)
        dists = np.linalg.norm(fcc_coords - center, axis=1)
        idx = np.argsort(dists)[:size]
        fcc_pos = fcc_coords[idx].copy()
        fcc_pos -= np.mean(fcc_pos, axis=0)
        
        fcc_nr = _newton_raphson_fold(fcc_pos, parameter, max_iter=100)
        if _check_min_dist(fcc_nr):
            fcc_e = _morse_energy(fcc_nr, parameter)
            if fcc_e < best_energy:
                best_energy = fcc_e
                best_positions = fcc_nr.copy()
        
        # Final polish
        if best_positions is not None and time.time() < _DEADLINE - 3.0:
            try:
                polished = _lbfgs_polish(best_positions, parameter, max_iter=200)
                if _check_min_dist(polished):
                    pol_e = _morse_energy(polished, parameter)
                    if pol_e < best_energy:
                        best_energy = pol_e
                        best_positions = polished
            except Exception:
                pass
    
    # Fallback
    if best_positions is None:
        best_positions = _generate_dense_bcc_grid(size, spacing=0.55, rng=rng)
        
    return best_positions
# EVOLVE-BLOCK-END
