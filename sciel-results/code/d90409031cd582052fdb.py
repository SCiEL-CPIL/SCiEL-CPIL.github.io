"""Rewritten program: Simulated Annealing for Logarithmic Energy Minimization."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _delta_energy(pts, i, new_p):
    """Calculate change in energy if point i is replaced by new_p."""
    # Distances from i to all others
    diffs_old = pts[i] - pts
    dists_old_sq = np.sum(diffs_old * diffs_old, axis=1)
    
    diffs_new = new_p - pts
    dists_new_sq = np.sum(diffs_new * diffs_new, axis=1)
    
    # Exclude self (index i)
    # We need sum over j != i
    # d_old = sqrt(dists_old_sq), d_new = sqrt(dists_new_sq)
    # term = log(1/d_new) - log(1/d_old) = -log(d_new) + log(d_old) = log(d_old/d_new)
    
    dists_old = np.sqrt(np.maximum(dists_old_sq, 1e-30))
    dists_new = np.sqrt(np.maximum(dists_new_sq, 1e-30))
    
    # Mask out self
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    # If distance is 0 (should not happen on sphere if distinct), handle log
    # Using log(d_old/d_new)
    ratio = dists_old[mask] / dists_new[mask]
    # If ratio is 0 or inf, handle?
    # log(0) is -inf, log(inf) is inf.
    # Since we project to sphere, min dist > 0 unless points coincide.
    # If points coincide, energy is -inf. We want to avoid that.
    # If new_p is very close to some j, d_new is small, log(d_old/d_new) is large positive -> energy increases.
    # If new_p is far, d_new is large, log(d_old/d_new) is negative -> energy decreases.
    
    return np.sum(np.log(ratio))

def _optimize_sa(pts0, steps=200000, T0=1.0, T_end=1e-6, rng_seed=42):
    """Run Simulated Annealing."""
    rng = np.random.default_rng(rng_seed)
    pts = pts0.copy()
    current_energy = _log_energy(pts)
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Pre-compute indices? No, random selection.
    
    for k in range(steps):
        # Cooling schedule
        # Geometric: T = T0 * (T_end/T0)^(k/steps)
        # Or exponential: T = T0 * exp(-k/tau)
        # Let's use geometric for robustness
        T = T0 * (T_end / T0) ** (k / steps)
        
        # Select random point
        i = rng.integers(0, N)
        
        # Generate random perturbation
        # Step size should decrease? Or constant?
        # Constant step size with projection usually works if T handles acceptance.
        # Let's use a Gaussian perturbation in the tangent space.
        # Random vector in R3
        v = rng.normal(size=3)
        # Project to tangent space at pts[i]
        # v_tan = v - (v . p) p
        p = pts[i]
        v_tan = v - np.dot(v, p) * p
        norm_v = np.linalg.norm(v_tan)
        if norm_v < 1e-12:
            continue
        v_tan /= norm_v
        
        # Step size alpha. Maybe scale with T?
        # Let's use a fixed small step for SA, e.g., 0.1 to 0.5 radians?
        # Or random step size?
        # Let's try a fixed step size of 0.2 (approx 11 degrees)
        step_size = 0.2
        new_p = p + step_size * v_tan
        new_p = _project_to_sphere(new_p.reshape(1,3))[0]
        
        # Calculate delta E
        delta_e = _delta_energy(pts, i, new_p)
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            # Accept
            pts[i] = new_p
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                
    return best_pts, best_energy

def _optimize_lbfgsb(pts0, maxiter=500, ftol=1e-16, gtol=1e-12):
    """Run L-BFGS-B optimization from given starting points."""
    def _energy_and_grad(pts):
        pts = _project_to_sphere(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-30)
        
        iu = np.triu_indices(N, k=1)
        d = np.sqrt(dist_sq[iu])
        d = np.maximum(d, 1e-15)
        energy = -np.sum(np.log(d))
        
        weight = 1.0 / dist_sq
        np.fill_diagonal(weight, 0.0)
        
        grad = -np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad -= grad_dot * pts
        
        return energy, grad

    try:
        result = minimize(
            lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
            pts0.flatten(),
            jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': False, 'maxls': 50}
        )
        if result.fun < 1e10:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            cand_energy = _log_energy(candidate)
            if cand_energy < 1e10:
                return candidate, cand_energy
    except Exception:
        pass
    return None, np.inf

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    # Start with Fibonacci spiral
    pts0 = _fibonacci_spiral(N)
    
    # Run Simulated Annealing
    # 120s limit. 
    # 200k steps in Python loop: ~2-5 seconds?
    # _delta_energy involves numpy ops on arrays of size 100.
    # 200k * 100 ops = 20M ops. Should be fast.
    # Let's run 300k steps to be safe, or check time.
    # We can run multiple SA restarts if time permits?
    # Let's stick to one long SA run with a good schedule.
    
    pts_sa, energy_sa = _optimize_sa(pts0, steps=300000, T0=2.0, T_end=1e-5, rng_seed=42)
    
    # Polish with L-BFGS-B
    pts_final, energy_final = _optimize_lbfgsb(pts_sa, maxiter=1000, ftol=1e-16, gtol=1e-14)
    
    if pts_final is None or energy_final > energy_sa:
        pts_final = pts_sa
        energy_final = energy_sa
        
    # Validation
    pts_final = _project_to_sphere(pts_final)
    diff = pts_final[:, np.newaxis, :] - pts_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        rng = np.random.default_rng(42)
        pts_final = pts_final + rng.normal(0, 1e-6, pts_final.shape)
        pts_final = _project_to_sphere(pts_final)
        
    return pts_final
# EVOLVE-BLOCK-END
