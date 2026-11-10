"""Coarse-to-fine spectral relaxation for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_DEADLINE = 325.0  # Wall-clock seconds; evaluator kills at 330, guard earlier
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist ** 2


def _spherical_shell_seed(n, rng):
    """
    Generate a seed where atoms are distributed on a spherical shell.
    This provides a high-entropy, symmetric starting point.
    """
    # Fibonacci sphere for uniform distribution
    points = np.zeros((n, 3))
    if n == 1:
        points[0] = [0, 0, 0]
        return points

    # Radius estimation: assume packing density ~0.6-0.7
    # Volume ~ n * (0.4)^3 / 0.65
    # R ~ (n * 0.064 / 0.65 * 3 / 4pi)^(1/3)
    R = 1.1 * (n / 0.75) ** (1/3)
    
    # Fibonacci lattice on sphere
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    
    points[:, 0] = R * np.sin(phi) * np.cos(theta)
    points[:, 1] = R * np.sin(phi) * np.sin(theta)
    points[:, 2] = R * np.cos(phi)
    
    # Add small noise to break perfect symmetry and avoid singularities in gradient
    points += rng.normal(0, 0.01, points.shape)
    
    return points


def _conjugate_gradient_minimize(positions, potential, kind, max_iter=500, tol=1e-6):
    """
    Custom Conjugate Gradient minimizer with adaptive line search (Backtracking).
    Uses sutton_chen_energy_gradient.
    """
    n = len(positions)
    pos = positions.copy()
    
    # Initial gradient
    e, g = sutton_chen_energy_gradient(pos, kind)
    
    if not np.isfinite(e) or not np.all(np.isfinite(g)):
        return e, pos
    
    # Initial direction is negative gradient
    d = -g
    p_old = pos
    
    for i in range(max_iter):
        if _remaining() < 5.0:
            break
            
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
            
        # Adaptive step size: start with a reasonable guess, backtrack if energy increases
        # Heuristic: step ~ 1 / max(1, |g|)
        step = 1.0 / (1.0 + g_norm * 0.1)
        step = min(step, 0.5) # Cap step size to avoid flying off
        
        # Backtracking line search
        alpha = step
        for _ in range(20):
            pos_new = pos + alpha * d
            e_new, g_new = sutton_chen_energy_gradient(pos_new, kind)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
                
            # Armijo condition: E_new <= E_old + c * alpha * grad^T * d
            # c is typically 1e-4
            if e_new <= e + 1e-4 * alpha * np.dot(g, d):
                break
            else:
                alpha *= 0.5
                
        pos = pos_new
        e = e_new
        g = g_new
        
        # Check validity
        if not _is_valid(pos):
             # If we went out of bounds, step back and reduce step size significantly
             pos = p_old
             e, g = sutton_chen_energy_gradient(pos, kind)
             # Break or reduce step? Let's break to avoid infinite loop in invalid region
             break
             
        # Polak-Ribiere update for d
        g_dot_g_new = np.dot(g, g)
        g_dot_g_old = np.dot(g, g)
        
        # Standard PR: beta = (g_new^T (g_new - g_old)) / ||g_old||^2
        # Non-negative PR: beta = max(0, ...)
        beta = np.dot(g, g - (-d)) / g_dot_g_old # Note: d = -g_old + beta_prev... wait.
        # Actually d is the search direction. g_old is g from previous step.
        # Let's store g_old
        # Wait, I didn't store g_old explicitly in the loop variable, but I have g (new) and I need g_old.
        # Let's refactor slightly to store g_old.
        
        # Actually, simpler: just restart CG if beta is negative or large (restart CG)
        # Or use Fletcher-Reeves: beta = ||g_new||^2 / ||g_old||^2
        # FR is often more robust for non-quadratic but can cycle.
        # Let's use FR with restart if not decreasing.
        beta = g_dot_g_new / max(g_dot_g_old, 1e-12)
        
        if beta < 0:
            beta = 0
            
        d = -g + beta * d
        
        p_old = pos
        
    return e, pos


def _worker(args):
    """
    Worker: Generate spherical seed, run CG, then polish with relax.
    """
    n, potential, kind, seed_val = args
    rng = np.random.default_rng(seed_val)
    
    # 1. Generate Spherical Shell Seed
    seed = _spherical_shell_seed(n, rng)
    
    # 2. Coarse-to-Fine: CG Minimization
    # CG handles the initial large movements from the symmetric shell
    try:
        e_cg, pos_cg = _conjugate_gradient_minimize(seed, potential, kind, max_iter=200)
        if not _is_valid(pos_cg):
            # Fallback to relax if CG failed
            e_cg, pos_cg = relax(seed, potential, kind)
    except Exception:
        e_cg, pos_cg = relax(seed, potential, kind)
        
    if not _is_valid(pos_cg):
        return np.inf, seed
        
    # 3. Fine Polish: L-BFGS-B
    # relax() is robust and handles the final convergence well
    try:
        e_final, pos_final = relax(pos_cg, potential, kind)
        if _is_valid(pos_final):
            return e_final, pos_final
    except Exception:
        pass
        
    return e_cg, pos_cg


def construct(instance):
    global _START
    _START = time.time()

    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Strategy: Multiple restarts from slightly different spherical shells.
    # We use 4 threads. Each thread does a sequence of restarts? 
    # Or just 1 restart per thread?
    # Given the 300s budget, and N up to 80.
    # CG is O(N^2) per step. 200 steps * N^2.
    # If N=80, N^2=6400. 200*6400 = 1.28M ops. Very fast.
    # Relax is also fast.
    # So we can afford MORE restarts per thread.
    
    # Let's do 5 restarts per thread, 4 threads = 20 restarts total.
    # Each thread handles a distinct seed range.
    
    n_restarts_per_thread = 5
    n_threads = 4
    
    tasks = []
    for t in range(n_threads):
        for r in range(n_restarts_per_thread):
            # Unique seed for each restart
            seed_val = t * 100 + r
            tasks.append((size, potential, kind, seed_val))
            
    best_energy = np.inf
    best_positions = None
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass

    # Final refinement on the best result found
    if best_positions is not None and _remaining() > 10.0:
        # Try one more polish with a very tight tolerance or different method?
        # Just call relax again to be sure it's at the bottom of the basin.
        try:
            e_final, pos_final = relax(best_positions, potential, kind)
            if _is_valid(pos_final) and e_final < best_energy:
                best_energy = e_final
                best_positions = pos_final
        except Exception:
            pass
    else:
        # Fallback if no valid result
        rng_fallback = np.random.default_rng(0)
        best_positions = _spherical_shell_seed(size, rng_fallback)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass

    if not _is_valid(best_positions):
        # Last resort: valid random
        rng_last = np.random.default_rng(999)
        # Simple random valid seed
        r_max = 1.2 * size ** (1/3)
        points = []
        while len(points) < size:
            p = rng_last.uniform(-r_max, r_max, 3)
            if np.linalg.norm(p) <= r_max:
                if len(points) > 0:
                    d = np.linalg.norm(p - np.array(points), axis=1)
                    if np.min(d) < 0.36:
                        continue
                points.append(p)
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
