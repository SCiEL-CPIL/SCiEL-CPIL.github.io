"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # points is (N, 3)
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    # Get unique pairs
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    # diff[i, j] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists[i, j] = ||c[i] - c[j]||
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-9, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        # Return current best if timeout? minimize doesn't easily return intermediate state.
        # We rely on the outer loop checking time.
        return init_pts, False
    except Exception:
        return init_pts, False

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s

    # ---- Seed: Simple cubic 5x5x5 grid ----
    # 5^3 = 125 points. We take the first 100.
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy: Run BFGS with increasing target radii
    # Start with a target slightly above the grid score (0.1)
    # Grid score is 0.1 (spacing 0.2, radius 0.1).
    # Best known is ~0.11138.
    
    # We will try a sequence of targets.
    # If a target fails to improve, we keep the best points found so far.
    
    targets = [0.105, 0.110, 0.112, 0.114, 0.116, 0.118]
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Add small noise to break symmetry and help escape local minima
        # Noise magnitude should be small enough not to cause massive violations
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=2000, 
            gtol=1e-10,
            time_limit=DEADLINE - (time.time() - t0) - 5.0
        )
        
        if np.all(np.isfinite(opt_pts)):
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                # Update current_pts to the best found so far for next iteration
                current_pts = opt_pts.copy()
            else:
                # Even if r didn't improve, the configuration might be better positioned
                # for a larger target. But usually, we stick to the best radius.
                # However, if the optimizer converged to a local min with lower r,
                # using it as start for a *larger* target might be bad.
                # We'll stick to best_pts as the base for next noise if no improvement.
                pass

    # If time permits, try random restarts with the best target found
    # or a slightly higher target
    if time.time() - t0 < DEADLINE - 15.0:
        # Try a few restarts from the best configuration with larger noise
        for i in range(5):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            
            noise = rng.normal(0, 0.02, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            # Use a target slightly above best_r
            target_r = best_r * 1.01
            
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=1000, 
                gtol=1e-9,
                time_limit=DEADLINE - (time.time() - t0) - 5.0
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()

    # Final Polish: High precision optimization on the best found radius
    if time.time() - t0 < DEADLINE - 5.0 and best_r > 0.1:
        # Try to optimize exactly for the best_r found, with very tight tolerance
        # This can squeeze out a tiny bit more if the previous runs stopped early
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            best_r, 
            max_iter=5000, 
            gtol=1e-12,
            time_limit=DEADLINE - (time.time() - t0) - 2.0
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()

    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
        
    return best_pts
# EVOLVE-BLOCK-END
