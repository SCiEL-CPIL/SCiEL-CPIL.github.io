import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

def compute_score(centers):
    """Compute the exact radius for a given set of centers."""
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    
    return min(min_pair, min_boundary)

def soft_min_objective(centres_flat, beta=100.0):
    """
    Smooth approximation of the negative minimum radius.
    We want to maximize r = min(d_ij/2, b_i).
    So we minimize -soft_min(d_ij/2, b_i).
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    """
    centres = centres_flat.reshape(N, 3)
    
    # Boundary terms: 0.5 - |c_ik|
    boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
    
    # Pairwise distance terms: |c_i - c_j| / 2
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    
    all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
    
    # Soft-min calculation
    # To avoid overflow/underflow, shift by max
    max_term = np.max(all_terms)
    exp_terms = np.exp(-beta * (all_terms - max_term))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 1e10
    
    soft_min_val = max_term - np.log(sum_exp) / beta
    
    return -soft_min_val

def nudge_active_constraints(centers, r_target, max_iter=500, tol=1e-12):
    """
    Coordinate descent that specifically nudges points away from active constraints.
    Active constraints are: pairs with distance/2 <= r_target + tol, and walls with 0.5-|c| <= r_target + tol.
    """
    centers = centers.copy()
    for _ in range(max_iter):
        # Find all active or near-active constraints
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0
        
        # Boundary distances
        boundary = 0.5 - np.abs(centers)
        
        # Find pairs that are too close
        iu = np.triu_indices(N, k=1)
        pair_too_close = half_dists[iu] < r_target + tol
        
        # Find wall violations
        wall_too_close = boundary < r_target + tol
        
        if not np.any(pair_too_close) and not np.any(wall_too_close):
            break
        
        # Nudge pairs that are too close
        if np.any(pair_too_close):
            i, j = iu[0][pair_too_close], iu[1][pair_too_close]
            # Move them apart slightly
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-15:
                direction = diff / norm
                step = max(0.001, (r_target + tol - half_dists[i, j]) * 0.5)
                centers[i] += direction * step
                centers[j] -= direction * step
        
        # Nudge wall violations
        if np.any(wall_too_close):
            # For each wall violation, push inward
            for idx in range(N):
                for k in range(3):
                    if boundary[idx, k] < r_target + tol:
                        # Push inward
                        if centers[idx, k] > 0:
                            centers[idx, k] -= max(0.001, (r_target + tol - boundary[idx, k]) * 0.5)
                        else:
                            centers[idx, k] += max(0.001, (r_target + tol - boundary[idx, k]) * 0.5)
        
        # Clip to bounds
        centers = np.clip(centers, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return centers

def polish_lbfgs(centers, max_iter=200, ftol=1e-12, time_limit=5.0):
    """
    Polish a configuration using L-BFGS-B on the soft-min objective.
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use a range of beta values for annealing
    betas = [30.0, 80.0, 200.0, 500.0, 1000.0]
    
    current_centers = centers.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj(x):
            return soft_min_objective(x, beta=beta)
        
        try:
            res = minimize(obj, current_centers.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
            
            if res.success or res.fun < obj(current_centers.flatten()):
                new_centers = res.x.reshape(N, 3)
                # Ensure validity
                new_centers = np.clip(new_centers, -0.5 + 1e-12, 0.5 - 1e-12)
                if np.all(np.isfinite(new_centers)):
                    current_centers = new_centers
        except Exception:
            pass
            
    return current_centers

def high_precision_polish(centers, time_limit=10.0):
    """
    Final high-precision polish: alternate between L-BFGS with very high beta
    and coordinate nudging of active constraints.
    """
    start_time = time.time()
    current = centers.copy()
    best_score = compute_score(current)
    best = current.copy()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Phase 1: Very high beta L-BFGS
    betas = [500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > time_limit * 0.6:
            break
        def obj(x):
            return soft_min_objective(x, beta=beta)
        try:
            res = minimize(obj, current.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.fun < obj(current.flatten()):
                current = res.x.reshape(N, 3)
                current = np.clip(current, -0.5 + 1e-12, 0.5 - 1e-12)
        except Exception:
            pass
        
        score = compute_score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
    
    # Phase 2: Nudge active constraints repeatedly
    if time.time() - start_time < time_limit:
        r_target = best_score
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            current = nudge_active_constraints(best, r_target, max_iter=200)
            score = compute_score(current)
            if score > best_score:
                best_score = score
                best = current.copy()
                r_target = best_score
    
    return best, best_score

def local_search_kick(centers, max_kicks=100, kick_size=0.05, time_limit=20.0):
    """
    Local search by randomly kicking subsets of points and re-polishing.
    """
    start_time = time.time()
    current_score = compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    kick_magnitude = kick_size
    
    for i in range(max_kicks):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random subset of points to kick
        # Usually kicking 1-5 points is effective
        n_kick = rng.integers(1, 6)
        indices = rng.choice(N, size=n_kick, replace=False)
        
        # Generate random kick with adaptive magnitude
        kick = rng.normal(0, kick_magnitude, size=(n_kick, 3))
        
        # Apply kick
        new_centers = best_centers.copy()
        new_centers[indices] += kick
        
        # Clip to bounds
        new_centers = np.clip(new_centers, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Polish the kicked configuration
        polished = polish_lbfgs(new_centers, max_iter=100, ftol=1e-12, time_limit=2.0)
        
        new_score = compute_score(polished)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = polished
        else:
            # Decay kick magnitude if no improvement
            kick_magnitude *= 0.93
            
        # Early stopping if we've been stuck for a while
        if i > 30 and kick_magnitude < 0.0005:
            break
            
    return best_centers, best_score

def generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Jittered Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Sort by distance from center to prioritize central points?
    dists_to_center = np.sum(grid**2, axis=1)
    sorted_indices = np.argsort(dists_to_center)
    grid_50 = grid[sorted_indices[:N]]
    
    # Add jitter
    rng = np.random.default_rng(1)
    jittered_grid = grid_50 + rng.uniform(-0.02, 0.02, size=(N, 3))
    inits.append(np.clip(jittered_grid, -0.49, 0.49))
    
    # 2. Halton Sequence
    def halton_sequence(n, base):
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                x += f * (i_val % base)
                i_val //= base
            result.append(x)
        return np.array(result)
    
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    halton_pts = np.column_stack((h1, h2, h3)) * 0.9 - 0.45
    inits.append(np.clip(halton_pts, -0.49, 0.49))
    
    # 3. Random
    rng2 = np.random.default_rng(2)
    inits.append(rng2.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 4. Fibonacci Sphere scaled to cube
    rng3 = np.random.default_rng(3)
    golden_ratio = (1 + np.sqrt(5)) / 2
    fib_pts = []
    for i in range(N):
        t = i / N
        z = 1 - 2 * t
        radius = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        # Scale to fit in cube
        pt = np.array([x, y, z]) * 0.45
        pt += rng3.normal(0, 0.01, size=3)
        fib_pts.append(pt)
    fib_pts = np.array(fib_pts)
    inits.append(np.clip(fib_pts, -0.49, 0.49))
    
    # 5. Another Grid with different jitter
    rng4 = np.random.default_rng(4)
    jittered_grid2 = grid_50 + rng4.uniform(-0.05, 0.05, size=(N, 3))
    inits.append(np.clip(jittered_grid2, -0.49, 0.49))
    
    # 6. Body-centered cubic-like: 4x4x4 grid but offset some layers
    rng5 = np.random.default_rng(5)
    bcc_grid = grid_50.copy()
    # Offset every other layer in z
    for idx in range(N):
        z_coord = bcc_grid[idx, 2]
        if abs(z_coord - ticks[1]) < 0.01 or abs(z_coord - ticks[3]) < 0.01:
            bcc_grid[idx, 0] += 0.0625
            bcc_grid[idx, 1] += 0.0625
    bcc_grid = bcc_grid + rng5.uniform(-0.01, 0.01, size=(N, 3))
    inits.append(np.clip(bcc_grid, -0.49, 0.49))
    
    # 7. Face-centered cubic-like
    rng6 = np.random.default_rng(6)
    fcc_grid = grid_50.copy()
    for idx in range(N):
        z_coord = fcc_grid[idx, 2]
        if abs(z_coord - ticks[0]) < 0.01 or abs(z_coord - ticks[2]) < 0.01:
            fcc_grid[idx, 0] += 0.0625
            fcc_grid[idx, 1] -= 0.0625
    fcc_grid = fcc_grid + rng6.uniform(-0.01, 0.01, size=(N, 3))
    inits.append(np.clip(fcc_grid, -0.49, 0.49))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    inits = generate_initializations()
    
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Stage 1: Initial Polish
        polished = polish_lbfgs(init, max_iter=200, ftol=1e-12, time_limit=5.0)
        score = compute_score(polished)
        
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
            
        # Stage 2: Local Search with Kicks
        # Allocate remaining time for local search
        time_left = deadline - time.time()
        if time_left > 5:
            # Adjust local search time based on how much time we have
            ls_time = min(12.0, time_left - 5)
            
            if ls_time > 0:
                new_centers, new_score = local_search_kick(polished, max_kicks=50, kick_size=0.05, time_limit=ls_time)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
    
    # Stage 3: Final high-precision polish on the best configuration found
    if best_centers is not None and time.time() < deadline:
        time_left = deadline - time.time()
        if time_left > 3:
            hp_time = min(15.0, time_left - 2)
            if hp_time > 0:
                final_centers, final_score = high_precision_polish(best_centers, time_limit=hp_time)
                if final_score > best_score:
                    best_score = final_score
                    best_centers = final_centers.copy()
    
    # Final check
    if best_centers is None:
        # Fallback to a simple grid if something went wrong
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
