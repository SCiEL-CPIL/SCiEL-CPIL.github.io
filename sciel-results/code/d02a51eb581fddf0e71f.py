"""Combined program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _combined_loss(P_flat, samples, lam=10.0, spectral_weight=0.1, target_dist=None):
    """
    Combined loss: Soft-max covering + pairwise distance variance.
    The spectral/uniformity term is approximated by penalizing deviations 
    from the ideal pairwise distance sqrt(4*pi/N).
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    n = len(Pn)

    if target_dist is None:
        target_dist = math.sqrt(4 * np.pi / n)

    # 1. Soft-max covering loss
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    
    # Use log-sum-exp trick for stability if lam is large, but sum(exp) is fine for lam=10
    # To avoid overflow, we can scale, but N=130 and lam=10 is manageable
    cover_loss = np.sum(np.exp(lam * min_dists))
    
    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom
    
    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    # 2. Spectral/Uniformity loss (Pairwise distance variance)
    # Compute all pairwise distances
    dots_all = Pn @ Pn.T
    np.fill_diagonal(dots_all, 2.0)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    dists_all = np.arccos(dots_all)
    
    iu = np.triu_indices(n, 1)
    pair_dists = dists_all[iu]
    
    spec_loss = np.sum((pair_dists - target_dist) ** 2)
    
    # Gradient for spectral loss
    # d/dP_i sum_{j != i} (dist_ij - target)^2
    # = 2 * sum_{j != i} (dist_ij - target) * d(dist_ij)/dP_i
    # d(dist_ij)/dP_i = - (P_j - (P_i.P_j)P_i) / sqrt(1 - (P_i.P_j)^2)
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        dot_ij = dots_all[i, j]
        dist_ij = dists_all[i, j]
        denom_rep = np.sqrt(max(1e-12, 1.0 - dot_ij ** 2))
        factor = 2 * (dist_ij - target_dist) * (-1.0 / denom_rep)
        
        tangent_i = Pn[j] - dot_ij * Pn[i]
        tangent_j = Pn[i] - dot_ij * Pn[j]
        
        grad_P[i] += spectral_weight * factor * tangent_i
        grad_P[j] += spectral_weight * factor * tangent_j
    
    total_loss = cover_loss + spectral_weight * spec_loss
    
    # Project gradient to tangent space
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    
    grad_flat = (grad_tan / norms).flatten()
    
    return total_loss, grad_flat

def _fast_local_search(P, samples, deadline, max_iters=200):
    """
    Fast local search: Move the point responsible for the worst sample.
    Uses a smaller sample set for speed during iterations.
    """
    n = len(P)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        # Direction to move: toward worst_sample, projected to tangent space
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            # Random perturbation if stuck
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm < 1e-8:
                continue
            grad_dir = rand / rand_norm
        else:
            grad_dir = grad_dir / grad_norm
        
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        # Adaptive step size
        steps = [0.15, 0.10, 0.05, 0.02, 0.01]
        
        for step in steps:
            new_p = p_old + step * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            # Update min_dists and nearest only if we accepted a move
            # To save time, we can skip full update every iteration, but it's safer to update
            _, min_dists, nearest = get_max_dist(P)
        else:
            # If no improvement, try a small random kick to escape local minima
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.03 * rand).reshape(1, 3))[0]
            _, min_dists, nearest = get_max_dist(P)
            
    return P, current_max

def construct():
    start_time = time.time()
    # Safety margin: 10s before the 150s limit
    deadline = start_time + 140.0
    
    best_P = None
    best_score = float('inf')
    
    # Restart parameters
    restarts = [
        (0.03, 42),
        (0.05, 123),
        (0.08, 999),
        (0.0, 777) # Pure Fibonacci
    ]
    
    # Determine budget per restart
    # If we have 140s, and 4 restarts, that's ~35s each.
    # But later restarts might need less if we are already good.
    # Let's allocate 80% of time to the first 2, 20% to the rest.
    total_time_budget = deadline - start_time
    time_per_restart = total_time_budget / len(restarts)
    
    for r_idx, (noise, seed) in enumerate(restarts):
        current_deadline = start_time + (r_idx + 1) * time_per_restart
        # Ensure we don't exceed the absolute deadline
        current_deadline = min(current_deadline, deadline - 5.0)
        
        if time.time() > current_deadline - 2.0:
            break
            
        # 1. Initialize
        P = _fibonacci_sphere(N, seed_noise=noise, seed=seed)
        
        # 2. Optimization Phase
        # Use a moderate number of samples for optimization (10k)
        S_opt = 10000
        samples_opt = _fibonacci_sphere(S_opt, seed=seed+100)
        
        # L-BFGS optimization
        opt_deadline = current_deadline - 10.0 # Reserve 10s for local search
        
        def time_callback(xk):
            if time.time() > opt_deadline:
                raise ValueError("Time limit reached")
        
        try:
            res = minimize(
                _combined_loss,
                P.flatten(),
                args=(samples_opt,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'callback': time_callback, 'maxfun': 2000}
            )
            P = res.x.reshape(N, 3)
            P = _normalize(P)
        except Exception:
            pass
        
        # 3. Local Search Polish
        # Use a larger sample set for the final polish to be more accurate
        S_polish = 50000
        samples_polish = _fibonacci_sphere(S_polish, seed=seed+200)
        
        P, current_max = _fast_local_search(P, samples_polish, current_deadline)
        
        # 4. Evaluate
        # Use a very large sample set for the final score estimation
        S_eval = 100000
        samples_eval = _fibonacci_sphere(S_eval, seed=seed+300)
        
        dots = samples_eval @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        score = np.degrees(np.max(min_dists))
        
        if score < best_score:
            best_score = score
            best_P = P.copy()
            
    # If we didn't find anything, use the first Fibonacci spiral
    if best_P is None:
        best_P = _fibonacci_sphere(N, seed_noise=0.03, seed=42)
    
    # Final normalization and deduplication
    best_P = _normalize(best_P)
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_P[i] - best_P[j]) < 1e-6:
                best_P[j] += np.random.randn(3) * 1e-4
                best_P[j] = _normalize(best_P[j].reshape(1, 3))[0]
    
    return best_P
# EVOLVE-BLOCK-END
