"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of the minimum radius using soft-min.
    Differentiable and suitable for gradient-based optimization.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    pair_dist = 0.5 * d
    
    all_constraints = np.concatenate([wall, pair_dist.ravel()])
    
    min_val = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - min_val))
    soft_min = min_val + np.log(exp_vals.sum()) / beta
    
    return soft_min

def _neg_soft_min(pts, beta=50.0):
    return -_soft_min_score(pts, beta)

def _make_hex_initial():
    """Create a hexagonal lattice configuration for 50 points."""
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5, 5, 5, 10],
        [8, 8, 8, 8, 8, 8, 6, 4],
        [9, 8, 8, 8, 8, 8, 5, 4],
        [10, 9, 8, 8, 8, 7, 6, 4],
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
            
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 != 1) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _make_random_initial(seed=0):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.45, 0.45, (N, 2))
    return pts

def _vectorized_push_apart(pts, r_target, iters=2000, learning_rate=0.002, deadline=None):
    """
    Vectorized local search that pushes overlapping circles apart.
    More efficient than the loop-based version.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        
        wall = np.minimum(
            np.minimum(best_pts[:, 0] + 0.5, 0.5 - best_pts[:, 0]),
            np.minimum(best_pts[:, 1] + 0.5, 0.5 - best_pts[:, 1])
        )
        
        current_r = min(wall.min(), 0.5 * d.min())
        
        if current_r >= r_target:
            break
        
        force = np.zeros_like(best_pts)
        
        # Wall forces (vectorized)
        d_left = best_pts[:, 0] + 0.5
        d_right = 0.5 - best_pts[:, 0]
        d_bottom = best_pts[:, 1] + 0.5
        d_top = 0.5 - best_pts[:, 1]
        
        # For each point, find nearest wall and push away
        for i in range(N):
            walls = [d_left[i], d_right[i], d_bottom[i], d_top[i]]
            min_d = min(walls)
            if min_d < r_target:
                if min_d == d_left[i]:
                    force[i, 0] += (r_target - min_d)
                elif min_d == d_right[i]:
                    force[i, 0] -= (r_target - min_d)
                elif min_d == d_bottom[i]:
                    force[i, 1] += (r_target - min_d)
                elif min_d == d_top[i]:
                    force[i, 1] -= (r_target - min_d)
        
        # Pair forces (vectorized)
        for i in range(N):
            for j in range(i + 1, N):
                dist = d[i, j]
                if dist < 2 * r_target:
                    vec = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm
                        overlap = 2 * r_target - dist
                        force[i] += overlap * vec
                        force[j] -= overlap * vec
        
        best_pts += learning_rate * force
        best_pts = np.clip(best_pts, -0.5, 0.5)
        
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
    
    return best_pts, best_score

def _anneal_optimize(pts, t0_deadline, n_restarts=8):
    """
    Simulated annealing with L-BFGS-B refinement.
    Perturb, optimize, accept/reject based on score.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    rng = np.random.RandomState(12345)
    
    T = 0.005  # Initial temperature
    T_min = 0.0001
    cooling = 0.9
    
    for restart in range(n_restarts):
        if time.time() > t0_deadline - 30.0:
            break
        
        # Perturbation
        scale = T
        perturbed = best_pts + rng.normal(0, scale, (N, 2))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Optimize with L-BFGS-B on soft-min
        try:
            res = minimize(_neg_soft_min, perturbed.flatten(), method="L-BFGS-B",
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 10})
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            
            # Accept if better, or with probability exp(-(new-best)/T)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
            else:
                delta = best_score - new_score
                if delta < T and rng.random() < np.exp(-delta / max(T, 1e-10)):
                    best_pts = new_pts
        except Exception:
            pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_score

def _nelder_mead_optimize(pts, deadline):
    """Nelder-Mead optimization with time guard."""
    try:
        res = minimize(_neg_score, pts.flatten(), method="Nelder-Mead",
                       options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 2000, 'adaptive': True})
        if time.time() > deadline:
            return pts, _score(pts)
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        return new_pts, new_score
    except Exception:
        return pts, _score(pts)

def _lbfgs_optimize(pts, deadline, beta=50.0):
    """L-BFGS-B optimization on soft-min with time guard."""
    try:
        res = minimize(_neg_soft_min, pts.flatten(), method="L-BFGS-B",
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 15})
        if time.time() > deadline:
            return pts, _score(pts)
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        return new_pts, new_score
    except Exception:
        return pts, _score(pts)

def _refine_single(pts, deadline):
    """
    Full refinement pipeline for a single configuration.
    Combines push-apart, L-BFGS, and Nelder-Mead.
    """
    current_pts = pts.copy()
    current_score = _score(pts)
    
    # Step 1: Push-apart to resolve overlaps
    if time.time() < deadline - 20.0:
        target_r = current_score + 0.001
        pts1, score1 = _vectorized_push_apart(current_pts, target_r, iters=1500, 
                                               learning_rate=0.0015, deadline=deadline - 20.0)
        if score1 > current_score:
            current_pts = pts1
            current_score = score1
    
    # Step 2: L-BFGS-B on soft-min
    if time.time() < deadline - 10.0:
        pts2, score2 = _lbfgs_optimize(current_pts, deadline - 10.0)
        if score2 > current_score:
            current_pts = pts2
            current_score = score2
    
    # Step 3: Nelder-Mead polish
    if time.time() < deadline - 5.0:
        pts3, score3 = _nelder_mead_optimize(current_pts, deadline - 5.0)
        if score3 > current_score:
            current_pts = pts3
            current_score = score3
    
    # Step 4: Final push-apart
    if time.time() < deadline - 2.0:
        target_r = current_score + 0.0005
        pts4, score4 = _vectorized_push_apart(current_pts, target_r, iters=800,
                                               learning_rate=0.0008, deadline=deadline - 2.0)
        if score4 > current_score:
            current_pts = pts4
            current_score = score4
    
    return current_pts, current_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple initial candidates
    candidates = []
    
    # Hexagonal baseline
    hex_pts = _make_hex_initial()
    if hex_pts is not None:
        candidates.append(hex_pts)
    
    # Random restarts with different seeds
    for seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline - 100.0:
            break
        rand_pts = _make_random_initial(seed)
        candidates.append(rand_pts)
    
    if not candidates:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Quick evaluation of all candidates
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((pts, s))
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[1])
    
    # Phase 3: Refine top candidates with full pipeline
    best_pts = scored_candidates[0][0]
    best_score = scored_candidates[0][1]
    
    # Refine the best candidate first
    refined_pts, refined_score = _refine_single(best_pts, deadline - 40.0)
    if refined_score > best_score:
        best_pts = refined_pts
        best_score = refined_score
    
    # If time allows, refine the second-best candidate
    if len(scored_candidates) > 1 and time.time() < deadline - 50.0:
        second_pts, second_score = scored_candidates[1]
        if second_score > best_score - 0.005:  # Close enough to be worth trying
            refined2_pts, refined2_score = _refine_single(second_pts, deadline - 40.0)
            if refined2_score > best_score:
                best_pts = refined2_pts
                best_score = refined2_score
    
    # Phase 4: Simulated annealing on the best configuration
    if time.time() < deadline - 30.0:
        anneal_pts, anneal_score = _anneal_optimize(best_pts, deadline - 25.0, n_restarts=6)
        if anneal_score > best_score:
            best_pts = anneal_pts
            best_score = anneal_score
    
    # Phase 5: Final refinement
    if time.time() < deadline - 10.0:
        final_pts, final_score = _refine_single(best_pts, deadline - 5.0)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    # Ensure valid output
    best_pts = np.asarray(best_pts, dtype=float).reshape(N, 2)
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
