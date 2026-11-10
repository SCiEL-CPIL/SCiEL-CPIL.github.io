# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective_exact(points_flat):
        """
        Exact objective for local optimization. Minimize -R.
        """
        points = points_flat.reshape(16, 2)
        score = compute_score(points)
        if not np.isfinite(score):
            return 1e10
        return -score

    def objective_smooth(points_flat, beta=50.0):
        """
        Smooth surrogate using log-sum-exp.
        Normalizes points internally to fix scale gauge.
        """
        points = points_flat.reshape(16, 2)
        
        # Center
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Scale: normalize so that sum of squared distances to origin is 1
        norm = np.linalg.norm(points_centered, axis=1)
        sum_sq = np.sum(norm**2)
        if sum_sq < 1e-12:
            return 1e10
        points_normalized = points_centered / np.sqrt(sum_sq)
        
        d = pdist(points_normalized)
        if d.size == 0 or np.any(d < 1e-12):
            return 1e10
        
        # Soft max of d
        d_max_val = np.max(d)
        log_sum_exp_max = d_max_val + np.log(np.sum(np.exp(beta * (d - d_max_val))))
        d_max_soft = log_sum_exp_max / beta
        
        # Soft min of d
        d_min_val = np.min(d)
        log_sum_exp_min = -beta * d_min_val + np.log(np.sum(np.exp(-beta * (d - d_min_val))))
        d_min_soft = -log_sum_exp_min / beta
        
        if d_min_soft <= 1e-12:
            return 1e10
            
        # Minimize log(d_max) - log(d_min)
        return np.log(d_max_soft) - np.log(d_min_soft)

    n = 16
    d = 2

    # Generate diverse initial candidates
    candidates = []

    # 1. Regular 4x4 grid
    grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 2. Triangular lattice subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 3. Points on a circle (regular 16-gon)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_points)

    # 4. Two concentric circles (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_radius = 0.5
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)

    # 5. Jittered Grid
    np.random.seed(123)
    jittered_grid = grid + np.random.normal(0, 0.1, grid.shape)
    candidates.append(jittered_grid)

    # 6. Jittered Lattice
    jittered_lattice = candidates[1].copy()
    jittered_lattice += np.random.normal(0, 0.1, jittered_lattice.shape)
    candidates.append(jittered_lattice)

    # 7. Random points
    for seed in [42, 100, 200]:
        np.random.seed(seed)
        random_points = np.random.randn(n, d)
        candidates.append(random_points)

    # Evaluate initial scores
    best_points = None
    best_score = -np.inf

    for cand in candidates:
        score = compute_score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()

    # Sort candidates by score
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: Simulated Annealing on Smooth Objective
    # This is the core of the new strategy: global search using a smooth landscape
    if time.time() < deadline - 180:
        current_points = scored_candidates[0][1].copy()
        current_obj = objective_smooth(current_points.flatten())
        best_sa_points = current_points.copy()
        best_sa_obj = current_obj
        
        # Annealing schedule
        T_start = 0.5
        T_end = 0.001
        num_steps = 5000
        cooling = (T_end / T_start) ** (1.0 / num_steps)
        
        for step in range(num_steps):
            if time.time() > deadline - 180:
                break
            
            T = T_start * (cooling ** step)
            
            # Propose a move: Gaussian perturbation
            # Scale perturbation by temperature to allow larger jumps when hot
            noise_scale = 0.1 * (T + 0.01)
            perturbed = current_points + np.random.normal(0, noise_scale, current_points.shape)
            
            new_obj = objective_smooth(perturbed.flatten())
            
            if not np.isfinite(new_obj):
                continue
                
            # Acceptance criterion
            if new_obj < current_obj or np.random.rand() < np.exp((current_obj - new_obj) / T):
                current_points = perturbed
                current_obj = new_obj
                
                if new_obj < best_sa_obj:
                    best_sa_obj = new_obj
                    best_sa_points = current_points.copy()
        
        # Evaluate the best SA result with exact score
        sa_score = compute_score(best_sa_points)
        if sa_score > best_score:
            best_score = sa_score
            best_points = best_sa_points.copy()
        
        # Add SA result to polish candidates
        scored_candidates.append((sa_score, best_sa_points))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: Polish top candidates with Local Optimization
    # Use BFGS on smooth objective, then Nelder-Mead on exact objective
    polish_candidates = []
    
    # Take top 5 distinct candidates
    for score, cand in scored_candidates[:5]:
        is_dup = False
        for q in polish_candidates:
            if np.allclose(cand, q, atol=1e-4):
                is_dup = True
                break
        if not is_dup:
            polish_candidates.append(cand)
    
    for cand in polish_candidates:
        if time.time() > deadline - 60:
            break
        try:
            # BFGS polish
            result = minimize(objective_smooth, cand.flatten(), method='BFGS',
                             options={'maxiter': 200, 'gtol': 1e-6, 'disp': False})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
            # Nelder-Mead polish on exact objective
            if time.time() < deadline - 30:
                result2 = minimize(objective_exact, opt_points.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'disp': False})
                opt_points2 = result2.x.reshape(16, 2)
                opt_score2 = compute_score(opt_points2)
                
                if opt_score2 > best_score:
                    best_score = opt_score2
                    best_points = opt_points2.copy()
                    
        except Exception as e:
            pass

    # Phase 3: Random Restarts with Local Optimization
    if time.time() < deadline - 30:
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            np.random.seed(attempt * 100 + 55)
            noise = np.random.normal(0, 0.05, (16, 2))
            perturbed = best_points + noise
            try:
                result = minimize(objective_smooth, perturbed.flatten(), method='BFGS',
                                 options={'maxiter': 50, 'gtol': 1e-4, 'disp': False})
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                    if time.time() < deadline - 10:
                        result2 = minimize(objective_exact, opt_points.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10, 'disp': False})
                        opt_points2 = result2.x.reshape(16, 2)
                        opt_score2 = compute_score(opt_points2)
                        if opt_score2 > best_score:
                            best_score = opt_score2
                            best_points = opt_points2.copy()
            except Exception as e:
                pass

    if best_points is None:
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
