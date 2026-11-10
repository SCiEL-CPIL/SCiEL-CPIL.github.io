# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin of 40 seconds

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

    best_points = None
    best_score = -np.inf

    n = 16
    d = 2

    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        points = np.array(points, dtype=float)
        if points.shape != (16, 2) or not np.all(np.isfinite(points)):
            return
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate diverse initial candidates based on known topological structures
    candidates = []

    # 1. Regular 4x4 grid (distorted)
    base_grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(base_grid)
    
    # Radial distortion of grid: scale each point by r^alpha
    for alpha in [0.5, 0.8, 1.2, 1.5]:
        r = np.linalg.norm(base_grid, axis=1)
        r_safe = np.where(r == 0, 1e-6, r)
        scaled = base_grid * (r_safe ** (alpha - 1))[:, np.newaxis]
        candidates.append(scaled)

    # 2. Two concentric circles (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8  # Staggered
    
    # Try different radius ratios
    for r_ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
        inner_radius = r_ratio
        outer_radius = 1.0
        inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
        outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
        two_circle = np.vstack([inner_points, outer_points])
        candidates.append(two_circle)

    # 3. Three rings: 2 center, 6 mid, 8 outer
    p0 = np.array([[0.0, 0.0]])
    p1 = np.array([[0.1, 0.0]])
    mid_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    mid_radius = 0.6
    mid_points = np.column_stack([mid_radius * np.cos(mid_angles), mid_radius * np.sin(mid_angles)])
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_radius = 1.0
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    three_ring = np.vstack([p0, p1, mid_points, outer_points])
    candidates.append(three_ring)

    # 4. Triangular lattice subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 5. Random points (for diversity)
    for seed in [1, 2, 3, 42, 123]:
        np.random.seed(seed)
        random_points = np.random.randn(n, d)
        random_points = random_points / np.std(random_points)
        candidates.append(random_points)

    # Evaluate initial scores
    for cand in candidates:
        update_best(cand)

    # Sort candidates by score to prioritize optimization
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Strategy: Direct Nelder-Mead on the true (non-smooth) objective.
    # This avoids the smoothing errors of soft-min/soft-max and can find exact
    # local optima where the active min/max pairs are fixed.
    
    top_k = min(15, len(scored_candidates))
    
    for i in range(top_k):
        if time.time() > deadline:
            break
        
        score, cand = scored_candidates[i]
        x0 = cand.flatten()
        
        # Nelder-Mead on the true objective
        try:
            result = minimize(
                lambda x: -compute_score(x.reshape(16, 2)), 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True}
            )
            
            opt_points = result.x.reshape(16, 2)
            update_best(opt_points)
            
            # Local perturbation: "Pair Flipping"
            # If we have time, try to nudge points to increase the min distance
            # or decrease the max distance by identifying the critical pairs.
            if time.time() < deadline - 5:
                current_score = compute_score(opt_points)
                d = pdist(opt_points)
                d_min = np.min(d)
                d_max = np.max(d)
                
                # Find indices of min and max distances
                min_idx = np.argmin(d)
                max_idx = np.argmax(d)
                
                # Convert flat index to pair indices
                def flat_to_pair(idx, n):
                    # pdist ordering: (0,1), (0,2)...(0,n-1), (1,2)...
                    # i = floor((2*n - 2 - sqrt(1+8*idx))/2)
                    # j = i + 1 + (idx - n*(i+1) + i*(i+1)/2)
                    # Simpler: just iterate
                    count = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if count == idx:
                                return i, j
                            count += 1
                    return -1, -1

                i_min, j_min = flat_to_pair(min_idx, n)
                i_max, j_max = flat_to_pair(max_idx, n)
                
                # Try to move points apart for min pair and together for max pair
                # This is a heuristic local search step
                for step in range(10):
                    if time.time() > deadline:
                        break
                    perturbed = opt_points.copy()
                    # Move min pair apart
                    if i_min != -1:
                        vec = perturbed[i_min] - perturbed[j_min]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            vec = vec / norm
                            perturbed[i_min] += 0.01 * vec
                            perturbed[j_min] -= 0.01 * vec
                    # Move max pair together
                    if i_max != -1:
                        vec = perturbed[i_max] - perturbed[j_max]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-6:
                            vec = vec / norm
                            perturbed[i_max] -= 0.01 * vec
                            perturbed[j_max] += 0.01 * vec
                            
                    update_best(perturbed)
                    
        except Exception:
            pass

    # If we still have time, do a few random restarts with the same strategy
    if time.time() < deadline - 60:
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7 + 13)
            random_points = np.random.randn(n, d)
            random_points = random_points / np.std(random_points)
            
            # Quick Nelder-Mead optimization
            try:
                result = minimize(
                    lambda x: -compute_score(x.reshape(16, 2)), 
                    random_points.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
                )
                opt_points = result.x.reshape(16, 2)
                update_best(opt_points)
            except Exception:
                pass

    # Final check
    if best_points is None:
        # Fallback to the best initial candidate
        best_points = scored_candidates[0][1]

    return best_points

# EVOLVE-BLOCK-END
