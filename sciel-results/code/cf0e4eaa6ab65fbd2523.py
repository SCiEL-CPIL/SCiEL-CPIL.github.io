"""
Strategy: 
1. Coarse search over aspect ratios.
2. For each aspect, use Simulated Annealing to position centers 
   to maximize the minimum pairwise distance (or minimize overlap) 
   with a fixed 'target' radius.
3. Compute Delaunay triangulation of centers.
4. Solve a Linear Program to maximize sum of radii subject to 
   non-overlap constraints on Delaunay edges and boundary constraints.
5. Iterate: adjust positions based on LP results (or just re-run SA with 
   better initial guess from previous LP if time permits).
"""

import numpy as np
import time
import random
from scipy.spatial import Delaunay
from scipy.optimize import linprog
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = -1.0
    best_result = None
    
    # Aspects to try: w/h. Perimeter 4 => w+h=2.
    # w = 2a/(1+a), h = 2/(1+a)
    aspects = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    # Precompute pairs for distance checks if needed, but Delaunay is sparse.
    
    def get_dimensions(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def solve_lp_radii(centers, w, h):
        """
        Solve LP to find max sum radii for fixed centers.
        Variables: r_0, ..., r_{N-1}
        Objective: max sum(r) -> min -sum(r)
        Constraints:
        1. r_i + r_j <= d_ij for Delaunay edges (i,j)
        2. r_i <= x_i (left bound)
        3. r_i <= w - x_i (right bound)
        4. r_i <= y_i (bottom bound)
        5. r_i <= h - y_i (top bound)
        6. r_i >= 0
        """
        # Get Delaunay edges
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                edges.add((simplex[0], simplex[1]))
                edges.add((simplex[1], simplex[2]))
                edges.add((simplex[0], simplex[2]))
            # Also add edges between close non-Delaunay pairs? 
            # For now, just Delaunay is a good approximation for contact graph.
            # To be safe, we can add a few nearest neighbors if they are very close.
            # But Delaunay usually captures the tight constraints.
        except Exception:
            # Fallback: complete graph (slow but safe)
            edges = set()
            for i in range(N):
                for j in range(i+1, N):
                    edges.add((i, j))
        
        edge_list = list(edges)
        M = len(edge_list)
        
        # c: objective coefficients (minimize -sum(r))
        c = -np.ones(N)
        
        # A_ub: inequality constraints A_ub @ r <= b_ub
        # 1. Pairwise: r_i + r_j <= d_ij
        A_pairs = np.zeros((M, N))
        b_pairs = np.zeros(M)
        for k, (i, j) in enumerate(edge_list):
            dist = np.linalg.norm(centers[i] - centers[j])
            A_pairs[k, i] = 1.0
            A_pairs[k, j] = 1.0
            b_pairs[k] = dist
            
        # 2. Boundary constraints:
        # r_i <= x_i
        # r_i <= w - x_i
        # r_i <= y_i
        # r_i <= h - y_i
        A_bounds = np.zeros((4 * N, N))
        b_bounds = np.zeros(4 * N)
        
        for i in range(N):
            # r_i - 0 <= x_i  => 1*r_i <= x_i
            A_bounds[4*i, i] = 1.0
            b_bounds[4*i] = centers[i][0]
            
            # r_i <= w - x_i
            A_bounds[4*i+1, i] = 1.0
            b_bounds[4*i+1] = w - centers[i][0]
            
            # r_i <= y_i
            A_bounds[4*i+2, i] = 1.0
            b_bounds[4*i+2] = centers[i][1]
            
            # r_i <= h - y_i
            A_bounds[4*i+3, i] = 1.0
            b_bounds[4*i+3] = h - centers[i][1]
            
        # Combine
        A_ub = np.vstack([A_pairs, A_bounds])
        b_ub = np.concatenate([b_pairs, b_bounds])
        
        # Bounds: 0 <= r_i <= min_dim/2 (loose upper bound for speed)
        bounds = [(0, None) for _ in range(N)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            return res.x
        else:
            # If LP fails (shouldn't happen if centers are valid), return small radii
            return np.zeros(N)

    def check_validity(circles, w, h):
        """Check if circles are disjoint and inside rectangle."""
        xy = circles[:, :2]
        r = circles[:, 2]
        
        # Boundary
        if np.any(xy[:, 0] < r - 1e-5) or np.any(xy[:, 0] > w - r + 1e-5):
            return False
        if np.any(xy[:, 1] < r - 1e-5) or np.any(xy[:, 1] > h - r + 1e-5):
            return False
            
        # Pairwise
        # Use broadcasting for small N
        dists = np.sqrt(np.sum((xy[:, np.newaxis] - xy[np.newaxis, :])**2, axis=-1))
        r_sums = r[:, np.newaxis] + r[np.newaxis, :]
        # Check off-diagonal
        mask = ~np.eye(N, dtype=bool)
        if np.any(dists[mask] < r_sums[mask] - 1e-5):
            return False
        return True

    def optimize_positions(centers0, w, h, target_r, max_iter=500):
        """
        Use L-BFGS-B to optimize positions to minimize overlap with target_r.
        If no overlap, maximize minimum distance.
        """
        # Objective: sum of squared overlaps + soft boundary penalty
        # Overlap(i,j) = max(0, r_i + r_j - dist_ij)^2
        # Boundary penalty = sum(max(0, r_i - x_i)^2 + ...)
        
        def obj(x):
            centers = x.reshape(-1, 2)
            # Boundary penalty
            pen = 0.0
            for i in range(N):
                cx, cy = centers[i]
                pen += max(0, target_r - cx)**2
                pen += max(0, target_r - (w - cx))**2
                pen += max(0, target_r - cy)**2
                pen += max(0, target_r - (h - cy))**2
            
            # Pairwise overlap
            # Vectorized
            diff = centers[:, np.newaxis] - centers[np.newaxis, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            # Avoid division by zero issues, though dists > 0 usually
            # We only care about dists < 2*target_r
            # Overlap amount = 2*target_r - dists
            # Penalty = sum(overlap^2) for overlap > 0
            overlap = 2.0 * target_r - dists
            overlap = np.where(overlap > 0, overlap, 0)
            # Exclude self
            overlap = overlap[~np.eye(N, dtype=bool)]
            pen += np.sum(overlap**2)
            
            return pen
        
        # Bounds for positions
        # x in [target_r, w-target_r], y in [target_r, h-target_r]
        # If target_r > w/2, this is impossible, but we assume reasonable aspect.
        # For safety, clamp bounds
        r_min = 1e-3
        r_max = min(w, h) / 2.0
        bounds = [(r_min, w - r_min) for _ in range(N)] + [(r_min, h - r_min) for _ in range(N)]
        
        try:
            res = minimize(obj, centers0.flatten(), method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': max_iter, 'ftol': 1e-6})
            return res.x.reshape(-1, 2)
        except Exception:
            return centers0

    # Main Loop
    random.seed(42)
    np.random.seed(42)
    
    # We will try multiple aspects. For each, we do a few restarts.
    # Time budget: ~140s.
    # Each iteration: L-BFGS (fast, ~0.1-0.5s) + LP (very fast, ~0.01s).
    # We can afford ~100-200 iterations.
    
    iterations_per_aspect = 10
    
    for aspect in aspects:
        w, h = get_dimensions(aspect)
        
        # Estimate target radius
        # Area = w*h. N circles. 
        # r_approx = sqrt(w*h / (N * pi)) * 0.9
        r_approx = np.sqrt(w * h / (N * np.pi)) * 0.95
        
        # If aspect is very elongated, grid might be better.
        # Let's generate initial guesses.
        
        for restart in range(iterations_per_aspect):
            if time.time() > deadline:
                break
                
            # Generate initial centers
            if restart == 0:
                # Grid
                side = int(np.ceil(np.sqrt(N)))
                centers = np.zeros((N, 2))
                for i in range(N):
                    row = i // side
                    col = i % side
                    centers[i, 0] = (col + 0.5) * (w / side)
                    centers[i, 1] = (row + 0.5) * (h / side)
            elif restart == 1:
                # Hex-like
                # Rows of circles
                # This is tricky to generate perfectly, so just random with repulsion
                centers = np.random.uniform(r_approx, w-r_approx, (N, 2))
                centers[:, 1] = np.random.uniform(r_approx, h-r_approx, N)
            else:
                # Random
                centers = np.random.uniform(r_approx, w-r_approx, (N, 2))
                centers[:, 1] = np.random.uniform(r_approx, h-r_approx, N)
            
            # Optimize positions
            centers_opt = optimize_positions(centers, w, h, r_approx, max_iter=200)
            
            # Solve LP for radii
            radii = solve_lp_radii(centers_opt, w, h)
            
            # Check validity
            circles = np.column_stack([centers_opt, radii])
            
            if check_validity(circles, w, h):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = circles.copy()
                    
                    # Optional: Refine positions using the new radii?
                    # The LP radii might be larger than r_approx.
                    # We could re-run position optimization with the max radius as target?
                    # Or just accept it.
                    
    if best_result is None:
        # Fallback: Grid of tiny circles
        w, h = 1.0, 1.0
        side = 5
        res = np.zeros((N, 3))
        r = 0.08
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    return best_result
