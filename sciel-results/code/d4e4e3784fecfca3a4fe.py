"""
Strategy: Smooth Surrogate Optimization with Multi-Start from Symmetric Topologies.

Key structural change:
- The min-triangle-area objective is inherently nonsmooth (it's the minimum of many linear/quadratic functions).
- We replace it with a smooth surrogate: soft_min = -beta * log(sum(exp(-beta * areas)))
  As beta -> infinity, this approaches the true minimum.
- We use L-BFGS-B (a gradient-based method) on this smooth objective, which can navigate the landscape more effectively than derivative-free Nelder-Mead.
- We use multi-start: initialize from multiple symmetric configurations (regular 13-gon, 7+6 concentric, 6+6+1, etc.) and run L-BFGS-B from each.
- We anneal beta: start with small beta (smooth, easy to optimize), then increase beta to sharpen toward the true min.
- Final polish: use the best result as seed for a final high-beta L-BFGS-B run.

This is fundamentally different from the current approach which uses:
1. A coarse grid search over symmetric parameters
2. Nelder-Mead (derivative-free) on the raw nonsmooth objective

The new approach uses gradient information through a smooth approximation, enabling more precise navigation of the optimization landscape.
"""

import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    """Exact minimum triangle area."""
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_triangle_area(points, beta):
    """Smooth approximation to min triangle area using log-sum-exp."""
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -beta * log(sum(exp(-beta * areas)))
    # For numerical stability, shift by max
    max_area = np.max(areas)
    shifted = areas - max_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = max_area - (1.0 / beta) * log_sum
    return soft_min

def convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """Exact score: min triangle area / convex hull area."""
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def neg_soft_score(points, beta):
    """Negative of soft score for minimization."""
    pts = points
    soft_min = soft_min_triangle_area(pts, beta)
    a_hull = convex_hull_area(pts)
    if a_hull < 1e-10:
        return 1e10
    return -soft_min / a_hull

def make_initial_points_topology1():
    """Regular 13-gon."""
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def make_initial_points_topology2(r, alpha):
    """7-outer heptagon + 6-inner hexagon."""
    angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles_7), np.sin(angles_7)])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + alpha
    inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    return np.vstack([outer, inner])

def make_initial_points_topology3(r, alpha):
    """6-outer hexagon + 6-inner hexagon + 1 center."""
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles_6), np.sin(angles_6)])
    inner_angles = angles_6 + alpha
    inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([outer, inner, center])

def make_initial_points_topology4(r):
    """5-outer pentagon + 7-inner heptagon."""
    angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer = np.column_stack([np.cos(angles_5), np.sin(angles_5)])
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    return np.vstack([outer, inner])

def make_initial_points_topology5(r, alpha):
    """4-outer square + 9-inner nonagon."""
    angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer = np.column_stack([np.cos(angles_4), np.sin(angles_4)])
    inner_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False) + alpha
    inner = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    return np.vstack([outer, inner])

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s
    
    # --- Phase 1: Generate diverse initial points ---
    initial_configs = []
    
    # Topology 1: Regular 13-gon
    initial_configs.append(make_initial_points_topology1())
    
    # Topology 2: 7+6 concentric, various r and alpha
    for r in [0.3, 0.5, 0.7]:
        for alpha in [0.0, np.pi / 6, np.pi / 3]:
            initial_configs.append(make_initial_points_topology2(r, alpha))
    
    # Topology 3: 6+6+1, various r and alpha
    for r in [0.3, 0.5, 0.7]:
        for alpha in [0.0, np.pi / 12, np.pi / 6]:
            initial_configs.append(make_initial_points_topology3(r, alpha))
    
    # Topology 4: 5+7, various r
    for r in [0.4, 0.6, 0.8]:
        initial_configs.append(make_initial_points_topology4(r))
    
    # Topology 5: 4+9, various r and alpha
    for r in [0.5, 0.7]:
        for alpha in [0.0, np.pi / 9]:
            initial_configs.append(make_initial_points_topology5(r, alpha))
    
    # Add a few random perturbations of the regular 13-gon
    rng = np.random.RandomState(42)
    base_13 = make_initial_points_topology1()
    for _ in range(5):
        perturbed = base_13 + rng.normal(0, 0.1, size=(13, 2))
        initial_configs.append(perturbed)
    
    # --- Phase 2: Multi-start L-BFGS-B with beta annealing ---
    # Beta schedule: start smooth, end sharp
    beta_schedule = [5, 10, 20, 50, 100]
    
    for config_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30.0:
            break
        
        # Evaluate initial config
        update_best(init_pts)
        
        # Run L-BFGS-B with increasing beta
        current_pts = init_pts.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline - 10.0:
                break
            
            x0 = current_pts.flatten()
            
            def objective(x):
                pts = x.reshape(-1, 2)
                return neg_soft_score(pts, beta)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                current_pts = res.x.reshape(-1, 2)
                update_best(current_pts)
            except Exception:
                break
        
        # If we found a good config, do a final high-beta polish
        if best_score > 0.02 and time.time() < deadline - 5.0:
            x0 = best_points.flatten()
            
            def objective_final(x):
                pts = x.reshape(-1, 2)
                return neg_soft_score(pts, 200)
            
            try:
                res = minimize(objective_final, x0, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                pts = res.x.reshape(-1, 2)
                update_best(pts)
            except Exception:
                pass
    
    # --- Phase 3: Final exact-score-based local search from best point ---
    if best_points is not None and time.time() < deadline - 2.0:
        x0 = best_points.flatten()
        
        def objective_exact(x):
            pts = x.reshape(-1, 2)
            return -score(pts)
        
        try:
            res = minimize(objective_exact, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            pts = res.x.reshape(-1, 2)
            update_best(pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
