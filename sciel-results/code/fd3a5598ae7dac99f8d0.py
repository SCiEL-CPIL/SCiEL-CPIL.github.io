"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import random

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 140.0

_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _bary_to_cart(bary):
    return bary[:, 0:1] * _V0 + bary[:, 1:2] * _V1 + bary[:, 2:3] * _V2

def _project_cartesian(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    return np.column_stack([x, y])

def _min_area(points):
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _generate_grid_points(K):
    points = []
    for i in range(K + 1):
        for j in range(K + 1):
            u = i / K
            v = j / K
            w = 1 - u - v
            if w >= 0:
                bary = np.array([u, v, w])
                cart = _bary_to_cart(bary.reshape(1, 3))[0]
                points.append(cart)
    return np.array(points)

def _select_diverse_points(points, k):
    n = len(points)
    if n <= k:
        return points.copy()
    selected = [np.random.randint(n)]
    remaining = set(range(n)) - {selected[0]}
    for _ in range(k - 1):
        best_idx = None
        best_min_dist = -1
        for i in remaining:
            min_d = float('inf')
            for j in selected:
                d = np.linalg.norm(points[i] - points[j])
                if d < min_d:
                    min_d = d
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return points[selected]

def _simulated_annealing(initial_points, time_limit):
    start_time = time.time()
    best_points = initial_points.copy()
    best_score = _min_area(best_points)
    current_points = initial_points.copy()
    current_score = best_score
    
    T = 1e-3
    step_size = 0.05
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        num_perturbed = np.random.randint(1, 4)
        indices = np.random.choice(N, num_perturbed, replace=False)
        
        new_points = current_points.copy()
        noise = np.random.normal(0, step_size, (num_perturbed, 2))
        new_points[indices] += noise
        new_points = _project_cartesian(new_points)
        
        new_score = _min_area(new_points)
        
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling_rate
        step_size *= 0.999
        if T < 1e-10: T = 1e-10
        if step_size < 1e-6: step_size = 1e-6

    return best_points, best_score

def _polish(points):
    def objective(flat):
        p = flat.reshape(N, 2)
        p = _project_cartesian(p)
        return -_min_area(p)
    res = minimize(objective, points.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
    polished_points = res.x.reshape(N, 2)
    polished_points = _project_cartesian(polished_points)
    return polished_points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    np.random.seed(42)
    random.seed(42)
    
    grid_points = _generate_grid_points(25)
    num_grid = len(grid_points)
    
    best_points = None
    best_score = 0.0
    
    while time.time() < deadline - 10:
        initial_points = _select_diverse_points(grid_points, N)
        
        sa_time_limit = 5.0
        if time.time() > deadline - 15:
            sa_time_limit = 2.0
            
        sa_points, sa_score = _simulated_annealing(initial_points, sa_time_limit)
        
        if time.time() < deadline - 5:
            polished_points = _polish(sa_points)
            polished_score = _min_area(polished_points)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points
        else:
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points

    if best_points is None:
        best_points = grid_points[:11]
        
    return best_points
# EVOLVE-BLOCK-END
