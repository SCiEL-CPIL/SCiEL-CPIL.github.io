"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 145.0  # Safety margin from 150s limit
_START = time.time()

# Pre-compute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _neg_score(x):
    """Negative score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)

def _generate_structured_inits():
    """
    Generate structured initial configurations based on the hypothesis 
    that optimal Heilbronn points for N=13 form concentric regular polygons.
    Common splits for 13: 12+1, 9+4, 7+6, etc.
    """
    inits = []
    
    # 1. 12-gon + Center
    # This is a very strong candidate for even/odd N near 12.
    angles_12 = 2 * np.pi * np.arange(12) / 12
    r_12 = 1.0
    pts_12 = np.column_stack([r_12 * np.cos(angles_12), r_12 * np.sin(angles_12)])
    pts_center = np.array([[0.0, 0.0]])
    pts_13_1 = np.vstack([pts_12, pts_center])
    inits.append(pts_13_1)
    
    # 2. 9-gon + 4-gon (Square)
    # Two rings.
    angles_9 = 2 * np.pi * np.arange(9) / 9
    r_9 = 1.0
    pts_9 = np.column_stack([r_9 * np.cos(angles_9), r_9 * np.sin(angles_9)])
    
    angles_4 = 2 * np.pi * np.arange(4) / 4 + np.pi/4 # Rotated square
    r_4 = 0.5 # Inner ring
    pts_4 = np.column_stack([r_4 * np.cos(angles_4), r_4 * np.sin(angles_4)])
    
    pts_13_2 = np.vstack([pts_9, pts_4])
    inits.append(pts_13_2)
    
    # 3. 7-gon + 6-gon
    angles_7 = 2 * np.pi * np.arange(7) / 7
    r_7 = 1.0
    pts_7 = np.column_stack([r_7 * np.cos(angles_7), r_7 * np.sin(angles_7)])
    
    angles_6 = 2 * np.pi * np.arange(6) / 6 + np.pi/6
    r_6 = 0.6
    pts_6 = np.column_stack([r_6 * np.cos(angles_6), r_6 * np.sin(angles_6)])
    
    pts_13_3 = np.vstack([pts_7, pts_6])
    inits.append(pts_13_3)
    
    # 4. 13-gon (Regular)
    angles_13 = 2 * np.pi * np.arange(13) / 13
    r_13 = 1.0
    pts_13_4 = np.column_stack([r_13 * np.cos(angles_13), r_13 * np.sin(angles_13)])
    inits.append(pts_13_4)
    
    # 5. Jittered versions to break symmetry slightly if needed
    rng = np.random.RandomState(42)
    for i, base in enumerate(inits[:3]):
        jittered = base + rng.normal(0, 0.05, base.shape)
        inits.append(jittered)
        
    return inits

def construct():
    global _START, _DEADLINE
    
    # 1. Generate structured inits
    inits = _generate_structured_inits()
    
    best_x = None
    best_score = -np.inf
    
    # 2. Evaluate inits and select the best few for polishing
    init_scores = []
    for init in inits:
        s = _score(init)
        init_scores.append((s, init))
        
    # Sort by score descending
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 3 diverse candidates for local optimization
    candidates = [x[1] for x in init_scores[:3]]
    
    # Ensure best_x is set to the best initial
    best_score = init_scores[0][0]
    best_x = init_scores[0][1].flatten()
    
    # 3. Local Optimization
    # Nelder-Mead is robust to the non-differentiability of the min() operator.
    # We run it from the best structured starts.
    
    for x0 in candidates:
        if time.time() > _DEADLINE:
            break
            
        try:
            # Nelder-Mead options:
            # maxiter: high enough to converge
            # xatol/fatol: high precision
            # adaptive: True helps with scaling issues
            res = minimize(
                _neg_score, 
                x0.flatten(), 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-10, 
                    'adaptive': True,
                    'disp': False
                }
            )
            
            candidate_score = -res.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_x = res.x
                
        except Exception:
            continue
            
    # 4. Return result
    # The score is min_area / hull_area.
    # The evaluator expects an array of shape (n, 2).
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
