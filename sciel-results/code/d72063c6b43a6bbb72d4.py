"""Heilbronn triangles: maximize the minimum area of any triangle formed by 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize, basinhopping
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the unit equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

def point_in_triangle(p, v0, v1, v2, eps=1e-10):
    """Check if point p is inside or on the boundary of triangle v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points using vectorization."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Pre-compute indices for all triplets
    # This is constant for N=11
    if not hasattr(min_triangle_area, 'indices'):
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        min_triangle_area.indices = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
        
    i_idx, j_idx, k_idx = min_triangle_area.indices
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """Objective to minimize: negative of the minimum triangle area."""
    points = x.reshape((N, 2))
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for i in range(N):
        p = points[i]
        if not point_in_triangle(p, V0, V1, V2, eps=-1e-6):
            penalty += 1000.0
    
    if penalty > 0:
        return penalty
    
    area = min_triangle_area(points)
    return -area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    # Start with a reasonable initial configuration
    # Use a grid-like arrangement that respects the triangular shape
    points = []
    rows = [5, 3, 3]
    y_positions = [0.1, 0.4, 0.7]
    for r, y in enumerate(y_positions):
        n_in_row = rows[r]
        x_left = y / SQRT3
        x_right = 1 - y / SQRT3
        width = x_right - x_left
        if n_in_row > 0:
            if n_in_row == 1:
                x_positions = [(x_left + x_right) / 2]
            else:
                x_positions = np.linspace(x_left + 0.05 * width, x_right - 0.05 * width, n_in_row)
            for x in x_positions:
                points.append([x, y])
    
    initial_points = np.array(points[:N])
    x0 = initial_points.flatten()
    
    # Basin Hopping
    try:
        bh = basinhopping(
            objective, 
            x0, 
            niter=100, 
            stepsize=0.2, 
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}},
            accept_test=lambda f_new, f_old, x_new, x_old: f_new < f_old,
            seed=42
        )
        
        opt_points = bh.x.reshape((N, 2))
        
        # Verify validity
        valid = True
        for i in range(N):
            if not point_in_triangle(opt_points[i], V0, V1, V2, eps=1e-6):
                valid = False
                break
        
        if valid:
            return opt_points
        else:
            return initial_points
            
    except Exception:
        return initial_points

