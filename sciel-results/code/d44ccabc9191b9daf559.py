"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

# Pre-compute triangle indices
TRIS = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIS.append((i, j, k))
TRIS = np.array(TRIS)
NUM_TRIS = len(TRIS)

def _triangle_areas_grad(points):
    """
    Compute areas and gradients for all triangles.
    points: (N, 2)
    Returns: areas (M,), grad (N, 2, M) 
    """
    n = len(points)
    m = NUM_TRIS
    areas = np.zeros(m)
    # grad[i, j, k] will be the gradient of area k w.r.t point i? 
    # No, grad is (N, 2) for each triangle? 
    # Let's return grad as (M, N, 2)
    grads = np.zeros((m, n, 2))
    
    for idx, (i, j, k) in enumerate(TRIS):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Cross product C
        c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        area = 0.5 * abs(c)
        areas[idx] = area
        
        sign = 1.0 if c > 0 else -1.0
        if c == 0:
            sign = 0.0
            
        # Gradient of C
        # dC/dx1 = -(y3 - y2) = y2 - y3
        # dC/dy1 = -(x3 - x2) = x2 - x3
        # dC/dx2 = (y3 - y1)
        # dC/dy2 = -(x3 - x1) = x1 - x3
        # dC/dx3 = -(y2 - y1) = y1 - y2
        # dC/dy3 = (x2 - x1)
        
        g_c = np.zeros((n, 2))
        g_c[i, 0] = p2[1] - p3[1]
        g_c[i, 1] = p2[0] - p3[0]
        
        g_c[j, 0] = p3[1] - p1[1]
        g_c[j, 1] = p1[0] - p3[0]
        
        g_c[k, 0] = p1[1] - p2[1]
        g_c[k, 1] = p2[0] - p1[0]
        
        grads[idx] = 0.5 * sign * g_c
        
    return areas, grads

def _hull_area_grad(points):
    """
    Compute area and gradient of convex hull area.
    Assumes points are in CCW order.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Shoelace
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    
    area = 0.5 * np.sum(x * y_next - x_next * y)
    
    # Gradient
    grad = np.zeros((n, 2))
    grad[:, 0] = 0.5 * (y_next - y_prev)
    grad[:, 1] = 0.5 * (x_prev - x_next)
    
    return abs(area), grad

def _objective_and_grad(p_flat, beta):
    p = p_flat.reshape(N, 2)
    
    # Check for degeneracy
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        return 1e10, np.zeros_like(p_flat)
        
    areas, tri_grads = _triangle_areas_grad(p)
    
    # Soft min weights
    # w_m = exp(-beta * A_m) / sum(exp(-beta * A_k))
    # Shift by max to avoid underflow
    a_max = np.max(areas)
    exp_terms = np.exp(-beta * (areas - a_max))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 1e10, np.zeros_like(p_flat)
        
    weights = exp_terms / sum_exp
    
    # Objective: -log(softmin) + log(hull_area)
    # softmin = a_max - (1/beta) * log(sum_exp)  (using shifted formula)
    # Wait, log(sum(exp(-beta A))) = -beta a_max + log(sum(exp(-beta(A - a_max))))
    # softmin = -1/beta * (-beta a_max + log_sum_shifted) = a_max - (1/beta) log_sum_shifted
    
    log_sum_shifted = np.log(sum_exp)
    softmin_val = a_max - (1.0 / beta) * log_sum_shifted
    
    if softmin_val <= 0:
        return 1e10, np.zeros_like(p_flat)
        
    hull_area, hull_grad = _hull_area_grad(p)
    
    if hull_area < 1e-9:
        return 1e10, np.zeros_like(p_flat)
        
    # Objective value: -log(softmin_val) + log(hull_area)
    obj = -np.log(softmin_val) + np.log(hull_area)
    
    # Gradient
    # d/dP (-log(softmin)) = -1/softmin * d(softmin)/dP
    # d(softmin)/dP = d(a_max)/dP - (1/beta) d(log_sum_shifted)/dP
    # This is tricky because a_max is the max of areas.
    # However, in the soft-min limit, the gradient of the log-sum-exp term dominates.
    # Let's use the gradient of the log-sum-exp term directly.
    # We are minimizing g = (1/beta) log(sum(exp(-beta A))) + log(H)
    # Wait, my objective was -log(softmin) + log(H).
    # softmin = -1/beta log(sum(exp(-beta A)))
    # -log(softmin) = log(-1/beta) - log(log(sum(exp(-beta A)))) ? No.
    # -log( -1/beta * L ) = -log(L) - log(-1/beta) = -log(L) + log(beta)
    # So we minimize -log(L) + log(beta) + log(H) where L = log(sum(exp(-beta A)))
    # This is not quite right. L is a scalar.
    # Let's stick to the chain rule on the smooth approximation.
    
    # Approximation:
    # f(P) = -log(softmin(P)) + log(H(P))
    # softmin(P) = a_max - (1/beta) log_sum_shifted
    # This is only valid if we treat a_max as constant? No.
    
    # Better approach:
    # Minimize J = log(sum(exp(-beta A))) / beta + log(H)
    # This is equivalent to minimizing -log(softmin) + log(H) + log(beta) ?
    # softmin = -1/beta log(sum(exp(-beta A)))
    # -log(softmin) = log(beta) - log(log(sum(exp(-beta A)))) ?? No.
    # -log( -1/beta * S ) = log(beta) - log(S) where S = log(sum(exp(-beta A)))? No.
    # S is the log-sum-exp.
    # softmin = -1/beta * S.
    # -log(softmin) = -log(-1/beta) - log(S) = log(beta) - log(S).
    # So we minimize log(beta) - log(S) + log(H).
    # Gradient: -1/S * dS/dP + 1/H * dH/dP.
    
    # S = log(sum(exp(-beta A)))
    # dS/dP = 1/sum(exp) * sum(exp(-beta A) * (-beta) * dA/dP)
    # dS/dP = -beta * sum( w_m * dA_m/dP ) where w_m = exp(-beta A_m)/sum(exp(-beta A_k))
    # Note: w_m are the same weights calculated above.
    
    grad_soft = -beta * np.sum(weights[:, np.newaxis, np.newaxis] * tri_grads, axis=0)
    
    grad_hull = hull_grad / hull_area
    
    grad_total = -grad_soft + grad_hull  # Wait, objective is -log(S) + log(H)
    # d(-log(S))/dP = -1/S * dS/dP
    # So grad_obj = -1/S * dS/dP + 1/H * dH/dP
    # dS/dP = -beta * sum(w * dA)
    # So -1/S * dS/dP = beta/S * sum(w * dA)
    
    # Let's re-verify:
    # f = -log(S) + log(H)
    # df = -dS/S + dH/H
    # dS = -beta * sum(w dA)
    # df = beta/S * sum(w dA) + dH/H
    
    grad_obj = (beta / sum_exp) * np.sum(weights[:, np.newaxis, np.newaxis] * tri_grads, axis=0) + grad_hull
    
    # Flatten gradient
    return obj, grad_obj.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s safety
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score = -np.inf
    
    # Calculate initial score
    areas = _triangle_areas_grad(points)[0]
    hull_area = _hull_area_grad(points)[0]
    if hull_area > 1e-9:
        best_score = np.min(areas) / hull_area
    
    betas = [5, 10, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        def obj_func(p_flat):
            val, _ = _objective_and_grad(p_flat, beta)
            return val
        
        def grad_func(p_flat):
            _, grad = _objective_and_grad(p_flat, beta)
            return grad
            
        # L-BFGS-B
        try:
            res = minimize(
                obj_func, 
                points.flatten(), 
                jac=grad_func,
                method='L-BFGS-B', 
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success:
                p_opt = res.x.reshape(N, 2)
                # Verify with exact min
                areas_exact = _triangle_areas_grad(p_opt)[0]
                hull_exact = _hull_area_grad(p_opt)[0]
                if hull_exact > 1e-9:
                    score_opt = np.min(areas_exact) / hull_exact
                    if score_opt > best_score:
                        best_score = score_opt
                        best_points = p_opt
                points = p_opt
            else:
                # If optimization fails, keep current points
                pass
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
