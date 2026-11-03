"""
Frequency-domain optimization for minimizing the normalized peak autoconvolution.

Strategy:
- Represent the step function by its first M Fourier cosine coefficients (even, periodic).
- The autoconvolution of the periodic extension has peak determined by the squared magnitudes.
- Optimize the coefficient vector u on the unit sphere to minimize max_t g(t), where
  g(t) = (1/L) * [u_0^2 + sum_{k=1}^M u_k^2 cos(2*pi*k*t/L)].
- Enforce f(x) >= 0 via penalty method with adaptive penalty weight.
- Use L-BFGS-B on the sphere parameterization with a trust-region update.

The key insight: minimizing max g(t) on the sphere is equivalent to finding the
distribution of energy across harmonics that minimizes the worst-case interference.
"""

import numpy as np
from scipy.optimize import minimize

N = 600
M = 15
L = 0.5
N_GRID = 4096
N_T = 8192

def construct():
    # Precompute grids and basis matrices
    x = np.linspace(-L / 2, L / 2, N_GRID, endpoint=False)
    t = np.linspace(-L, L, N_T, endpoint=False)
    
    # Basis for f: f(x) = sum u_k * phi_k(x)
    # phi_0 = 1/sqrt(L), phi_k = sqrt(2/L) * cos(2*pi*k*x/L)
    # Matrix A: A[j, k] = phi_k(x_j)
    A = np.zeros((N_GRID, M + 1))
    A[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M + 1):
        A[:, k] = np.sqrt(2.0 / L) * np.cos(2 * np.pi * k * x / L)
    
    # Matrix for g: g(t) = (1/L) * [u_0^2 + sum u_k^2 cos(2*pi*k*t/L)]
    # B[j, k] = cos(2*pi*k*t_j/L) for k>=1, B[j,0] = 1
    B = np.zeros((N_T, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2 * np.pi * k * t / L)
    
    def compute_f(u):
        return A @ u
    
    def compute_g(u):
        # g(t) = (1/L) * (u_0^2 + sum u_k^2 * cos(...))
        # = (1/L) * (u[0]^2 + u[1:]^T @ B[:,1:])
        u_sq = u ** 2
        g_vals = (u_sq[0] + u_sq[1:] @ B[:, 1:]) / L
        return g_vals
    
    def objective_penalty(u, penalty_weight):
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        max_g = np.max(g_vals)
        neg_penalty = penalty_weight * np.sum(np.maximum(0, -f_vals) ** 2)
        return max_g + neg_penalty
    
    def gradient_penalty(u, penalty_weight):
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        idx_max = np.argmax(g_vals)
        t_max = t[idx_max]
        
        # Gradient of max_g w.r.t u:
        # d/d u_k [ (1/L) * (u_0^2 + sum u_k^2 cos(2*pi*k*t_max/L)) ]
        grad = np.zeros(M + 1)
        grad[0] = 2 * u[0] / L
        for k in range(1, M + 1):
            grad[k] = 2 * u[k] / L * np.cos(2 * np.pi * k * t_max / L)
        
        # Gradient of penalty
        neg_f = np.maximum(0, -f_vals)
        if np.any(neg_f > 0):
            # d/d u [ sum (max(0,-f_j))^2 ] = -2 * sum neg_f_j * f_j * (A^T)
            # where f_j = A[j] @ u
            grad -= 2 * penalty_weight * (A.T @ (neg_f * f_vals))
        
        return grad
    
    def sphere_constrained_optimize(u_init, n_iter=500, lr=0.02):
        """Optimize on the unit sphere with penalty for negativity."""
        u = u_init.copy()
        u /= np.linalg.norm(u)
        
        best_obj = np.inf
        best_u = u.copy()
        penalty_weight = 10.0
        
        for i in range(n_iter):
            f_vals = compute_f(u)
            g_vals = compute_g(u)
            
            # Adaptive penalty
            min_f = np.min(f_vals)
            if min_f < -0.01:
                penalty_weight = min(penalty_weight * 1.5, 1000.0)
            elif min_f > 0.01:
                penalty_weight = max(penalty_weight * 0.9, 1.0)
            
            obj = objective_penalty(u, penalty_weight)
            if obj < best_obj:
                best_obj = obj
                best_u = u.copy()
            
            grad = gradient_penalty(u, penalty_weight)
            
            # Gradient step
            u_new = u - lr * grad
            
            # Project to sphere
            norm_new = np.linalg.norm(u_new)
            if norm_new > 1e-10:
                u_new /= norm_new
            
            # Adaptive learning rate
            if i % 20 == 0 and i > 0:
                old_obj = objective_penalty(u, penalty_weight)
                new_obj = objective_penalty(u_new, penalty_weight)
                if new_obj > old_obj:
                    lr *= 0.8
                elif i % 100 == 0:
                    lr = min(lr * 1.1, 0.1)
            
            u = u_new
        
        # Final check
        f_final = compute_f(u)
        if np.min(f_final) >= -1e-6:
            g_final = compute_g(u)
            if np.max(g_final) < best_obj:
                best_u = u.copy()
        
        return best_u
    
    # Multiple restarts
    best_u = np.zeros(M + 1)
    best_u[0] = 1.0  # Start with constant function
    best_score = np.inf
    
    # Restart 1: uniform
    u_opt = sphere_constrained_optimize(best_u, n_iter=300, lr=0.03)
    f_check = compute_f(u_opt)
    if np.min(f_check) >= -1e-6:
        g_check = compute_g(u_opt)
        score = np.max(g_check)
        if score < best_score:
            best_score = score
            best_u = u_opt.copy()
    
    # Restart 2-5: random initializations
    np.random.seed(42)
    for restart in range(4):
        u_rand = np.random.randn(M + 1)
        u_rand /= np.linalg.norm(u_rand)
        
        u_opt = sphere_constrained_optimize(u_rand, n_iter=200, lr=0.02)
        f_check = compute_f(u_opt)
        if np.min(f_check) >= -1e-6:
            g_check = compute_g(u_opt)
            score = np.max(g_check)
            if score < best_score:
                best_score = score
                best_u = u_opt.copy()
    
    # Convert to step heights
    interval_width = L / N
    centers = -L / 2 + (np.arange(N) + 0.5) * interval_width
    heights = compute_f(centers)
    heights = np.maximum(0, heights)
    
    return list(heights)
