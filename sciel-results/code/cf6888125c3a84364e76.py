import numpy as np
from scipy.optimize import minimize
import sys

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 50   # Number of harmonics
    
    # Grids for optimization
    # x grid for f(x) >= 0 constraint
    # Need enough points to resolve the highest frequency in f.
    # f has freq up to M. Period L/M.
    # n_x should be >> M.
    n_x = 2048
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    # t grid for autoconvolution peak
    # g(t) has freq up to 2M. Period L/(2M).
    # n_t should be >> 2M.
    n_t = 4096
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    # psi_0 = sqrt(2)
    # psi_k = 2 * cos(2 * pi * k * x / L) = 2 * cos(4 * pi * k * x)
    
    # A: shape (n_x, M+1)
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    # B: shape (n_t, M+1)
    # g(t) = 2 * (u_0^2 * 1 + sum_{k=1} u_k^2 * cos(2 pi k t / L))
    # We can precompute the cos terms
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    # Smooth max parameter
    eps = 0.05
    
    def objective_and_gradient(u):
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        # Compute smooth max: eps * log(sum(exp(g/eps)))
        # For numerical stability, subtract max(g)
        g_max = np.max(g_vals)
        exp_g = np.exp((g_vals - g_max) / eps)
        sum_exp_g = np.sum(exp_g)
        log_sum_exp = g_max + eps * np.log(sum_exp_g)
        
        # Gradient: 4 * u * (p^T B)
        # p_j = exp(g_j - g_max) / sum_exp_g
        p = exp_g / sum_exp_g
        # p^T B is a vector of length M+1
        # (p^T B)_k = sum_j p_j B[j, k]
        pB = p @ B
        grad = 4.0 * u * pB
        
        return log_sum_exp, grad

    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        # f(x) = A @ u >= 0
        f_vals = A @ u
        return f_vals

    # Initial guess: Uniform function
    # u_0 = 1, others 0 => f(x) = sqrt(2), constant.
    # Energy = 2 * 0.5 = 1. Correct.
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    # Multiple restarts to avoid local minima
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 8
    
    for r in range(n_restarts):
        if r == 0:
            u_start = u0.copy()
        else:
            # Random start on sphere
            # Add some bias towards low frequencies?
            # Just random is fine, SLSQP should handle it
            u_rand = np.random.randn(M + 1)
            # Scale down high frequencies to start smoother?
            # u_rand[k] /= (1 + 0.1*k)
            u_start = u_rand / np.linalg.norm(u_rand)
            
        # Define constraints for SLSQP
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A} # Using A directly
        ]
        
        try:
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            if res.success:
                u_opt = res.x
                # Check energy constraint
                energy = np.dot(u_opt, u_opt)
                if energy > 1e-12:
                    u_opt = u_opt / np.sqrt(energy)
                
                # Check non-negativity
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-6):
                    # Calculate score (actual max, not smooth approx)
                    u_sq = u_opt**2
                    g_vals = 2.0 * (B @ u_sq)
                    score = np.max(g_vals)
                    if score < best_score:
                        best_score = score
                        best_u = u_opt.copy()
                        
        except Exception as e:
            pass
    
    # If optimization failed or didn't improve, use best_u (which might be u0)
    # Evaluate on the final 600 step intervals
    # Intervals are [x_i, x_{i+1}] where x_i = -0.25 + i * (0.5/600)
    # Sample at centers
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    # Reuse A logic but for centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    # Ensure non-negative
    f_centers = np.maximum(0, f_centers)
    
    # The "normalized maximum" score returned by the evaluator is likely 
    # max_autoconv / ||f||_2^2
    # Our optimization minimized max_autoconv subject to ||f||_2^2 = 1.
    # So the score is best_score.
    # But we just return the heights.
    
    return list(f_centers)

if __name__ == "__main__":
    # Quick self-check
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
