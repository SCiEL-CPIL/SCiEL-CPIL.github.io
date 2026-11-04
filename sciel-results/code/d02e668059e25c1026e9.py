import numpy as np
from scipy.optimize import minimize

def construct():
    N = 600
    L = 0.5
    M = 25
    
    # Grids
    n_x = 512
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    n_t = 1024
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def get_score(u):
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        return np.max(g_vals)
        
    # Baseline: Constant function (DC only)
    u_dc = np.zeros(M + 1)
    u_dc[0] = 1.0
    score_dc = get_score(u_dc)
    
    best_u = u_dc.copy()
    best_score = score_dc
    
    # Try a few random restarts with SLSQP to see if we can beat DC (unlikely)
    n_restarts = 3
    for r in range(n_restarts):
        u_rand = np.random.randn(M + 1)
        u_start = u_rand / np.linalg.norm(u_rand)
        
        def objective(u):
            u_sq = u**2
            g_vals = 2.0 * (B @ u_sq)
            return np.max(g_vals)
            
        def gradient(u):
            u_sq = u**2
            g_vals = 2.0 * (B @ u_sq)
            idx_max = np.argmax(g_vals)
            return 4.0 * u * B[idx_max, :]
            
        def constraint_energy(u):
            return np.dot(u, u) - 1.0
            
        def constraint_energy_grad(u):
            return 2.0 * u
            
        def constraint_nonneg(u):
            return A @ u
            
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A}
        ]
        
        try:
            res = minimize(
                objective,
                u_start,
                method='SLSQP',
                jac=gradient,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-5}
            )
            if res.success:
                u_opt = res.x
                energy = np.dot(u_opt, u_opt)
                if energy > 0:
                    u_opt = u_opt / np.sqrt(energy)
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-4):
                    score_opt = get_score(u_opt)
                    if score_opt < best_score:
                        best_score = score_opt
                        best_u = u_opt.copy()
        except Exception:
            pass
            
    # Evaluate best_u on 600 intervals
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    f_centers = np.maximum(0, f_centers)
    
    return list(f_centers)
