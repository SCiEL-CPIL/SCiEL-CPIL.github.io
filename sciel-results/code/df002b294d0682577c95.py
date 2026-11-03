import numpy as np

N = 600

def construct():
    L = 0.5
    M = 20  # Number of harmonics
    
    # Precompute basis matrices
    # f(x) = u[0] / sqrt(L) + sum_{k=1}^{M} u[k] * sqrt(2/L) * cos(2*pi*k*x/L)
    # ||f||_2^2 = sum u[k]^2
    # g(t) = (1/L) * (u[0]^2 + sum_{k=1}^{M} u[k]^2 * cos(2*pi*k*t/L))
    
    # Grid for f(x) nonnegativity check
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    k_vals = np.arange(1, M+1)
    
    # Precompute cosine matrices for efficiency
    # cos_x[k, j] = cos(2*pi*k*x[j]/L)
    cos_x = np.cos(2*np.pi * np.outer(k_vals, x) / L)
    
    # Grid for g(t) maximization
    n_t = 4096
    t = np.linspace(-L, L, n_t, endpoint=False)
    # cos_t[k, m] = cos(2*pi*k*t[m]/L)
    cos_t = np.cos(2*np.pi * np.outer(k_vals, t) / L)
    
    def compute_f(u):
        """Compute f(x) values on the grid."""
        # u[0] term + u[1:] @ cos_x
        return u[0] / np.sqrt(L) + np.sqrt(2.0/L) * (cos_x @ u[1:])
    
    def compute_g(u):
        """Compute g(t) values on the grid."""
        # u[0]^2/L + (1/L) * (u[1:]^2 @ cos_t)
        return (u[0]**2 + (u[1:]**2) @ cos_t) / L
    
    def compute_grad_g(u, t_idx):
        """Compute gradient of g(t[t_idx]) w.r.t u."""
        grad = np.zeros(M+1)
        grad[0] = 2 * u[0] / L
        grad[1:] = (2 * u[1:] / L) * cos_t[:, t_idx]
        return grad
    
    def compute_grad_penalty(u, f_vals):
        """Compute gradient of penalty sum(max(0, -f(x))^2)."""
        neg = np.maximum(0, -f_vals)
        if np.all(neg == 0):
            return np.zeros(M+1)
        
        # Only compute gradient where f < 0
        mask = f_vals < 0
        if not np.any(mask):
            return np.zeros(M+1)
        
        # grad of sum((-f)^2) = -2 * sum(f * grad_f) for f<0
        # grad_f[0] = 1/sqrt(L)
        # grad_f[k] = sqrt(2/L) * cos(2*pi*k*x/L)
        
        neg_vals = -f_vals[mask]
        # grad_penalty[0] = -2 * sum(neg_vals * (1/sqrt(L)))
        # grad_penalty[k] = -2 * sum(neg_vals * sqrt(2/L) * cos_k)
        
        grad_pen = np.zeros(M+1)
        grad_pen[0] = -2 * np.sum(neg_vals) / np.sqrt(L)
        # For k=1..M: -2 * sqrt(2/L) * sum(neg_vals * cos_x[k-1, mask])
        cos_mask = cos_x[:, mask]
        grad_pen[1:] = -2 * np.sqrt(2.0/L) * (cos_mask.T @ neg_vals)
        
        return grad_pen
    
    best_score = np.inf
    best_u = np.zeros(M+1)
    best_u[0] = 1.0
    
    # Try uniform first
    f_init = compute_f(best_u)
    if np.all(f_init >= 0):
        g_init = compute_g(best_u)
        best_score = np.max(g_init)
    
    n_restarts = 8
    n_iter = 300
    
    for restart in range(n_restarts):
        if restart == 0:
            u = best_u.copy()
        else:
            # Random initialization with emphasis on low frequencies
            u = np.zeros(M+1)
            u[0] = 1.0
            # Add small random perturbations to higher harmonics
            for k in range(1, M+1):
                u[k] = np.random.randn() * 0.1 / (k + 1)
            u /= np.linalg.norm(u)
        
        lr = 0.02
        penalty_weight = 100.0
        
        for i in range(n_iter):
            f_vals = compute_f(u)
            
            # Check feasibility
            if np.any(f_vals < -1e-4):
                # Compute penalty gradient
                grad_pen = compute_grad_penalty(u, f_vals)
                grad = penalty_weight * grad_pen
            else:
                grad = np.zeros(M+1)
            
            # Compute g and its gradient
            g_vals = compute_g(u)
            idx_max = np.argmax(g_vals)
            grad_g = compute_grad_g(u, idx_max)
            
            # Total gradient
            total_grad = grad_g + grad
            
            # Update
            u_new = u - lr * total_grad
            u_new /= np.linalg.norm(u_new)
            
            # Check if new point is better
            f_new = compute_f(u_new)
            if np.all(f_new >= -1e-6):
                g_new = compute_g(u_new)
                score = np.max(g_new)
                if score < best_score:
                    best_score = score
                    best_u = u_new.copy()
                    u = u_new
                else:
                    # Try smaller step
                    u_half = u - 0.5 * lr * total_grad
                    u_half /= np.linalg.norm(u_half)
                    f_half = compute_f(u_half)
                    if np.all(f_half >= -1e-6):
                        u = u_half
            else:
                # Infeasible, try smaller step
                u_half = u - 0.5 * lr * total_grad
                u_half /= np.linalg.norm(u_half)
                f_half = compute_f(u_half)
                if np.all(f_half >= -1e-6):
                    u = u_half
                else:
                    # Keep u unchanged
                    pass
        
        # Early stopping
        if best_score < 1.5:
            break
    
    # Convert to step heights
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    cos_centers = np.cos(2*np.pi * np.outer(k_vals, centers) / L)
    heights = best_u[0] / np.sqrt(L) + np.sqrt(2.0/L) * (cos_centers @ best_u[1:])
    heights = np.maximum(0, heights)
    
    return list(heights)
