import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for projections
x_indices = np.arange(10)
y_indices = np.arange(10)
X_grid, Y_grid = np.meshgrid(x_indices, y_indices, indexing='xy')

D_vals = (X_grid - Y_grid)
H0_vals = X_grid
H1_vals = (X_grid + Y_grid)
H2_vals = (X_grid + 2 * Y_grid)

D_uniq, D_inv = np.unique(D_vals, return_inverse=True)
H0_uniq, H0_inv = np.unique(H0_vals, return_inverse=True)
H1_uniq, H1_inv = np.unique(H1_vals, return_inverse=True)
H2_uniq, H2_inv = np.unique(H2_vals, return_inverse=True)

D_inv_flat = D_inv.ravel()
H0_inv_flat = H0_inv.ravel()
H1_inv_flat = H1_inv.ravel()
H2_inv_flat = H2_inv.ravel()

D_bins = len(D_uniq)
H0_bins = len(H0_uniq)
H1_bins = len(H1_uniq)
H2_bins = len(H2_uniq)

def get_projection_entropy_and_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    q_nonzero = q[mask]
    if len(q_nonzero) == 0:
        return 0.0, np.zeros_like(P)
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_P = grad_q[inv_flat]
    return H, grad_P

def objective_and_grad(z, beta=50.0):
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    # Smooth soft-max using log-sum-exp
    h_max = np.max(H012)
    exps = np.exp(beta * (H012 - h_max))
    M = h_max + np.log(np.sum(exps)) / beta
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    # Gradient of soft-max: M_grad = sum_i (exp_i / sum_exp) * grad_i
    weights = exps / np.sum(exps)
    M_grad = np.zeros_like(P)
    for i in range(3):
        M_grad += weights[i] * grads_012[i]
    
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def score_P(P):
    P_flat = P.ravel()
    D_val, _ = get_projection_entropy_and_grad(P_flat, D_inv_flat, D_bins)
    H0_val, _ = get_projection_entropy_and_grad(P_flat, H0_inv_flat, H0_bins)
    H1_val, _ = get_projection_entropy_and_grad(P_flat, H1_inv_flat, H1_bins)
    H2_val, _ = get_projection_entropy_and_grad(P_flat, H2_inv_flat, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def objective_and_grad_exact(z):
    """Exact (non-smooth) objective for final polish."""
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    M = np.max(H012)
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    idx_max = np.argmax(H012)
    M_grad = grads_012[idx_max]
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    inits = [np.ones((n, n)) / (n * n)]
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # Heuristic inits: try to spread x-y while concentrating x
    # Concentrate on a diagonal band
    for c in range(-4, 5):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Try 2D Gaussian-like inits
    for cx in [2, 5, 8]:
        for cy in [2, 5, 8]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    p[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 4.0)
            p /= p.sum()
            inits.append(p)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        P_flat = P_init.ravel()
        
        # SLSQP with smooth objective
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
        
        # trust-constr with smooth objective
        if time.time() - start_time > timeout - safety_margin:
            break
        try:
            from scipy.optimize import LinearConstraint
            lc = LinearConstraint(np.ones((1, 100)), 1.0, 1.0)
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='trust-constr',
                bounds=bounds,
                constraints=[lc],
                options={'maxiter': 200, 'gtol': 1e-10, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Local refinement from best using smooth objective
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final high-precision polish using exact objective
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try:
            result = opt.minimize(
                objective_and_grad_exact,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
