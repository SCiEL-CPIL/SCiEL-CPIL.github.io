import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Uses L-BFGS-B optimization on log-probabilities with structured initializations.
    """
    n = 10
    N = n * n
    # Deadline: 600s limit. Use 590s to be safe.
    deadline = time.time() + 590
    
    # Pre-compute indices for entropy calculation
    # P is 10x10, P[y, x]. Flatten in row-major order: index = y*n + x
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # 0..18
    idx_x   = xs.astype(np.int32)                      # 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    # Pre-compute binning matrices for speed? 
    # bincount is fast, but matrix multiplication might be faster for gradients.
    # Let's use matrix representation for the pooled probabilities.
    # M_dm1: 19x100, M_x: 10x100, M_d1: 19x100, M_d2: 28x100
    
    def make_bin_matrix(indices, size):
        M = np.zeros((size, N))
        for i in range(N):
            M[indices[i], i] = 1.0
        return M

    M_dm1 = make_bin_matrix(idx_dm1, bin_sizes['dm1'])
    M_x   = make_bin_matrix(idx_x,   bin_sizes['x'])
    M_d1  = make_bin_matrix(idx_d1,  bin_sizes['d1'])
    M_d2  = make_bin_matrix(idx_d2,  bin_sizes['d2'])

    def compute_entropies(P_flat):
        """
        Compute D and M for a given probability vector P_flat.
        Returns (D, M, gradient_D, gradient_M_components) if needed, 
        but for L-BFGS-B we need the gradient of the objective -D/M.
        """
        c_dm1 = M_dm1 @ P_flat
        c_x   = M_x @ P_flat
        c_d1  = M_d1 @ P_flat
        c_d2  = M_d2 @ P_flat
        
        def entropy_and_grad(c):
            # c sums to 1
            mask = c > 1e-15
            p = c[mask]
            if len(p) == 0:
                return 0.0, np.zeros_like(c)
            # H = -sum p log2 p
            # dH/dp_i = -log2(p_i) - 1/ln(2)
            # But we only need it for non-zero p.
            H = -np.sum(p * np.log2(p))
            
            grad_c = np.zeros_like(c)
            # Gradient with respect to counts c
            # dH/dc_i = -log2(c_i) - 1/ln(2) for c_i > 0
            grad_c[mask] = -np.log2(p) - 1.0/np.log(2.0)
            
            # Gradient with respect to P_flat: M.T @ grad_c
            return H, grad_c

        H_dm1, grad_c_dm1 = entropy_and_grad(c_dm1)
        H_x, grad_c_x = entropy_and_grad(c_x)
        H_d1, grad_c_d1 = entropy_and_grad(c_d1)
        H_d2, grad_c_d2 = entropy_and_grad(c_d2)
        
        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        # Gradient of D with respect to P_flat
        grad_D = M_dm1.T @ grad_c_dm1
        
        # Gradient of M with respect to P_flat
        # M = max(H_x, H_d1, H_d2)
        # Assume the max is unique or pick one if tie. For gradient, we can just use the max one.
        # To be robust, we can sum gradients weighted by how much they contribute to max?
        # No, max is not differentiable at ties. But ties are measure zero.
        # We'll pick the index of the max.
        Hs = np.array([H_x, H_d1, H_d2])
        Ms = np.array([grad_c_x, grad_c_d1, grad_c_d2])
        max_idx = np.argmax(Hs)
        grad_M = M_d1.T @ Ms[max_idx] # Wait, need correct matrix
        
        # Correct mapping:
        # 0: H_x -> M_x
        # 1: H_d1 -> M_d1
        # 2: H_d2 -> M_d2
        matrices = [M_x, M_d1, M_d2]
        grad_M = matrices[max_idx].T @ Ms[max_idx]
        
        if M < 1e-9:
            return -np.inf, np.zeros(N)
        
        # Objective: Maximize D/M. Minimize -D/M.
        # f = -D/M
        # df/dP = -(1/M * dD/dP - D/M^2 * dM/dP)
        #        = -1/M * (dD/dP - D/M * dM/dP)
        
        grad_f = -1.0/M * (grad_D - (D/M) * grad_M)
        
        return -D/M, grad_f

    def objective_and_grad(theta):
        # theta is unconstrained log-probabilities
        P_flat = np.exp(theta - np.max(theta))
        P_flat /= P_flat.sum()
        
        val, grad_P = compute_entropies(P_flat)
        
        if not np.isfinite(val):
            return 1e10, np.zeros_like(theta)
            
        # Chain rule: dP_flat/dtheta_j = P_flat[j] * (delta_jk - sum(P_flat))
        # Since sum(P_flat) is normalized to 1, dP_flat/dtheta_j = P_flat[j] * (delta_jk - 1)
        # grad_theta_k = sum_j grad_P_j * dP_j/dtheta_k
        # grad_theta_k = grad_P_k * P_flat_k - P_flat_k * sum_j(grad_P_j * P_flat_j)
        
        dot = np.dot(grad_P, P_flat)
        grad_theta = P_flat * (grad_P - dot)
        
        return val, grad_theta

    # --- Initializations ---
    
    def to_log_prob(P):
        P = P.copy()
        P[P < 1e-15] = 1e-15
        P /= P.sum()
        return np.log(P)

    candidates = []
    
    # 1. Uniform
    P_uniform = np.ones(N) / N
    candidates.append(to_log_prob(P_uniform))
    
    # 2. Diagonal (x=y)
    P_diag = np.zeros(N)
    for i in range(n):
        P_diag[i*n + i] = 1.0 / n
    candidates.append(to_log_prob(P_diag))
    
    # 3. Anti-diagonal (x+y=9)
    P_anti = np.zeros(N)
    for i in range(n):
        P_anti[i*n + (n-1-i)] = 1.0 / n
    candidates.append(to_log_prob(P_anti))
    
    # 4. Row y=0
    P_row = np.zeros(N)
    P_row[0:n] = 1.0 / n
    candidates.append(to_log_prob(P_row))
    
    # 5. Column x=0
    P_col = np.zeros(N)
    P_col[0::n] = 1.0 / n
    candidates.append(to_log_prob(P_col))
    
    # 6. Random smooth
    rng = np.random.default_rng(42)
    for _ in range(10):
        z = rng.standard_normal(N)
        P_rand = np.exp(z)
        P_rand /= P_rand.sum()
        candidates.append(to_log_prob(P_rand))
        
    # 7. Random sparse (top k)
    for _ in range(10):
        P_sparse = np.zeros(N)
        k = rng.integers(5, 15)
        idx = rng.choice(N, size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        P_sparse[idx] = weights
        candidates.append(to_log_prob(P_sparse))

    best_theta = None
    best_val = np.inf
    
    # Optimize from each candidate
    for i, theta0 in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(objective_and_grad, theta0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            val = res.fun
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_theta = res.x
        except Exception:
            pass

    if best_theta is None:
        # Fallback to uniform
        best_theta = to_log_prob(np.ones(N) / N)
        best_val = compute_entropies(np.exp(best_theta - np.max(best_theta)))[0]

    # Convert back to P
    P_final = np.exp(best_theta - np.max(best_theta))
    P_final /= P_final.sum()
    
    # Final sanity check and cleanup
    P_final[P_final < 0] = 0
    P_final = P_final / P_final.sum()
    
    return P_final.reshape((n, n))

if __name__ == "__main__":
    pass
