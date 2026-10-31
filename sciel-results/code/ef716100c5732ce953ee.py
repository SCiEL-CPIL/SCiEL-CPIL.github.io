import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _basis_matrix(num_modes):
    """
    Construct a matrix where each column is a DCT-II basis vector of length N.
    We use the first 'num_modes' basis vectors.
    Note: Standard DCT-II is orthogonal. We will normalize columns to have unit L2 norm
    to make the coefficients more interpretable, though scaling doesn't strictly matter 
    for the ratio if we normalize h later.
    """
    n = N
    k = num_modes
    # DCT-II definition: X_k = sum_{n=0}^{N-1} x_n * cos(pi/N * (n+0.5) * k)
    # We want the basis vectors b_k(n) = cos(pi/N * (n+0.5) * k)
    n_idx = np.arange(n)
    k_idx = np.arange(k)
    
    # Matrix B of shape (N, k)
    # B[n, k] = cos(pi/N * (n + 0.5) * k)
    arg = np.pi / N * (n_idx[:, None] + 0.5) * k_idx[None, :]
    B = np.cos(arg)
    
    # Normalize columns to unit L2 norm for stability
    norms = np.linalg.norm(B, axis=0)
    norms[norms < 1e-10] = 1.0
    B = B / norms
    
    return B

def _compute_h_from_coeffs(c, B):
    """
    Compute h from coefficients c using basis B.
    h = B @ c
    We enforce non-negativity by clipping, but during optimization we use softplus
    or similar to keep gradient flow. However, for the objective function,
    we must use the actual non-negative h.
    """
    h = B @ c
    # Ensure non-negativity for the ratio calculation
    # Note: The optimization will try to make c such that h is mostly positive.
    # If h has negative parts, the ratio might be ill-defined or low.
    # A better approach for optimization: parameterize h_i = exp(c_i) ? 
    # No, the directive says "linear combination of basis functions".
    # So h = B @ c. We must ensure h >= 0.
    # We will optimize c, but evaluate ratio on h_clipped = np.maximum(h, 0).
    # To guide the optimizer, we add a penalty for negative values in the objective.
    return h

def _neg_ratio_smooth_basis(c, B, beta=100.0):
    """
    Objective function for scipy.optimize.
    Minimizes -ratio.
    c: coefficients for the basis.
    """
    h_raw = B @ c
    h = np.maximum(h_raw, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e6 # Large penalty for zero vector
    
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    
    # Smooth max to handle non-differentiability
    m = np.max(conv)
    if m < 1e-12:
        return 1e6
    smooth_max = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    
    ratio = l2_sq / (l1 * smooth_max)
    
    # Penalty for negative raw values to encourage h_raw >= 0
    neg_penalty = 10.0 * np.sum(np.minimum(h_raw, 0.0)**2)
    
    return -ratio + neg_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try different numbers of basis modes
    # More modes allow more complex shapes, but higher dimensionality.
    # Fewer modes enforce more "flatness".
    num_modes_list = [5, 10, 15, 20]
    
    for num_modes in num_modes_list:
        if time.time() > deadline - 10:
            break
            
        B = _basis_matrix(num_modes)
        
        # Initial guess for coefficients
        # Start with a positive constant vector (DC component dominant)
        # The DC component is index 0.
        # Let's initialize c such that h is roughly uniform.
        # Uniform vector u has u_i = 1/N.
        # Project u onto B: c_init = B.T @ u
        u = np.ones(N) / N
        c_init = B.T @ u
        
        # Add some small random noise to break symmetry
        np.random.seed(42 + num_modes)
        c_init += np.random.randn(num_modes) * 0.1
        
        # Optimization
        try:
            res = minimize(
                _neg_ratio_smooth_basis,
                c_init,
                args=(B,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            c_opt = res.x
            h_opt = np.maximum(B @ c_opt, 0.0)
            
            # Normalize h to ensure it's not trivially small (ratio is scale invariant, but numerical stability)
            # Actually, ratio is scale invariant: (a*h)^2 / (a*h * a*(h*h)) = a^2 h^2 / (a^2 h (h*h)) = same.
            # So we don't need to normalize for the score, but it helps for stability.
            score_opt = _exact_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
        # Try a few random restarts for each num_modes if time permits
        if time.time() < deadline - 15:
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                np.random.seed(100 + num_modes * 10 + restart)
                c_rand = np.random.randn(num_modes)
                # Bias towards positive DC
                c_rand[0] += 5.0 
                
                try:
                    res = minimize(
                        _neg_ratio_smooth_basis,
                        c_rand,
                        args=(B,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    c_opt = res.x
                    h_opt = np.maximum(B @ c_opt, 0.0)
                    score_opt = _exact_ratio(h_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    # Fallback: if the basis approach failed completely, use the seed-like behavior
    # But the seed was sparse. Let's ensure we have a valid non-zero vector.
    if best_score < 0.1:
        # Try a simple Gaussian bump
        x = np.linspace(-0.5, 0.5, N)
        h_gauss = np.exp(-10 * x**2)
        score_gauss = _exact_ratio(h_gauss)
        if score_gauss > best_score:
            best_h = h_gauss
            best_score = score_gauss
            
    # Final check: ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # If still zero (shouldn't happen), return uniform
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N) / N
        best_score = _exact_ratio(best_h)
        
    return best_h.tolist()

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print(f"Score: {_exact_ratio(np.array(h))}")
    print(f"Time taken: {time.time() - time.time()}") # Just to check it runs
