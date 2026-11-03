"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _build_prolate_basis(N_basis, W):
    """
    Build prolate spheroidal wavelet basis functions.
    These are eigenfunctions of the time-bandwidth limiting operator.
    
    Parameters:
    -----------
    N_basis : int - number of basis functions to use
    W : float - bandwidth parameter (fraction of Nyquist)
    
    Returns:
    --------
    P : (N, N_basis) array of prolate basis functions
    """
    # Build the time-bandwidth limiting matrix
    # T[k,m] = integral from -W to W of sin(pi*(k-m)*t) / (pi*(k-m)) dt
    # For discrete case, we can use the sinc matrix
    
    # The prolate spheroidal wave functions are eigenvectors of the matrix
    # A = S^T S where S is the sinc matrix
    # S[k,m] = sin(pi*(k-m)*W) / (pi*(k-m)) for k != m, and W for k == m
    
    # Actually, the standard formulation uses:
    # (PSW)[n,m] = sum_{k=-W}^{W} e^{2*pi*i*k*n/N}
    # But let's use the integral form
    
    # For discrete prolate spheroidal wavelets, the matrix is:
    # A[n,m] = sin(pi*(n-m)*W) / (pi*(n-m)) for n != m
    # A[n,n] = W
    
    W_eff = W  # This should be in [0, 0.5]
    
    # Build the A matrix
    n_idx = np.arange(N)
    m_idx = np.arange(N)
    
    # Compute differences
    diff = n_idx[:, None] - m_idx[None, :]
    
    # Sinc kernel
    A = np.zeros((N, N))
    mask = np.abs(diff) > 1e-10
    A[mask] = np.sin(np.pi * W_eff * diff[mask]) / (np.pi * diff[mask])
    A[~mask] = W_eff
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(A)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take the top N_basis eigenvectors
    P = eigenvectors[:, :N_basis]
    
    # Normalize each column to have unit L2 norm
    norms = np.linalg.norm(P, axis=0)
    P = P / norms
    
    return P

def _objective_prolate(coeffs, P, beta=50.0):
    """
    Compute soft-max surrogate objective and gradient in prolate basis.
    
    Parameters:
    -----------
    coeffs : array of shape (N_basis,) - coefficients in prolate basis
    P : array of shape (N, N_basis) - prolate basis matrix
    beta : float - temperature for soft-max
    
    Returns:
    --------
    val : float - objective value
    grad : array of shape (N_basis,) - gradient with respect to coeffs
    """
    # Reconstruct x from coefficients
    x = P @ coeffs
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(coeffs)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(coeffs)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient computation
    # d(val)/dc[j] = (1/s^2) * d(soft_max)/dc[j] - (2*soft_max/s^3) * d(s)/dc[j]
    
    # d(soft_max)/dc[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dc[j]
    # d(ac[n])/dc[j] = sum_i P[i,j] * (x[i+n] + x[i-n]) (with boundary handling)
    
    # Build the gradient weight vector
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Compute d(soft_max)/dx first, then chain rule
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    # Now chain rule: d(soft_max)/dc[j] = sum_i P[i,j] * d_softmax_dx[i]
    d_softmax_dc = P.T @ d_softmax_dx
    
    # d(s)/dc[j] = sum_i P[i,j] = sum of each column of P
    d_s_dc = np.sum(P, axis=0)
    
    grad = (1.0 / (s * s)) * d_softmax_dc - (2.0 * soft_max / (s ** 3)) * d_s_dc
    
    return val, grad

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Build prolate basis with different bandwidths
    # Try different numbers of basis functions and bandwidths
    best_coeffs = None
    best_P = None
    best_bandwidth = None
    
    bandwidths = [0.1, 0.15, 0.2, 0.25, 0.3]
    n_basis_list = [10, 20, 30, 50]
    
    # Build basis for the most promising configuration
    # Start with a moderate choice
    W_init = 0.2
    N_basis_init = 30
    
    try:
        P = _build_prolate_basis(N_basis_init, W_init)
    except Exception:
        # Fallback to standard approach if prolate basis fails
        P = np.eye(N, N)[:N_basis_init].T if N_basis_init <= N else None
        if P is None:
            return best_x
    
    if _check_deadline():
        return best_x
    
    # Initialize coefficients
    # Start with coefficients that give x ≈ ones
    # We want P @ c ≈ ones, so c ≈ P^T @ ones / ||P^T||^2
    ones_vec = np.ones(N)
    c_init = np.linalg.lstsq(P, ones_vec, rcond=None)[0]
    
    # Also try uniform coefficients
    c_uniform = np.ones(N_basis_init) / np.sqrt(N_basis_init)
    
    # Evaluate both initializations
    val1, _ = _objective_prolate(c_init, P, beta=50.0)
    val2, _ = _objective_prolate(c_uniform, P, beta=50.0)
    
    if val1 < val2:
        current_coeffs = c_init.copy()
        current_val = val1
    else:
        current_coeffs = c_uniform.copy()
        current_val = val2
    
    # Check if this is better than all-ones
    x_test = P @ current_coeffs
    val_test = _objective(x_test)
    if val_test < best_val:
        best_val = val_test
        best_x = x_test.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Optimize in prolate basis with varying beta
    betas = [10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if _check_deadline():
            break
        
        def _obj_grad_prolate(c, b=beta, P_local=P):
            val, grad = _objective_prolate(c, P_local, beta=b)
            return val, grad

        try:
            result = minimize(_obj_grad_prolate, current_coeffs, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            
            if result.fun < current_val:
                current_val = result.fun
                current_coeffs = result.x.copy()
                
                # Update best if improved
                x_new = P @ current_coeffs
                val_new = _objective(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Try different bandwidths if time permits
    for W in [0.15, 0.25, 0.3]:
        if _check_deadline():
            break
        
        try:
            P_new = _build_prolate_basis(N_basis_init, W)
        except Exception:
            continue
        
        # Initialize coefficients for new basis
        c_init_new = np.linalg.lstsq(P_new, ones_vec, rcond=None)[0]
        
        # Quick optimization
        def _obj_grad_new(c, b=50.0, P_local=P_new):
            val, grad = _objective_prolate(c, P_local, beta=b)
            return val, grad

        try:
            result = minimize(_obj_grad_new, c_init_new, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-11})
            
            if result.fun < current_val:
                x_new = P_new @ result.x
                val_new = _objective(x_new)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish in full space if needed
    if not _check_deadline():
        def _final_obj(x):
            val, grad = _objective_grad_smooth(x, beta=100.0)
            return val, grad

        try:
            result = minimize(_final_obj, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-13})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x

def _objective_grad_smooth(x, beta=30.0):
    """
    Compute soft-max surrogate objective and its gradient.
    Uses log-sum-exp for numerical stability.
    Objective: soft_max(|ac|) / s^2
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    max_val = np.max(abs_ac)
    
    if max_val < 1e-15:
        return 1e10, np.zeros_like(x)
    
    # Soft-max via log-sum-exp
    shifted = beta * (abs_ac - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
    val = soft_max / (s * s)
    
    # Gradient:
    # val = soft_max / s^2
    # d(val)/dx[j] = (1/s^2) * d(soft_max)/dx[j] - (2*soft_max/s^3) * 1
    
    # d(soft_max)/dx[j] = sum_n weights[n] * sign(ac[n]) * d(ac[n])/dx[j]
    # d(ac[n])/dx[j] = x[j+n] + x[j-n] (with boundary handling)
    
    g = np.zeros(2*N-1)
    g[:len(ac)] = weights * np.sign(ac)
    
    # Compute d(soft_max)/dx efficiently using FFT
    L_fft = 1
    while L_fft < 3*N:
        L_fft *= 2
    
    G = np.fft.rfft(g, n=L_fft)
    Xf = np.fft.rfft(x, n=L_fft)
    
    # term1[j] = sum_n g[n] * x[j+n] (correlation)
    term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
    
    # term2[j] = sum_n g[n] * x[j-n] 
    term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
    
    d_softmax_dx = term1 + term2
    
    grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
    
    return val, grad
# EVOLVE-BLOCK-END
