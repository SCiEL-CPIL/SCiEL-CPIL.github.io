"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _objective_and_grad(h):
    """
    Compute the objective and its gradient.
    objective = 2*N*max(c_k) / s^2 where c = convolve(h,h), s = sum(h)
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-12:
        s = 1e-12
        s_sign = 1.0
    else:
        s_sign = 1.0 if s > 0 else -1.0
    
    c = np.convolve(h, h)
    max_idx = np.argmax(c)
    c_max = c[max_idx]
    
    # objective = 2*N*c_max / s^2
    obj = 2.0 * N * c_max / (s * s)
    
    # Gradient of c_max with respect to h
    # c_k = sum_i h_i * h_{i+k} for k>=0 (in full convolve indexing)
    # For full convolve of length N with itself, result has length 2N-1
    # c[m] = sum_{i=max(0,m-(N-1))}^{min(N-1,m)} h[i] * h[m-i]
    # 
    # Let's use the correlation formulation. For m >= 0 in standard conv:
    # Actually, let's think about it differently.
    # c[m] where m ranges 0 to 2N-2
    # c[m] = sum_{i=0}^{N-1} h[i] * h[i+m] if i+m < N, else... 
    # 
    # Let me use the standard autoconvolution gradient.
    # For the entry c[m] = sum_i h[i]*h[i+m] (with appropriate bounds)
    # d c[m] / d h[j] = h[j+m] (if j+m < N) + h[j-m] (if j-m >= 0)
    # where we interpret h[i]=0 for i outside [0,N-1]
    
    m = max_idx
    grad_c = np.zeros(N)
    if m >= 0 and m < N:
        # c[m] = sum_{i=0}^{N-1-m} h[i]*h[i+m]
        # d/dh[j] = h[j+m] (if j+m < N) + h[j-m] (if j-m >= 0 and j != ... )
        # Actually for j: contributes when i=j and i+m=j => m=0 (double count)
        # Let's just do it carefully.
        # c[m] = sum_{i: 0<=i<N, 0<=i+m<N} h[i]*h[i+m]
        # d c[m]/d h[j] = h[j+m] * [0<=j<N and 0<=j+m<N] + h[j-m] * [0<=j-m<N and 0<=j<N]
        # But if m=0, the two terms are the same and we get 2*h[j]
        if m == 0:
            grad_c[:] = 2.0 * h
        else:
            # First term: h[j+m] where j+m < N => j < N-m, j >= 0
            if N - m > 0:
                grad_c[:N-m] += h[m:]
            # Second term: h[j-m] where j-m >= 0 => j >= m, j < N
            if m < N:
                grad_c[m:] += h[:N-m]
    else:
        # m >= N, this means the index is in the "tail" part of convolve
        # Actually np.convolve(h,h) for full mode: index m corresponds to
        # c[m] = sum_i h[i] * h[m-i] where valid indices
        # For m >= N: c[m] = sum_{i=m-N+1}^{N-1} h[i] * h[m-i]
        # Let k = m - N + 1, so c[m] = sum_{i=k}^{N-1} h[i]*h[m-i]
        # where m-i ranges from m-k=N-1 down to m-(N-1)=m-N+1=k
        # This is symmetric with the m < N case but flipped.
        # Actually by symmetry c[m] = c[2N-2-m]
        m_sym = 2*N - 2 - m
        if m_sym == 0:
            grad_c[:] = 2.0 * h
        elif m_sym < N:
            if N - m_sym > 0:
                grad_c[:N-m_sym] += h[m_sym:]
            if m_sym < N:
                grad_c[m_sym:] += h[:N-m_sym]
    
    # Now objective = 2*N*c_max / s^2
    # d obj / d h = 2*N * [ (d c_max/dh * s^2 - c_max * 2*s*ds/dh) / s^4 ]
    #             = 2*N * [ grad_c / s^2 - 2*c_max*ds/dh / s^3 ]
    # ds/dh = ones(N)
    ds = np.ones(N)
    grad_obj = 2.0 * N * (grad_c / (s*s) - 2.0 * c_max * ds / (s*s*s))
    
    return obj, grad_obj


def construct():
    """Construct step heights to minimize the autocorrelation value."""
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_val, _ = _objective_and_grad(best_h)
    
    # Try a few different initializations and optimize
    initializations = [
        np.ones(N, dtype=np.float64),
        np.ones(N, dtype=np.float64) + 0.1 * np.random.randn(N),
        np.concatenate([np.ones(200), -np.ones(200)]),
        np.sin(2 * np.pi * np.arange(N) / N),
    ]
    
    for h0 in initializations:
        if time.time() > deadline - 30.0:
            break
        
        try:
            res = minimize(
                lambda h: _objective_and_grad(h)[0],
                h0,
                jac=lambda h: _objective_and_grad(h)[1],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxfun': 20000}
            )
            
            h_opt = res.x
            val_opt, _ = _objective_and_grad(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N, dtype=np.float64)
    if abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N, dtype=np.float64)
    
    return best_h
# EVOLVE-BLOCK-END
