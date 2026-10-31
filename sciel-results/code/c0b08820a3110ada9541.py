import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(f):
    """Compute the linear (full) autoconvolution of f."""
    return np.convolve(f, f, mode='full')


def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)


def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective, vectorized using convolution.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    weights = exp_term / sum_exp
    
    # A_k = (2/t) * weights_k * sign(C_k)
    A = (2.0 / t) * weights * signC
    
    # grad_M_i = sum_k A_k * f_{k-i}
    # This is the convolution of A and f, but we need the full convolution
    # and extract the part corresponding to indices 0..N-1 of f.
    # C_full = f * f. Length 2N-1.
    # A has length 2N-1.
    # conv(A, f) has length 3N-2.
    # We want grad_M_i for i in 0..N-1.
    # grad_M_i = sum_{k} A_k f_{k-i}.
    # Note that f_{k-i} is 0 if k-i < 0 or k-i >= N.
    
    # Using np.convolve(A, f, mode='full')
    # The result R[m] = sum_k A[k] f[m-k].
    # We want sum_k A[k] f[k-i].
    # Let j = k-i => k = j+i.
    # sum_j A[j+i] f[j].
    # This is the cross-correlation or a specific slice of convolution.
    
    # Actually, simpler:
    # grad_M = np.convolve(A, f, mode='full')
    # The index of the result corresponding to f_i is i + (N-1)?
    # Let's check.
    # R[m] = sum_k A[k] f[m-k].
    # We want G_i = sum_k A[k] f[k-i].
    # If we reverse A, let A_rev[k] = A[-1-k].
    # Then conv(A_rev, f)[i + N - 1] = sum_k A_rev[k] f[i + N - 1 - k] = sum_k A[N-1-k] f[i + N - 1 - k].
    # Let j = N-1-k. k = N-1-j.
    # sum_j A[j] f[i + N - 1 - (N-1-j)] = sum_j A[j] f[i+j].
    # This is not quite right.
    
    # Let's stick to the loop if vectorization is tricky, but optimize it.
    # Or use FFT.
    
    # FFT approach:
    # grad_M = ifft(fft(A) * fft(f))[0:N] ?
    # Convolution theorem: ifft(fft(A) * fft(f)) gives linear convolution if padded.
    
    size = len(C) + N - 1
    F_A = np.fft.fft(A, size=size)
    F_f = np.fft.fft(f, size=size)
    conv_result = np.fft.ifft(F_A * F_f).real
    
    # The convolution result conv_result[m] = sum_k A[k] f[m-k].
    # We want G_i = sum_k A[k] f[k-i].
    # Note that f[k-i] is non-zero for k-i in [0, N-1].
    # Let's check the index mapping.
    # If we define the gradient as the derivative of the convolution sum,
    # d/df_i sum_k A_k f_{k-i} = A_{2i} + ...?
    # No, d/df_i C_k = 2 f_{k-i}.
    # So dM/df_i = sum_k w_k sign(C_k) 2 f_{k-i}.
    # This is exactly the convolution of (w_k sign(C_k) 2) with f, evaluated at index i?
    # Let B_k = 2 w_k sign(C_k).
    # G_i = sum_k B_k f_{k-i}.
    # This is the cross-correlation of B and f at lag i?
    # Cross-correlation R_Bf[i] = sum_k B[k] f[k+i].
    # If we reverse B, R_{B_rev, f}[i] = sum_k B_rev[k] f[k+i] = sum_k B[-k] f[k+i].
    # Let j = -k. sum_j B[j] f[j+i] = sum_j B[j] f[i+j].
    # This is not sum_k B_k f_{k-i}.
    
    # Let's just use the loop but it's O(N^2) = 160k.
    # With 30 iterations, that's 5M ops. Very fast in Python/NumPy if vectorized.
    # But the loop is in Python.
    # I will vectorize the loop using matrix multiplication or similar.
    
    # G_i = sum_k B_k f_{k-i}
    # This is a Toeplitz matrix multiplication.
    # We can construct the matrix M where M[i, k] = f_{k-i} (0 if out of bounds).
    # G = M @ B.
    
    # Constructing M is O(N^2).
    # Indices:
    # i from 0 to N-1.
    # k from 0 to 2N-2.
    # f_{k-i} is valid if 0 <= k-i < N.
    
    # We can use np.lib.stride_tricks or just a simple loop if N is small enough.
    # N=400 is small. A Python loop of 400 iterations doing a dot product of length 400 is fine.
    
    B = 2.0 * weights * signC
    grad_M = np.zeros(N)
    for i in range(N):
        # k ranges from i to i+N-1
        k_start = i
        k_end = i + N
        if k_start < len(B) and k_end <= len(B):
            grad_M[i] = np.dot(B[k_start:k_end], f)
        elif k_start < len(B):
            # k_end exceeds B, so f slice is shorter
            valid_len = len(B) - k_start
            grad_M[i] = np.dot(B[k_start:], f[:valid_len])
            
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    return (1.0 / (S * S)) * grad_M - (G_val / (S * S * S)) * 2.0 * S


def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Continuous optimization with smooth objective
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        
        f0 = rng.uniform(-1, 1, N)
        if abs(np.sum(f0)) > 1e-6:
            f0 = f0 / np.sum(f0)
        else:
            f0 = np.ones(N)
        
        ts = [0.1, 0.05, 0.01, 0.005]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 50, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_score:
                    true_score = _objective_vectorized(res.x)
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
                    current_f = res.x
            except Exception:
                continue
    
    return best_f
