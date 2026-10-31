import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def autoconv_max_abs(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))
    
    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_max_abs(x) / (s * s)
    
    # Strategy: Since the metric is scale-invariant, fix sum=1 and minimize max|autoconv|
    # Use gradient descent with a soft-max surrogate for the max
    
    best_score = np.inf
    best_A = None
    
    # Initialize with various starting points
    np.random.seed(42)
    
    # Try constant sequence first
    A_const = np.ones(N)
    A_const = A_const / np.sum(A_const)  # Normalize sum to 1
    s_const = score(A_const)
    if s_const < best_score:
        best_score = s_const
        best_A = A_const.copy()
    
    # Try random sparse sequences
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Random pattern: mix of +1, -1, and 0
        density = 0.3 + 0.4 * np.random.rand()  # 30%-70% non-zero
        mask = np.random.rand(N) < density
        signs = np.where(np.random.rand(N) < 0.5, 1.0, -1.0)
        A = np.where(mask, signs, 0.0)
        
        # Ensure sum is positive and not too small
        s = np.sum(A)
        if abs(s) < 0.1:
            continue
        
        # Normalize to sum=1
        A = A / s
        
        s_val = score(A)
        if s_val < best_score:
            best_score = s_val
            best_A = A.copy()
    
    # Gradient descent on the best found
    if best_A is not None:
        A = best_A.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute autoconvolution
            ac = fftconvolve(A, A, mode='full')
            abs_ac = np.abs(ac)
            max_idx = np.argmax(abs_ac)
            max_val = abs_ac[max_idx]
            
            if max_val == 0:
                break
            
            # Soft gradient: use softmax weighting over large values
            beta = 10.0
            weights = np.exp(beta * abs_ac / max_val)
            weights /= np.sum(weights)
            
            # Gradient of weighted sum of |ac|
            # d/dc_k sum_j w_j |c_j| where c_j depends on A
            # This is complex, so use finite differences for a few coordinates
            
            # Perturb a few random coordinates
            for _ in range(10):
                i = np.random.randint(N)
                eps = 0.001
                A_plus = A.copy()
                A_plus[i] += eps
                s_plus = np.sum(A_plus)
                if abs(s_plus) < 1e-12:
                    continue
                A_plus = A_plus / s_plus  # Re-normalize
                score_plus = score(A_plus)
                
                A_minus = A.copy()
                A_minus[i] -= eps
                s_minus = np.sum(A_minus)
                if abs(s_minus) < 1e-12:
                    continue
                A_minus = A_minus / s_minus
                score_minus = score(A_minus)
                
                grad = (score_plus - score_minus) / (2 * eps)
                A[i] -= lr * grad
                A[i] = max(-1.0, min(1.0, A[i]))  # Clip
            
            # Re-normalize
            s = np.sum(A)
            if abs(s) < 1e-12:
                break
            A = A / s
            
            s_val = score(A)
            if s_val < best_score:
                best_score = s_val
                best_A = A.copy()
    
    # Return the best found
    if best_A is not None:
        return best_A
    else:
        return np.ones(N)
