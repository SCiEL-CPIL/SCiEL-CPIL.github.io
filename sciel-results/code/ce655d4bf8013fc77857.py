"""Rewritten program: minimize max |autoconv| / (sum x)^2 for signed step functions."""
import numpy as np
from scipy.special import logsumexp
import time

N = 400

def construct():
    """
    Construct a sequence of N signed step heights minimizing
    max_k |autoconv_k| / (sum x)^2.
    
    The evaluator scores as seed/value where value is the raw objective (lower is better).
    We aim to minimize the objective.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, keep safety margin
    
    # Objective: minimize J(x) = max |c(x)| / (sum x)^2
    # Scale invariance: J(alpha x) = J(x).
    # Fix sum(x) = 1. Minimize M(x) = max |c(x)| subject to sum(x) = 1.
    # Use soft-max approximation for max.
    
    # Initialize
    rng = np.random.default_rng(42)
    
    # Try a few initializations
    best_val = np.inf
    best_x = None
    
    def compute_J(x):
        S = np.sum(x)
        if abs(S) < 1e-9:
            return np.inf
        c = np.convolve(x, x, mode='full')
        return np.max(np.abs(c)) / (S * S)
    
    def gradient_soft_max_abs(c, x, tau):
        """
        Compute gradient of soft_max_abs(c) w.r.t x.
        c = x * x.
        """
        abs_c = np.abs(c)
        log_sum = logsumexp(tau * abs_c)
        weights = np.exp(tau * abs_c - log_sum)
        # w_k is d L / d c_k
        w = weights * np.sign(c)
        
        # G = sum_k w_k (d c_k / d x)
        # d c_k / d x_j = x_{k-j} + x_{j-k}
        G = np.zeros(N)
        # Loop over k is efficient enough for N=400
        for k in range(len(c)):
            wk = w[k]
            if abs(wk) < 1e-12:
                continue
            
            # Term 1: x[k-j]
            # Valid if 0 <= k-j < N  => k-N < j <= k
            j_start1 = max(0, k - (N - 1))
            j_end1 = min(N - 1, k)
            if j_start1 <= j_end1:
                js = np.arange(j_start1, j_end1 + 1)
                idx = k - js
                G[js] += wk * x[idx]
                
            # Term 2: x[j-k]
            # Valid if 0 <= j-k < N  => k < j <= k+N
            j_start2 = max(0, k)
            j_end2 = min(N - 1, k + N - 1)
            if j_start2 <= j_end2:
                js = np.arange(j_start2, j_end2 + 1)
                idx = js - k
                G[js] += wk * x[idx]
                
        return G

    # Initializations
    inits = []
    # 1. Random
    for _ in range(5):
        r = rng.standard_normal(N)
        # Ensure sum is not 0
        if abs(np.sum(r)) < 1e-6:
            r[0] += 1.0
        inits.append(r / np.sum(r))
        
    # 2. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    alt[0] += 1.0 # Make sum non-zero
    inits.append(alt / np.sum(alt))

    for x0 in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        x = x0.copy()
        # Annealing tau
        taus = [10, 50, 200, 1000]
        
        for tau in taus:
            # Gradient descent with projection
            step_size = 0.1 / tau # Heuristic
            for _ in range(50):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                c = np.convolve(x, x, mode='full')
                G = gradient_soft_max_abs(c, x, tau)
                
                # Update
                x = x - step_size * G
                
                # Project to sum=1
                x = x + (1.0 - np.sum(x)) / N
                
            # Check time
            if time.time() - start_time > DEADLINE - 5:
                break

        # Local search refinement
        # Try flipping signs of individual elements
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2):
            improved = False
            current_J = compute_J(x)
            
            for i in range(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                x_old = x[i]
                x[i] = -x_old
                
                new_J = compute_J(x)
                
                if new_J < current_J:
                    current_J = new_J
                    improved = True
                else:
                    x[i] = x_old
            
            if not improved:
                break

        val = compute_J(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
