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
    Gradient of the smooth objective.
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
    
    # A_k = weights_k * sign(C_k)
    A = weights * signC
    
    # grad_f_i = sum_k A_k * 2 f_{k-i}
    # For each i, k ranges from i to i+N-1
    grad = np.zeros_like(f)
    for i in range(N):
        k_start = i
        k_end = i + N
        if k_start < len(A) and k_end <= len(A):
            grad[i] = 2.0 * np.dot(A[k_start:k_end], f)
        elif k_start < len(A):
            grad[i] = 2.0 * np.dot(A[k_start:], f[:len(A)-k_start])
    
    # G = LSE/t
    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    # dObj/df = (1/S^2) * grad - (G_val / S^4) * 2*S
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S


def _local_search_binary(f, max_iter=2000, time_budget=8.0):
    """
    Local search on binary {-1, 1} sequences.
    Flips one sign at a time, accepting if it improves the objective.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Find the index with maximum |C|
        k_max = np.argmax(np.abs(C))
        
        # Try flipping each position i and see if it helps
        # For efficiency, only try a subset of positions
        # The change in C when flipping f[i] from v to -v is:
        # delta_C[k] = -2 * v * f[k-i] (for valid k-i)
        # This is a vector of length 2N-1
        
        # Try flipping positions that contribute most to C[k_max]
        # C[k_max] = sum_j f[j] * f[k_max - j]
        # The positions that contribute are j and k_max-j
        
        # For a quick heuristic, try flipping the position that has the largest
        # correlation with the current max
        # Let's try a few random positions plus the most impactful ones
        
        candidates = []
        
        # Add random positions
        for _ in range(10):
            candidates.append(np.random.randint(N))
        
        # Add positions that contribute to C[k_max]
        for j in range(N):
            if 0 <= k_max - j < N:
                # This position j contributes f[j]*f[k_max-j] to C[k_max]
                # Flipping it changes C[k_max] by -2*f[j]*f[k_max-j]
                impact = -2.0 * f[j] * f[k_max - j]
                if abs(impact) > 0:
                    candidates.append(j)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        best_improvement = 0
        best_flip_idx = -1
        
        for i in candidates:
            # Compute the change if we flip f[i]
            v = f[i]
            # delta_C[k] = -2 * v * f[k-i] for k where 0 <= k-i < N
            # i.e., k from i to i+N-1
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            
            if k_start >= len(C) or k_end <= 0:
                continue
            
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            # Apply the flip
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S -= 2.0 * v
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return f, score


def construct():
    start_time = time.time()
    
    # Start with all ones (baseline score = 1/N)
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    # Try continuous optimization with smooth objective
    rng = np.random.default_rng(42)
    
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
                    options={'maxiter': 30, 'maxls': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res.fun < best_score:
                    true_score = _objective_vectorized(res.x)
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
                    current_f = res.x
            except Exception:
                continue
    
    # Try binary local search
    for restart in range(8):
        if time.time() > DEADLINE:
            break
        
        f0 = rng.choice([-1.0, 1.0], size=N)
        S0 = np.sum(f0)
        
        if abs(S0) < 5:
            # Try to increase |S| by flipping signs
            for _ in range(N // 2):
                idx = rng.integers(0, N)
                if f0[idx] * S0 < 0:
                    f0[idx] = -f0[idx]
                    S0 = np.sum(f0)
            if abs(S0) < 5:
                continue
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(8.0, remaining_time * 0.8)
        
        if time_budget < 0.5:
            break
        
        f_local, score_local = _local_search_binary(f0, max_iter=3000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    return best_f
