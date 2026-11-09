import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Basic validity check
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 0.0
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return 0.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
        
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
        
    p = weights / w_sum
    
    # Sort support and reorder weights
    sort_idx = np.argsort(support)
    s_sorted = support[sort_idx]
    p_sorted = p[sort_idx]
    
    # For small n, use direct computation
    if n <= 12:
        # Vectorized direct computation using broadcasting
        # S[i,j] = s_i + s_j
        S = s_sorted[:, None] + s_sorted[None, :]
        # W[i,j] = p_i * p_j
        W = p_sorted[:, None] * p_sorted[None, :]
        
        # T[k,l] = 2*s_l - s_k
        T = 2.0 * s_sorted[None, :] - s_sorted[:, None]
        
        # For each (k,l), we need sum_{i,j} W[i,j] * (S[i,j] < T[k,l])
        # This is equivalent to summing W over the region S < T.
        # We can flatten S and W, sort them, and use searchsorted for each T.
        
        S_flat = S.flatten()
        W_flat = W.flatten()
        
        # Sort S_flat and accumulate W
        sort_idx2 = np.argsort(S_flat, kind='mergesort')
        S_sorted2 = S_flat[sort_idx2]
        W_sorted2 = W_flat[sort_idx2]
        W_cumsum = np.cumsum(W_sorted2)
        W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
        
        T_flat = T.flatten()
        
        # Find indices where S < T
        # searchsorted returns the index of the first element >= T
        # So index-1 is the last element < T
        # We want sum of W[0:index]
        indices = np.searchsorted(S_sorted2, T_flat, side='left')
        
        sums_for_kl = W_cumsum_padded[indices]
        sums_for_kl = sums_for_kl.reshape(n, n)
        
        # prob3_k = sum_l p_l * sums_for_kl[k,l]
        # Note: T[k,l] corresponds to s_k, s_l.
        # The sum over i,j is for X1, X2.
        # P = sum_k p_k * sum_l p_l * (sum_{i,j: s_i+s_j < 2s_l-s_k} p_i p_j)
        # My loop structure in the prompt was:
        # sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        # So k is X4? No.
        # Prompt: P = Prob[X1 + X2 + X3 < 2*X4]
        # Formula: sum over k, l of w_k * w_l * (sum ... 2*s_l - s_k)
        # This implies k is X3? and l is X4?
        # Let's check the indices.
        # X1, X2 are i, j.
        # X3 is k.
        # X4 is l.
        # Condition: s_i + s_j + s_k < 2*s_l  => s_i + s_j < 2*s_l - s_k.
        # So for fixed k (X3) and l (X4), we sum over i (X1), j (X2).
        # The term is p_k * p_l * (inner sum).
        # My code:
        # T = 2.0 * s_sorted[None, :] - s_sorted[:, None]
        # T[k,l] = 2*s_l - s_k. Correct.
        # sums_for_kl[k,l] = sum_{i,j} p_i p_j [s_i+s_j < T[k,l]]. Correct.
        # prob3 = sums_for_kl @ p_sorted?
        # sums_for_kl is (n, n). k is row, l is column.
        # We want sum_l p_l * sums_for_kl[k,l]. This is a matrix-vector product.
        # (n, n) @ (n,) -> (n,). The result at index k is sum_l sums_for_kl[k,l] * p_l.
        # This matches prob3_k.
        # Then total = sum_k p_k * prob3_k = dot(p_sorted, prob3).
        
        prob3 = sums_for_kl @ p_sorted
        total_prob = np.dot(p_sorted, prob3)
        return float(total_prob)
    
    # For larger n, the above is still O(n^4) effectively if not careful?
    # The searchsorted is O(n^2 log n^2) for the sort + O(n^2 log n^2) for the searches?
    # Actually, searchsorted on a sorted array is O(log N) per query.
    # We have n^2 queries. So O(n^2 log n^2) = O(n^2 log n).
    # This is much better than O(n^4).
    # So the code above works for large n too.
    
    # However, for very large n, memory might be an issue for S_flat (n^2).
    # n=100 -> 10,000 floats. Fine.
    # n=1000 -> 1,000,000 floats. Fine.
    
    # So we can use the same logic for all n.
    # I'll just return the result from the vectorized block.
    
    # (The code above already returned for n<=12, but the logic is valid for all n.
    # I will remove the if n<=12 branch and use the vectorized one for all n to be consistent.)
    
    # Re-implementing without the branch to ensure correctness for all n
    S = s_sorted[:, None] + s_sorted[None, :]
    W = p_sorted[:, None] * p_sorted[None, :]
    T = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    sort_idx2 = np.argsort(S_flat, kind='mergesort')
    S_sorted2 = S_flat[sort_idx2]
    W_sorted2 = W_flat[sort_idx2]
    W_cumsum = np.cumsum(W_sorted2)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    T_flat = T.flatten()
    indices = np.searchsorted(S_sorted2, T_flat, side='left')
    sums_for_kl = W_cumsum_padded[indices]
    sums_for_kl = sums_for_kl.reshape(n, n)
    
    prob3 = sums_for_kl @ p_sorted
    total_prob = np.dot(p_sorted, prob3)
    
    return float(total_prob)

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def _params_to_measure(params, n):
    """
    Convert unconstrained parameters to valid support and weights.
    params[0:n] -> log(support)
    params[n:2n] -> raw weights (softmax)
    """
    log_s = params[:n]
    raw_w = params[n:]
    
    # Support: ensure non-negative.
    # To allow 0, we can use a transformation that maps (-inf, inf) to (0, inf)
    # but we can't hit 0.
    # However, if the optimum is at 0, the optimizer might push log_s to -inf.
    # We can clip log_s to a minimum value, e.g., -20 (exp(-20) ~ 2e-9).
    log_s = np.maximum(log_s, -20.0)
    support = np.exp(log_s)
    
    # Weights: softmax
    exp_w = np.exp(raw_w - np.max(raw_w))
    weights = exp_w / np.sum(exp_w)
    
    return support, weights

def _objective(params, n):
    support, weights = _params_to_measure(params, n)
    if not _validate(support, weights):
        return 1.0 # Maximize P, so minimize -P. Invalid = high cost.
    prob = _calc_prob_fast(support, weights)
    return -prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    def try_update(support, weights, prob):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Initial guesses
    initializations = [
        ([0.0, 1.0], [0.578, 0.422]),
        ([0.0, 0.5, 1.0], [0.3, 0.4, 0.3]),
        ([0.0, 0.2, 0.8, 2.0], [0.2, 0.3, 0.3, 0.2]),
        ([0.0, 0.1, 0.5, 1.5, 4.0], [0.1, 0.2, 0.3, 0.25, 0.15]),
    ]
    
    for s_init, w_init in initializations:
        s = np.array(s_init, dtype=np.float64)
        w = np.array(w_init, dtype=np.float64)
        if _validate(s, w):
            prob = _calc_prob_fast(s, w)
            try_update(s, w, prob)
    
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
        best_prob = _calc_prob_fast(best_support, best_weights)
    
    # Search for n in 2..8
    for n in range(2, 9):
        if time.time() > deadline - 30:
            break
            
        # Define bounds for DE
        bounds = []
        for i in range(n):
            # Support log-space: -20 to 10 (s up to 22000)
            bounds.append((-20.0, 10.0))
        for i in range(n):
            # Weights log-space: -10 to 10
            bounds.append((-10.0, 10.0))
        
        def de_obj(p):
            return _objective(p, n)
            
        # Run DE
        try:
            res_de = differential_evolution(
                de_obj, 
                bounds, 
                maxiter=50, 
                popsize=20,
                seed=42 + n,
                tol=1e-8,
                polish=False, # We will polish manually
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            final_params = res_de.x
            support, weights = _params_to_measure(final_params, n)
            
            if _validate(support, weights):
                prob = _calc_prob_fast(support, weights)
                try_update(support, weights, prob)
                
                # Polish from DE result
                if time.time() < deadline - 10:
                    def obj_func(p):
                        return _objective(p, n)
                    res_polish = minimize(
                        obj_func,
                        final_params,
                        method='Nelder-Mead',
                        options={'maxfev': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    p_final = res_polish.x
                    s_final, w_final = _params_to_measure(p_final, n)
                    if _validate(s_final, w_final):
                        prob_final = _calc_prob_fast(s_final, w_final)
                        try_update(s_final, w_final, prob_final)
                        
        except Exception:
            pass
        
        # Also try a few random starts with Nelder-Mead for this n
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            params_rand = np.array([random.uniform(-5, 5) for _ in range(2*n)])
            def obj_func_rand(p):
                return _objective(p, n)
            try:
                res_rand = minimize(
                    obj_func_rand,
                    params_rand,
                    method='Nelder-Mead',
                    options={'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                p_final = res_rand.x
                s_final, w_final = _params_to_measure(p_final, n)
                if _validate(s_final, w_final):
                    prob_final = _calc_prob_fast(s_final, w_final)
                    try_update(s_final, w_final, prob_final)
            except Exception:
                pass

    # Final polish on the global best
    if best_support is not None and time.time() < deadline - 5:
        n_best = len(best_support)
        log_s = np.log(np.maximum(best_support, 1e-20))
        log_w = np.log(best_weights)
        params_best = np.concatenate([log_s, log_w])
        
        def obj_func_best(p):
            return _objective(p, n_best)
        
        try:
            res_final = minimize(
                obj_func_best,
                params_best,
                method='Nelder-Mead',
                options={'maxfev': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            p_final = res_final.x
            s_final, w_final = _params_to_measure(p_final, n_best)
            if _validate(s_final, w_final):
                prob_final = _calc_prob_fast(s_final, w_final)
                try_update(s_final, w_final, prob_final)
        except Exception:
            pass

    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
