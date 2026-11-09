import numpy as np
import time
import itertools

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Global random search over variable-size supports (2 to 6 points).
    We sample support points from a log-uniform distribution to capture both close
    and far apart points, and weights from a Dirichlet distribution.
    We use a vectorized probability calculator to evaluate many candidates quickly.
    Finally, we perform a local refinement on the best candidate found.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def calc_prob_vectorized(support, weights):
        """
        Compute P[X1 + X2 + X3 < 2*X4] efficiently using broadcasting.
        support: 1D array of shape (n,)
        weights: 1D array of shape (n,)
        """
        # Normalize weights to ensure they sum to 1 (safety)
        total_w = np.sum(weights)
        if total_w <= 1e-9:
            return 0.0
        w = weights / total_w
        
        n = len(support)
        if n == 0:
            return 0.0
            
        # Create meshes for indices
        # We need to sum over i, j, k, l: w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
        # Wait, the formula is:
        # P = sum_{k, l} w_k w_l * (sum_{i, j} w_i w_j * 1[s_i + s_j < 2*s_l - s_k])
        # Note: The prompt says: 
        # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
        # Here X4 corresponds to index k? Or l?
        # Let's re-read carefully:
        # "X1..X4 i.i.d. from mu"
        # "P = Prob[X1 + X2 + X3 < 2*X4]"
        # "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
        # This notation is slightly ambiguous. Usually:
        # X1 ~ i, X2 ~ j, X3 ~ l, X4 ~ k?
        # If X1, X2, X3 are the sum, and X4 is the multiplier.
        # Let X1=s_i, X2=s_j, X3=s_l, X4=s_k.
        # Condition: s_i + s_j + s_l < 2*s_k.
        # Probability: sum_{i,j,l,k} w_i w_j w_l w_k * 1[s_i + s_j + s_l < 2*s_k]
        
        # The prompt's formula:
        # sum over k, l of w_k * w_l * (sum of w_i * w_j ... s_i + s_j < 2*s_l - s_k)
        # This implies the sum is over 4 variables, but grouped.
        # Let's stick to the 4-variable sum which is unambiguous:
        # P = sum_{i,j,l,k} w_i w_j w_l w_k * 1[s_i + s_j + s_l < 2*s_k]
        
        # Vectorization strategy:
        # 1. Compute all pairwise sums S2[i, j] = s_i + s_j. Prob2[i, j] = w_i w_j.
        # 2. For each X3 (index l) and X4 (index k), we need sum_{i,j} Prob2[i,j] * 1[S2[i,j] + s_l < 2*s_k].
        #    This is equivalent to sum_{i,j} Prob2[i,j] * 1[S2[i,j] < 2*s_k - s_l].
        
        # Since n is small (<=6), we can do this with simple loops or small broadcasts.
        # Let's use a loop over k and l, and vectorize the i,j part.
        
        p_total = 0.0
        # Precompute S2 and Prob2
        S2 = support[:, None] + support[None, :]
        Prob2 = w[:, None] * w[None, :]
        
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[k] - support[l]
                # Sum Prob2 where S2 < threshold
                mask = S2 < threshold
                prob_sum = np.sum(Prob2[mask])
                p_total += w[k] * w[l] * prob_sum
                
        return p_total

    def is_valid(support, weights):
        if len(support) != len(weights):
            return False
        if len(support) < 1:
            return False
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            return False
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            return False
        if np.any(support < -1e-9):
            return False
        if np.any(weights < -1e-9):
            return False
        if np.sum(weights) <= 1e-9:
            return False
        return True

    best_score = -np.inf
    best_support = np.array([0.0, 1.0, 10.0])
    best_weights = np.array([0.5, 0.3, 0.2])
    
    # Random Search Phase
    # We will try different sizes n from 2 to 6.
    # To ensure we explore the "large separation" structure which is known to be good,
    # we sample support points from a log-uniform distribution between 1e-2 and 1e4.
    # We enforce strict ordering by sorting.
    
    rng = np.random.default_rng(42)
    
    num_iterations = 20000  # We will run as many as possible within time limit
    
    for _ in range(num_iterations):
        if time.time() > deadline - 10:
            break
            
        n = rng.integers(2, 7)  # 2 to 6 points
        
        # Sample support points
        # Mix of strategies:
        # 1. Log-uniform random
        # 2. Structured: 0, a, b, c...
        
        if rng.random() < 0.5:
            # Log-uniform sampling
            log_s = rng.uniform(np.log(1e-2), np.log(1e4), size=n)
            s = np.sort(np.exp(log_s))
            # Ensure 0 is included sometimes? The optimal often has 0.
            # Let's force 0 for half of these
            if rng.random() < 0.5:
                s[0] = 0.0
        else:
            # Structured sampling: Start with 0, then add random ratios
            s = np.zeros(n)
            current = 1.0
            for i in range(1, n):
                # Next point is current * factor
                # Factor between 1.1 and 10
                factor = rng.uniform(1.1, 10.0)
                current *= factor
                s[i] = current
            # Scale the whole thing randomly
            scale = rng.uniform(0.1, 10.0)
            s = s * scale
            s[0] = 0.0
            
        # Sample weights from Dirichlet
        # Use a parameter alpha that allows for sparsity (some weights near 0)
        # but also uniformity.
        alpha = rng.uniform(0.5, 2.0, size=n)
        w = rng.dirichlet(alpha)
        
        # Evaluate
        p_val = calc_prob_vectorized(s, w)
        
        if p_val > best_score:
            best_score = p_val
            best_support = s.copy()
            best_weights = w.copy()
            
    # Local Refinement Phase
    # Take the best candidate found and refine it using a local search (Nelder-Mead)
    # on the flattened parameters (support + weights).
    # We must handle the constraint that support is sorted and weights sum to 1.
    # To simplify, we will optimize the raw parameters and project them back.
    
    if best_score > -np.inf:
        n_best = len(best_support)
        
        def objective(params):
            # params: [s0, s1, ..., s_{n-1}, w0, w1, ..., w_{n-2}]
            # s_{n-1} is derived? No, let's just optimize all s and n-1 w.
            s_raw = params[:n_best]
            w_raw = params[n_best:-1]
            w_last = 1.0 - np.sum(w_raw)
            
            # Construct support and weights
            # Sort support? Nelder-Mead doesn't handle permutations well.
            # Let's assume the order found in random search is good enough for local refinement.
            # However, if the optimizer swaps them, the probability might change if we don't sort.
            # But the measure is defined by the set of points. The probability is invariant to permutation of support/weights.
            # So we can sort the support and corresponding weights to keep them aligned?
            # No, if we sort support, we must sort weights too.
            # Let's just use the raw values. If the optimizer finds a better configuration with swapped points,
            # the probability value will be the same.
            
            s = s_raw
            w = np.append(w_raw, w_last)
            
            # Check validity
            if np.any(s < -1e-9) or np.any(w < -1e-9) or np.sum(w) <= 1e-9:
                return -np.inf
            
            p = calc_prob_vectorized(s, w)
            return -p
        
        # Initial guess
        init_params = np.concatenate([best_support, best_weights[:-1]])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < -best_score:
                # Extract result
                s_res = res.x[:n_best]
                w_res = np.append(res.x[n_best:-1], 1.0 - np.sum(res.x[n_best:-1]))
                
                # Clean up: ensure non-negative
                s_res = np.maximum(s_res, 0.0)
                w_res = np.maximum(w_res, 0.0)
                
                # Normalize weights
                w_sum = np.sum(w_res)
                if w_sum > 1e-9:
                    w_res = w_res / w_sum
                else:
                    w_res = np.ones(n_best) / n_best
                
                # Re-evaluate to get the true score
                p_res = calc_prob_vectorized(s_res, w_res)
                
                if p_res > best_score:
                    best_score = p_res
                    best_support = s_res
                    best_weights = w_res
                    
        except Exception:
            pass

    # Final Validity Check and Formatting
    if not is_valid(best_support, best_weights):
        # Fallback to a known good distribution
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.5, 0.3, 0.2])

    # Ensure flattened 1D float arrays
    support_out = np.asarray(best_support, dtype=np.float64).flatten()
    weights_out = np.asarray(best_weights, dtype=np.float64).flatten()
    
    return support_out, weights_out

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s)
    print(w)
