import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Use a moderate n to balance expressivity and computation
    n = 32
    support = np.arange(n, dtype=np.float64)
    
    # Initialize weights
    weights = np.ones(n, dtype=np.float64) / n
    
    # Precompute the indicator matrix for the condition s_i + s_j < 2*s_l - s_k
    # This is a 4D boolean array, but we can optimize the gradient calculation.
    # Actually, for the gradient, we need:
    # g_k = Sum_{a,b,c: s_a+s_b+s_c < 2s_k} w_a w_b w_c  (k is X4)
    #       + Sum_{b,c,d: s_k+s_b+s_c < 2s_d} w_b w_c w_d (k is X1, by symmetry same form but indices swapped)
    
    # Let's compute the term T_k = Sum_{a,b,c: s_a+s_b+s_c < 2s_k} w_a w_b w_c
    # And U_k = Sum_{b,c,d: s_k+s_b+s_c < 2s_d} w_b w_c w_d
    # Note that U_k is the same as T_k if we swap roles?
    # In T_k, k is the "threshold" variable (X4).
    # In U_k, k is one of the "sum" variables (X1).
    # Due to symmetry of the problem (X1,X2,X3 are symmetric, X4 is distinct),
    # the contribution of w_k to the probability comes from:
    # 1. k being X4: Prob(X1+X2+X3 < 2s_k) * w_k * (product of others) -> Derivative is Prob(X1+X2+X3 < 2s_k)
    # 2. k being X1: Prob(s_k+X2+X3 < 2X4) * w_k * (product of others) -> Derivative is Prob(s_k+X2+X3 < 2X4)
    # 3. k being X2: Same as X1.
    # 4. k being X3: Same as X1.
    # So grad_k = P(X1+X2+X3 < 2s_k) + 3 * P(s_k+X2+X3 < 2X4)
    # where P(...) are calculated with the current weights for the other variables.
    
    # Let's implement a function to compute the gradient.
    def compute_gradient(support, weights):
        n = len(support)
        grad = np.zeros(n, dtype=np.float64)
        
        # Term 1: k is X4
        # Sum over a,b,c such that s_a+s_b+s_c < 2s_k
        # We can iterate k and sum over a,b,c. O(n^4)
        # Or precompute sums of w_a w_b for pairs.
        
        # Let's use a vectorized approach or simple loops if n is small.
        # n=32, n^4 is 1M. 100 iterations is 100M ops. Might be slow in Python loops.
        # Let's use NumPy broadcasting for the indicator.
        
        # s_a + s_b + s_c < 2*s_k
        # Shape: (n, n, n, n) is too big for memory? 32^4 = 1M floats = 8MB. OK.
        
        # s_a, s_b, s_c, s_k
        sa = support[:, None, None, None]
        sb = support[None, :, None, None]
        sc = support[None, None, :, None]
        sk = support[None, None, None, :]
        
        # Condition for term 1 (k is X4): s_a + s_b + s_c < 2*s_k
        cond1 = (sa + sb + sc) < (2 * sk)
        
        # Weights product: w_a * w_b * w_c * w_k
        # grad_k gets contribution from sum over a,b,c of w_a w_b w_c * I(cond1)
        # This is sum over a,b,c of (w_a w_b w_c) * I(cond1[a,b,c,k])
        
        # Let's compute the sum for each k.
        # w_prod_abc = w_a * w_b * w_c
        w_a = weights[:, None, None]
        w_b = weights[None, :, None]
        w_c = weights[None, None, :]
        w_abc = w_a * w_b * w_c # Shape (n, n, n)
        
        # Sum over a,b,c for each k
        # cond1 is (n, n, n, n). Index [a, b, c, k]
        # We want sum_{a,b,c} w_abc[a,b,c] * cond1[a,b,c,k]
        # This is a matrix multiplication or reduction.
        # term1_k = np.sum(w_abc[:, :, :, None] * cond1, axis=(0,1,2))
        
        # To save memory, we can do it in chunks or just use the 1M array.
        term1 = np.sum(w_abc[:, :, :, None] * cond1, axis=(0, 1, 2))
        
        # Term 2: k is X1 (and by symmetry X2, X3)
        # Condition: s_k + s_b + s_c < 2*s_d
        # grad_k gets 3 * Sum_{b,c,d} w_b w_c w_d * I(s_k+s_b+s_c < 2s_d)
        
        # s_k + s_b + s_c < 2*s_d
        # Indices: k, b, c, d
        sk2 = support[:, None, None, None]
        sb2 = support[None, :, None, None]
        sc2 = support[None, None, :, None]
        sd2 = support[None, None, None, :]
        
        cond2 = (sk2 + sb2 + sc2) < (2 * sd2)
        
        # Weights: w_b * w_c * w_d
        w_b = weights[:, None, None] # b
        w_c = weights[None, :, None] # c
        w_d = weights[None, None, :] # d
        w_bcd = w_b * w_c * w_d
        
        # Sum over b,c,d for each k
        # cond2 is [k, b, c, d]
        term2 = np.sum(w_bcd[None, :, :, :] * cond2, axis=(1, 2, 3))
        
        grad = term1 + 3 * term2
        return grad

    # Optimization loop
    eta = 0.05
    best_weights = weights.copy()
    best_score = -np.inf
    
    # Helper to compute score
    def compute_score(support, weights):
        n = len(support)
        sa = support[:, None, None, None]
        sb = support[None, :, None, None]
        sc = support[None, None, :, None]
        sk = support[None, None, None, :]
        cond = (sa + sb + sc) < (2 * sk)
        
        wa = weights[:, None, None, None]
        wb = weights[None, :, None, None]
        wc = weights[None, None, :, None]
        wk = weights[None, None, None, :]
        
        # P = Sum_{a,b,c,k} w_a w_b w_c w_k I(cond)
        return np.sum(wa * wb * wc * wk * cond)

    # Initial score
    current_score = compute_score(support, weights)
    best_score = current_score
    
    for step in range(1000):
        if time.time() > deadline - 10.0:
            break
            
        grad = compute_gradient(support, weights)
        
        # Update
        weights += eta * grad
        
        # Project to simplex
        weights[weights < 0] = 0
        weights /= np.sum(weights)
        
        # Compute new score
        new_score = compute_score(support, weights)
        
        if new_score > best_score:
            best_score = new_score
            best_weights = weights.copy()
            
        # Adaptive learning rate?
        # If score decreased, revert and reduce eta?
        # For simplicity, we keep the best found.
        # We can also try a line search or backtracking.
        # Let's just use a fixed small eta and keep the best.
        
    return support, best_weights
