# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = 630.0  # seconds

    # We will try a mixture of two geometric distributions.
    # Component 1: mass at 0 with prob p0, then geometric tail.
    # Actually, let's parameterize more flexibly.
    # Let's use a finite support measure with n points.
    # The inspiration suggests: "Optimizing the mixture parameters (r1, r2, p) of two geometric distributions".
    # A geometric distribution on {0, 1, 2, ...} with ratio r has mass w_k = (1-r) r^k.
    # Truncating at N gives a finite measure.
    # Let's define a measure as a mixture of two such truncated geometrics.
    # Measure mu = alpha * mu1 + (1-alpha) * mu2
    # mu1: support {0, 1, ..., N1}, weights proportional to r1^k
    # mu2: support {0, 1, ..., N2}, weights proportional to r2^k
    
    # To keep n small for speed, we can merge the supports.
    # Or simply use a single truncated geometric? 
    # The seed used uniform {0..31}.
    # A geometric with r close to 1 is like uniform but with more mass at 0.
    # Let's try optimizing parameters of a single truncated geometric first, then maybe a mixture.
    
    # Parameterization:
    # r in (0, 1): ratio
    # N: truncation point (integer)
    # The weights are w_k = (1-r) r^k for k=0..N.
    # Sum of weights is not 1, but evaluator normalizes.
    # So we can just return support = [0..N] and weights = r^k.
    
    # Let's define a function to compute P for a given measure.
    def compute_P(support, weights):
        n = len(support)
        if n == 0:
            return 0.0
        # Normalize weights just in case, though evaluator does it.
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        s = support
        
        # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j)
        # This is expensive for large n.
        # Let's precompute pairwise sums?
        # n^2 pairs for (i,j) and n^2 pairs for (k,l). Total n^4.
        # If n=100, n^4 = 10^8. Too slow in Python loops.
        # We need a faster way to compute P or keep n small.
        
        # Vectorized approach:
        # Create matrix S_ij = s_i + s_j
        # Create matrix T_kl = 2 s_l - s_k
        # We want sum_{k,l} w_k w_l * (sum_{i,j: S_ij < T_kl} w_i w_j)
        
        # Let A = w_i w_j (outer product of w with w)
        # Let B = S_ij
        # For each (k,l), we need sum of A where B < T_kl.
        # This is like a cumulative sum over B.
        
        # Sort B? But A is tied to indices.
        # Alternative:
        # P = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i+s_j < 2 s_l - s_k)
        # = sum_{i,j,k,l} w_i w_j w_k w_l * I(s_i+s_j+s_k < 2 s_l)
        
        # Let U = s_i + s_j + s_k.
        # We need P(U < 2 s_l).
        # U is the sum of 3 i.i.d. variables.
        # Let's compute the distribution of U?
        # Or just compute the 4D sum efficiently?
        
        # If n is small (e.g., < 50), n^4 might be okay with numpy broadcasting if careful?
        # n=50 -> 50^4 = 6.25e6. That's fine.
        # n=100 -> 100^4 = 1e8. Might be slow in Python/Numpy if not optimized.
        # Let's try to keep n <= 60.
        
        # Efficient computation:
        # 1. Compute S2 = s_i + s_j (n x n matrix)
        # 2. For each k, we have threshold 2 s_l - s_k.
        #    Actually, let's group by l.
        #    For a fixed l, we need sum_{k} w_k * (sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j)
        #    Let C_k = sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j
        #    Then term for l is w_l * sum_k w_k C_k
        
        # Let's precompute the matrix of pairwise sums S2 = s[:, None] + s[None, :]
        # And the weight matrix W2 = w[:, None] * w[None, :]
        
        # For each l (0..n-1):
        #   Thresholds T_k = 2 s_l - s_k for all k.
        #   We need to sum W2 over elements where S2 < T_k[None, :] (broadcasting T_k across rows? No, T_k depends on k, not i,j).
        #   Wait, the condition is s_i + s_j < 2 s_l - s_k.
        #   So for a fixed l and k, the mask is (S2 < (2 s_l - s_k)).
        #   Sum of W2 over that mask is C_k.
        
        # This is still O(n^3) if we loop over l and k and do O(n^2) mask sum.
        # n=50 -> 50^3 * 2500 = 31e6 operations. Okay.
        # n=100 -> 100^3 * 10000 = 1e9. Too slow.
        
        # Can we do better?
        # Sort s.
        # S2 is sorted in rows and columns if s is sorted.
        # For a fixed threshold T, the number of pairs (i,j) with S2[i,j] < T can be found in O(n) using two pointers, but we need the weighted sum.
        # Weighted sum is harder.
        
        # Let's stick to n <= 40 for safety.
        
        S2 = s[:, None] + s[None, :]
        W2 = w[:, None] * w[None, :]
        
        P_val = 0.0
        for l in range(n):
            # T_k = 2 s_l - s_k
            T = 2 * s[l] - s
            # For each k, we need sum of W2 where S2 < T[k]
            # This is a matrix of masks.
            # Mask[k, i, j] = (S2[i, j] < T[k])
            # Sum_k w_k * sum_{i,j} W2[i,j] * Mask[k,i,j]
            # = sum_{i,j} W2[i,j] * (sum_k w_k * Mask[k,i,j])
            
            # Let's compute the inner sum for each (i,j).
            # For fixed (i,j), sum_k w_k * I(S2[i,j] < T[k])
            # T[k] = 2 s_l - s_k.
            # Condition: S2[i,j] < 2 s_l - s_k  =>  s_k < 2 s_l - S2[i,j]
            # So we need sum of w_k for all k such that s_k < 2 s_l - S2[i,j].
            
            # Let V = 2 s_l - S2. (n x n matrix)
            # For each (i,j), we need sum of w_k where s_k < V[i,j].
            # Since s is sorted, we can use searchsorted.
            
            # V = 2 s_l - S2
            V = 2 * s[l] - S2
            # For each element V[i,j], find count of s_k < V[i,j]
            # Using searchsorted on s.
            # idx = np.searchsorted(s, V, side='left')
            # Then sum of w_k for k < idx.
            # We can precompute prefix sums of w.
            
            prefix_w = np.concatenate(([0.0], np.cumsum(w)))
            # idx gives the number of elements in s strictly less than V[i,j]
            # because s_k < V[i,j].
            # searchsorted(s, V, side='left') returns the index where V[i,j] would be inserted to keep order.
            # If s[k] < V[i,j], then k is before the insertion point.
            # So the sum is prefix_w[idx].
            
            idx = np.searchsorted(s, V, side='left')
            # idx has shape (n, n)
            # We need prefix_w[idx]
            # Then sum over i,j of W2[i,j] * prefix_w[idx[i,j]]
            
            inner_sum = np.sum(W2 * prefix_w[idx])
            P_val += w[l] * inner_sum
            
        return P_val

    # Now, let's optimize the parameters.
    # We'll use a mixture of two truncated geometrics.
    # Params: r1, r2, alpha, N1, N2.
    # To keep it simple, let's fix N1=N2=N.
    # Support is {0, 1, ..., N}.
    # Weights are alpha * r1^k + (1-alpha) * r2^k.
    
    def build_measure(params):
        r1, r2, alpha, N_log = params
        N = int(np.exp(N_log))
        N = max(1, min(N, 60)) # Cap N
        
        # Clip r to (1e-6, 1-1e-6)
        r1 = np.clip(r1, 1e-6, 1 - 1e-6)
        r2 = np.clip(r2, 1e-6, 1 - 1e-6)
        alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
        
        k = np.arange(N + 1)
        w1 = r1 ** k
        w2 = r2 ** k
        weights = alpha * w1 + (1 - alpha) * w2
        support = k.astype(np.float64)
        
        # Remove zero weights to reduce n?
        # If r is small, high k have tiny weights.
        # But evaluator handles small weights fine.
        # However, removing them speeds up computation.
        # Let's keep them if weight > 1e-12.
        mask = weights > 1e-12
        support = support[mask]
        weights = weights[mask]
        
        if len(support) == 0:
            return np.array([0.0]), np.array([1.0])
            
        return support, weights

    def objective(params):
        support, weights = build_measure(params)
        P = compute_P(support, weights)
        return -P

    # Initial guesses
    # Seed: uniform {0..31}. r=1, N=31.
    # But r=1 is boundary.
    # Let's try r1=0.9, r2=0.5, alpha=0.5, N=30.
    x0 = np.array([0.9, 0.5, 0.5, np.log(30)])
    
    best_params = x0
    best_score = -objective(x0)
    
    # Optimize
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
            
        # Try a few other starting points
        starts = [
            np.array([0.8, 0.6, 0.5, np.log(20)]),
            np.array([0.95, 0.3, 0.7, np.log(40)]),
            np.array([0.7, 0.9, 0.3, np.log(25)]),
        ]
        
        for x_start in starts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(objective, x_start, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass
                
    except Exception:
        pass

    # Build final measure
    support, weights = build_measure(best_params)
    
    # Ensure validity
    if len(support) == 0:
        support = np.array([0.0])
        weights = np.array([1.0])
        
    # Check for NaN/Inf
    if not (np.all(np.isfinite(support)) and np.all(np.isfinite(weights))):
        support = np.array([0.0])
        weights = np.array([1.0])
        
    if np.any(support < -1e-9):
        support = np.maximum(support, 0.0)
        
    if np.any(weights < -1e-9):
        weights = np.maximum(weights, 0.0)
        
    if np.sum(weights) <= 1e-9:
        weights = np.ones_like(weights)
        
    return support, weights

# EVOLVE-BLOCK-END
