import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h is a 1D array of non-negative floats.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    best_h = np.ones(N)
    best_score = compute_ratio(best_h)

    # Helper to optimize weights given a support mask
    # We want to maximize R(h) = sum(h^2) / (sum(h) * max(conv(h,h)))
    # Let S = support indices. h_i = 0 for i not in S.
    # We can fix sum(h) = 1 to remove scale invariance ambiguity during optimization.
    # Then maximize sum(h^2) / max(conv(h,h)).
    
    def optimize_weights_on_support(mask):
        """
        mask: boolean array of length N indicating which indices are non-zero.
        Returns: (score, h)
        """
        K = np.sum(mask)
        if K == 0:
            return 0.0, np.zeros(N)
        if K == 1:
            h = np.zeros(N)
            h[np.argmax(mask)] = 1.0
            return compute_ratio(h), h

        # Variables: weights w[0..K-1] for the active indices
        # Constraint: sum(w) = 1, w >= 0
        
        indices = np.where(mask)[0]
        
        def objective(w):
            # w is the vector of weights for active indices
            h = np.zeros(N)
            h[indices] = w
            # Compute ratio components
            l2_sq = np.sum(w * w)
            # Convolution of h with h
            # We only need the max. 
            # conv(h,h)_k = sum_i h_i h_{k-i}
            # Since h is sparse, we can compute efficiently or just use numpy.
            # N=50, so numpy convolve is fast enough.
            hh = np.convolve(h, h, mode='full')
            hh_max = np.max(hh)
            if hh_max < 1e-30:
                return -1e9 # Avoid division by zero, though w>=0 and sum=1 implies max>0
            return - (l2_sq / hh_max) # Minimize negative ratio

        def gradient(w):
            # Numerical gradient for simplicity, as analytic is complex with max()
            # K is small (<= 20), so this is fine.
            g = np.zeros_like(w)
            eps = 1e-7
            base = objective(w)
            for i in range(K):
                w_plus = w.copy()
                w_plus[i] += eps
                # Renormalize? The objective assumes sum(w)=1 implicitly in the ratio formula?
                # Wait, the objective function defined above uses w directly.
                # If we perturb w_i, sum(w) != 1.
                # It is better to parameterize w such that sum(w)=1 is always satisfied.
                # Or just use numerical gradient on the constrained manifold?
                # SLSQP handles constraints. It's better to provide the gradient of the 
                # function with respect to the variables w, assuming the constraint 
                # sum(w)=1 is handled by the optimizer.
                # However, the objective function as defined above does NOT enforce sum(w)=1 
                # inside the calculation, it just uses the values. 
                # If the optimizer moves off the simplex, the ratio formula changes.
                # Actually, the ratio is scale invariant. 
                # R(c*h) = R(h).
                # So we can just maximize R(h) over h >= 0.
                # The constraint sum(w)=1 is just a normalization.
                # Let's stick to the normalized formulation to keep numbers stable.
                # We will use a parameterization that guarantees sum=1? 
                # No, SLSQP with equality constraint is standard.
                # Let's just use finite differences if analytic is hard, but K is small.
                # Let's compute analytic gradient of f(w) = sum(w^2) / max(conv(w,w))
                # This is tricky due to max.
                # Let's rely on SLSQP's finite diff or use a simple Nelder-Mead on the 
                # simplex? Nelder-Mead doesn't handle constraints well.
                # Let's use SLSQP with finite diff gradient (default if not provided).
                # It's K dimensions. K <= 20.
                pass # We won't provide gradient, let scipy estimate it.
            
            return None # Indicate to use finite differences? No, scipy requires a function or None.
            # If we return None, we should not have defined the function to return a value.
            # Let's just not pass a jac argument.

        # We need to define the function correctly for scipy.
        # Let's re-define objective to return the value, and let scipy do finite diff.
        
        def obj_func(w):
            h = np.zeros(N)
            h[indices] = w
            l2_sq = np.sum(w * w)
            hh = np.convolve(h, h, mode='full')
            hh_max = np.max(hh)
            if hh_max < 1e-30:
                return 1e9
            return - (l2_sq / hh_max)

        # Initial guess: uniform weights
        w0 = np.ones(K) / K
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: w} # w >= 0
        ]
        # Note: scipy's ineq constraint is fun(x) >= 0. So w >= 0 is correct.
        
        try:
            res = minimize(obj_func, w0, method='SLSQP', bounds=[(0, 1)] * K, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                w_opt = res.x
                # Ensure non-negative and normalized
                w_opt = np.maximum(w_opt, 0)
                w_opt = w_opt / np.sum(w_opt)
                h_opt = np.zeros(N)
                h_opt[indices] = w_opt
                score = compute_ratio(h_opt)
                return score, h_opt
            else:
                # Fallback to uniform
                h_opt = np.zeros(N)
                h_opt[indices] = 1.0 / K
                return compute_ratio(h_opt), h_opt
        except Exception:
            h_opt = np.zeros(N)
            h_opt[indices] = 1.0 / K
            return compute_ratio(h_opt), h_opt

    # Strategy:
    # 1. Try all subsets of size K for small K? 
    #    C(50, 5) = 2,118,760. Too many for full optimization.
    #    C(50, 4) = 230,300. Maybe feasible if optimization is fast?
    #    Optimization is ~200 iters * 50*50 conv = 500k ops per opt? No, conv is 100*50=5000 ops.
    #    200 * 5000 = 1M ops per optimization.
    #    230k * 1M = 2.3 * 10^11 ops. Too slow.
    
    # 2. Heuristic Support Search:
    #    Start with a random support of size K.
    #    Optimize weights.
    #    Try swapping one element in support with one outside.
    #    If score improves, accept.
    #    Repeat until no improvement or time limit.
    
    # Let's try K in [3, 15].
    
    rng = np.random.default_rng(42)
    
    for K in range(3, 16):
        if time.time() > deadline:
            break
            
        # Initialize support randomly
        support = set(rng.choice(N, size=K, replace=False))
        
        # Initial optimization
        mask = np.zeros(N, dtype=bool)
        mask[list(support)] = True
        score, h = optimize_weights_on_support(mask)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        # Local search: Swap
        # We want to find a swap (i in support, j not in support) that improves score.
        # Instead of trying all N-K * K swaps, we can sample.
        # Or we can be smart: 
        # The ratio is roughly related to "sparsity" and "spread".
        # Let's just do random restarts of the local search for a fixed number of iterations.
        
        max_swaps = 500
        current_support = set(support)
        current_score = score
        current_h = h
        
        for swap_iter in range(max_swaps):
            if time.time() > deadline:
                break
                
            # Pick a random i in support and j not in support
            if not current_support:
                break
            i = rng.choice(list(current_support))
            candidates = [x for x in range(N) if x not in current_support]
            if not candidates:
                break
            j = rng.choice(candidates)
            
            new_support = set(current_support)
            new_support.remove(i)
            new_support.add(j)
            
            new_mask = np.zeros(N, dtype=bool)
            new_mask[list(new_support)] = True
            
            new_score, new_h = optimize_weights_on_support(new_mask)
            
            if new_score > current_score:
                current_support = new_support
                current_score = new_score
                current_h = new_h
                if current_score > best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                # If no improvement, we might get stuck. 
                # Occasionally restart or try a random perturbation of the support.
                if swap_iter % 50 == 0:
                    # Randomly change 1-2 elements
                    for _ in range(rng.integers(1, 3)):
                        i = rng.choice(list(current_support))
                        candidates = [x for x in range(N) if x not in current_support]
                        if candidates:
                            j = rng.choice(candidates)
                            current_support.remove(i)
                            current_support.add(j)
        
        # After local search for this K, try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            support = set(rng.choice(N, size=K, replace=False))
            mask = np.zeros(N, dtype=bool)
            mask[list(support)] = True
            score, h = optimize_weights_on_support(mask)
            if score > best_score:
                best_score = score
                best_h = h.copy()

    # Final check: Uniform distribution
    h_uniform = np.ones(N) / N
    score_uniform = compute_ratio(h_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_h = h_uniform.copy()
        
    # Final check: Delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    score_delta = compute_ratio(h_delta)
    if score_delta > best_score:
        best_score = score_delta
        best_h = h_delta.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
