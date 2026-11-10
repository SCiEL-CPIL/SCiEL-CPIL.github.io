import numpy as np
import time
import sys

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    """
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    
    C = np.cumsum(pp_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner
            
    return P

def generate_mixture_support_weights(q_zero, alpha, scale, n_tail, n_total=None):
    """
    Generate support and weights for a mixture distribution:
    - Mass q_zero at 0
    - Mass (1 - q_zero) distributed over n_tail points from a Pareto-like tail
    
    We use quantiles of the Pareto distribution to place the tail points.
    To maximize the chance that X4 is large, we want the tail to be heavy.
    Pareto CDF: F(x) = 1 - (scale/x)^alpha for x >= scale.
    Quantile: x_p = scale * (1 - p)^(-1/alpha)
    
    We place tail points at quantiles p_1, ..., p_n_tail.
    To cover the range well, we use logarithmically spaced quantiles in the tail.
    Specifically, we want to capture the mass from p=0 (start of tail) to p=1 (infinity).
    But we can't go to infinity. We pick n_tail points.
    
    Strategy:
    The tail mass is (1 - q_zero).
    We divide the tail probability into n_tail equal parts? 
    Or we pick specific quantiles.
    
    Let's pick quantiles u_1, ..., u_n_tail uniformly in [0, 1] but shifted?
    Actually, for a heavy tail, the mass is concentrated in the lower part of the tail range (near scale).
    But we want X4 to be large.
    
    Let's try:
    Support points:
    s[0] = 0
    s[1..n_tail] = Pareto quantiles.
    
    Weights:
    w[0] = q_zero
    w[1..n_tail] = (1 - q_zero) / n_tail ? 
    Or should we weight the tail points differently?
    If we use quantiles, the natural weight is the probability mass between quantiles.
    If we pick quantiles q_1 < q_2 < ... < q_n_tail,
    then the weight of point i is proportional to (q_i - q_{i-1}).
    
    Let's use uniform quantiles in the tail probability space.
    Tail CDF G(x) = (x/scale)^(-alpha) for x >= scale.
    Inverse CDF (Quantile) Q(p) = scale * (1-p)^(-1/alpha) for p in [0,1).
    
    Let's pick n_tail points at p = 0, 1/n_tail, 2/n_tail, ..., (n_tail-1)/n_tail.
    The last point corresponds to p close to 1, which is very large.
    
    Weights:
    w_0 = q_zero
    For i in 1..n_tail:
        p_i = (i-1) / n_tail
        p_{i+1} = i / n_tail (if i < n_tail)
        Actually, let's just assign equal weights to the tail points for simplicity in the first pass,
        or use the trapezoidal rule approximation for the integral.
        
    Let's stick to:
    s[0] = 0
    s[i] = scale * (1 - p_i)^(-1/alpha) where p_i = (i-1)/n_tail for i=1..n_tail
    Wait, if p_i = 0, s[1] = scale.
    If p_i = (n_tail-1)/n_tail, s[n_tail] is very large.
    
    Weights:
    w[0] = q_zero
    w[i] = (1 - q_zero) / n_tail for i=1..n_tail
    
    This is a discrete approximation.
    """
    if n_total is None:
        n_total = n_tail + 1
    
    s = np.zeros(n_total)
    w = np.zeros(n_total)
    
    s[0] = 0.0
    w[0] = q_zero
    
    if n_tail > 0 and (1 - q_zero) > 0:
        # Quantiles in the tail
        # p from 0 to 1
        # We want to cover the tail.
        # Let's use p_i = i / n_tail for i=0..n_tail-1?
        # Or p_i = (i + 0.5) / n_tail?
        
        # Let's try p_i = i / n_tail for i=0..n_tail-1
        # p_0 = 0 -> x = scale
        # p_{n_tail-1} = (n_tail-1)/n_tail -> x = scale * (1/n_tail)^(-1/alpha)
        
        p_vals = np.arange(n_tail) / n_tail
        # Avoid p=1 which is infinity
        # p_vals is 0, 1/n, ..., (n-1)/n
        
        # Quantiles
        # Q(p) = scale * (1 - p)^(-1/alpha)
        # If p=0, Q=scale.
        # If p->1, Q->inf.
        
        s_tail = scale * (1.0 - p_vals) ** (-1.0 / alpha)
        
        # Clip to avoid overflow
        max_val = 1e100
        s_tail = np.clip(s_tail, scale, max_val)
        
        s[1:n_total] = s_tail
        
        w[1:n_total] = (1.0 - q_zero) / n_tail
    
    return s, w

def optimize_mixture_params(q_zero, alpha, scale, n_tail, time_limit):
    """
    Coordinate ascent on q_zero, alpha, scale.
    n_tail is fixed for this call.
    """
    best_score = -1.0
    best_q = q_zero
    best_alpha = alpha
    best_scale = scale
    
    start_time = time.time()
    
    # Initial evaluation
    s, w = generate_mixture_support_weights(q_zero, alpha, scale, n_tail)
    current_score = evaluate_measure(s, w)
    if current_score > best_score:
        best_score = current_score
        best_q, best_alpha, best_scale = q_zero, alpha, scale
        
    # Coordinate ascent
    # Parameters: q_zero in (0, 1), alpha in (0.1, 5.0), scale in (0.1, 100.0)
    
    # We will do a grid search + local refinement
    # Since n_tail is small (<= 20), evaluation is fast.
    
    # Grid for q_zero
    q_grid = np.linspace(0.0, 0.99, 20)
    # Grid for alpha
    alpha_grid = np.logspace(-1, 0.7, 20) # 0.1 to ~5
    # Grid for scale
    scale_grid = np.logspace(-1, 2, 20) # 0.1 to 100
    
    # Coarse grid search
    for q in q_grid:
        if time.time() - start_time > time_limit * 0.5:
            break
        for a in alpha_grid:
            if time.time() - start_time > time_limit * 0.5:
                break
            for sc in scale_grid:
                s, w = generate_mixture_support_weights(q, a, sc, n_tail)
                score = evaluate_measure(s, w)
                if score > best_score:
                    best_score = score
                    best_q, best_alpha, best_scale = q, a, sc
                    
    # Local refinement around best
    # Use a smaller grid around the best found
    q_lo, q_hi = max(0, best_q - 0.05), min(0.99, best_q + 0.05)
    a_lo, a_hi = max(0.1, best_alpha * 0.8), min(10, best_alpha * 1.2)
    sc_lo, sc_hi = max(0.01, best_scale * 0.8), min(1000, best_scale * 1.2)
    
    # Refinement steps
    for step in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a fine grid around current best
        q_fine = np.linspace(q_lo, q_hi, 10)
        a_fine = np.linspace(a_lo, a_hi, 10)
        sc_fine = np.linspace(sc_lo, sc_hi, 10)
        
        improved = False
        for q in q_fine:
            for a in a_fine:
                for sc in sc_fine:
                    s, w = generate_mixture_support_weights(q, a, sc, n_tail)
                    score = evaluate_measure(s, w)
                    if score > best_score:
                        best_score = score
                        best_q, best_alpha, best_scale = q, a, sc
                        improved = True
                        
        if not improved:
            break
            
        # Shrink ranges
        q_lo, q_hi = max(0, best_q - 0.02), min(0.99, best_q + 0.02)
        a_lo, a_hi = max(0.1, best_alpha * 0.9), min(10, best_alpha * 1.1)
        sc_lo, sc_hi = max(0.01, best_scale * 0.9), min(1000, best_scale * 1.1)
        
    return best_q, best_alpha, best_scale, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 15.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different n_tail values
    # n_tail = 1 is just 2 points
    # n_tail = 2 is 3 points
    # ...
    # Larger n_tail allows better approximation of the continuous Pareto tail
    # But evaluation cost is O(n^2). n = n_tail + 1.
    # n=20 -> 400 ops per eval. Very fast.
    # n=50 -> 2500 ops. Still fast.
    
    n_tail_candidates = [2, 5, 10, 20, 30, 50]
    
    # Allocate time to each
    # We will run a full optimization for each n_tail
    
    for n_tail in n_tail_candidates:
        if time.time() + 10.0 > deadline - safety_margin:
            break
            
        # Time budget for this n_tail
        # We have ~500s left. 6 candidates. ~80s each.
        time_budget = min(80.0, deadline - safety_margin - time.time() - 5.0)
        if time_budget < 1.0:
            break
            
        # Initial guess
        # q_zero around 0.5, alpha around 1.0, scale around 1.0
        # We'll run a few random restarts for the initial grid center?
        # No, the grid search inside optimize_mixture_params covers the space.
        
        try:
            best_q, best_alpha, best_scale, score = optimize_mixture_params(
                0.5, 1.0, 1.0, n_tail, time_budget
            )
            
            if score > best_score:
                best_score = score
                s, w = generate_mixture_support_weights(best_q, best_alpha, best_scale, n_tail)
                best_support = s
                best_weights = w
                
        except Exception as e:
            # Fallback to a simple valid measure
            if best_support is None:
                best_support = np.array([0.0, 1.0])
                best_weights = np.array([0.5, 0.5])
            continue
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0, 2.0])
        best_weights = np.array([1/3, 1/3, 1/3])
        
    # Final validation and cleanup
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        n = len(best_support)
        best_weights = np.ones(n) / n
        
    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
