import numpy as np

N = 400

def construct():
    # 1. Generate a random antisymmetric vector x
    # x[i] = -x[N-1-i]
    rng = np.random.default_rng(42)
    half_n = N // 2
    r = rng.standard_normal(half_n)
    x = np.zeros(N)
    x[:half_n] = r
    x[half_n:] = -r[::-1]
    
    # Normalize x to have unit norm to make t interpretable
    norm_x = np.linalg.norm(x)
    if norm_x > 0:
        x = x / norm_x
    else:
        return np.ones(N)

    # 2. Compute T (autoconv of ones) and A (autoconv of x)
    ones = np.ones(N)
    T = np.convolve(ones, ones)
    A = np.convolve(x, x)
    
    # 3. Find t > 0 that minimizes max(T + t*A)
    # g(t) is convex.
    # t=0 gives max(T) = N.
    # As t increases, T + t*A at center (index N-1) is N - t (since A[N-1] = -1 for unit norm antisymmetric? 
    # Let's check A[N-1] for unit norm x.
    # A[N-1] = sum x[i]x[N-1-i] = sum -x[i]^2 = -1. Yes.)
    # So center value is N - t.
    # We want to keep center value high enough? No, we want to MINIMIZE the max.
    # So we want to push t up to lower the center, but not so high that other peaks (where A>0) rise above the center.
    
    # Let's do a ternary search for t in [0, 2*N] or similar.
    # Or just grid search since it's fast.
    
    best_t = 0
    best_max_val = np.max(T)
    
    # The max value can't go below 0 (if we ignore absolute value in max, but T is positive).
    # Actually, we just need to find the min of the max.
    
    # Heuristic bound for t: 
    # If t is very large, max ~ t * max(A).
    # We want to find the crossing point.
    
    ts = np.linspace(0, 2*N, 2000)
    for t in ts:
        curr_max = np.max(T + t * A)
        if curr_max < best_max_val:
            best_max_val = curr_max
            best_t = t
            
    # Refine around best_t
    # Simple binary/ternary refinement
    left = max(0, best_t - 1)
    right = best_t + 1
    for _ in range(50):
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        v1 = np.max(T + m1 * A)
        v2 = np.max(T + m2 * A)
        if v1 < v2:
            right = m2
        else:
            left = m1
    best_t = (left + right) / 2
    best_max_val = np.max(T + best_t * A)
    
    # Construct h
    # h = 1 + sqrt(t) * x
    # Note: If best_t is 0, we return ones.
    eps = np.sqrt(best_t)
    h = ones + eps * x
    
    # Verify sum is not zero
    # sum(h) = N + eps * sum(x) = N + 0 = N != 0.
    
    return h
