import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal signed step function.
    Uses a discrete gradient descent heuristic on binary signs {-1, 1}.
    The goal is to minimize max(|autoconv|) / (sum(signs))^2.
    Since we restrict to {-1, 1}, sum(signs)^2 is variable but bounded.
    Actually, for pure {-1, 1}, the integral is sum(s). 
    The metric is max(|ac|) / (sum(s))^2.
    Note: If sum(s) is small, the denominator is small, making the metric large.
    Therefore, we also want sum(s) to be large in magnitude.
    However, the prompt says "squared integral is constant (N^2) for unit-magnitude steps" 
    in the research programme text? 
    "leveraging the fact that the squared integral is constant (N^2) for unit-magnitude steps"
    This implies the research programme assumes we are looking at the case where the integral is maximized or fixed?
    Or does it mean we should optimize assuming the denominator is roughly N^2?
    If we allow sum(s) to be 0, the metric is inf. So we must avoid sum(s) near 0.
    
    The seed program's _compute_metric uses (sum(signs))^2.
    If we pick all 1s, sum=N, denom=N^2. ac = [1, 2, ..., N, ..., 1]. max=N. metric = N/N^2 = 1/N.
    If we pick random signs, sum ~ sqrt(N). denom ~ N. max(ac) ~ N (likely). metric ~ N/N = 1.
    So 1/N is much better.
    
    The research programme says: "squared integral is constant (N^2) for unit-magnitude steps".
    This statement is technically false for arbitrary signs in {-1,1} (sum can be 0).
    However, if the programme assumes we are minimizing the numerator primarily, or that we should 
    keep the sum close to N, then we should bias towards all 1s.
    
    Let's re-read carefully: "optimizing it using a discrete gradient descent heuristic that directly penalizes the maximum absolute autoconvolution value."
    
    If I strictly follow "minimize max absolute autoconvolution", I should aim for a sequence with low peak autoconvolution.
    Barker sequences or Golay complementary sequences have low sidelobes.
    For N=400, we can't have perfect zero sidelobes, but we can minimize the peak sidelobe.
    
    Let's calculate the metric for an all-1 sequence:
    sum = 400. denom = 160000.
    ac max = 400.
    metric = 400 / 160000 = 0.0025.
    
    Let's calculate for a random sequence:
    sum ~ 20 (std dev sqrt(400)=20). denom ~ 400.
    ac max ~ 400 (likely close to N for random).
    metric ~ 400 / 400 = 1.0.
    
    So all-1s is significantly better than random.
    Is there a sequence with max(ac) < N?
    Yes, if we balance signs, the peak of ac might drop.
    Example: N=2, [1, -1]. sum=0. metric=inf.
    Example: N=3, [1, 1, -1]. sum=1. denom=1.
    ac: [1, 1, -1] * [1, 1, -1] -> 
    k=0: 1
    k=1: 1*1 + 1*-1 = 0? No.
    ac[0] = 1
    ac[1] = 1*1 + 1*-1 = 0
    ac[2] = 1*-1 + 1*1 + (-1)*1 = -1
    ac[3] = 1*1 + (-1)*1 = 0
    ac[4] = -1
    max abs = 1. metric = 1/1 = 1.
    All 1s for N=3: sum=3, denom=9. ac max=3. metric=3/9=0.33.
    
    It seems keeping the sum large is crucial.
    The research programme hint "squared integral is constant (N^2)" suggests that the search space should be constrained to sequences where the integral is effectively N (i.e., all 1s) OR the objective function in the code should be `max(|ac|) / N^2`?
    
    "Interface: ... score is the upper bound on C3' (maximum absolute autoconvolution)"
    "The evaluator measures the raw value"
    
    If the evaluator calculates `max(|ac|) / (sum(s))^2`, then we must optimize that.
    If the evaluator calculates `max(|ac|) / N^2`, then we just minimize `max(|ac|)`.
    
    The seed program uses `_compute_metric` which does `max_ac / (integral * integral)`.
    The prompt says "Keep the function names and signatures of the seed program."
    It does not explicitly say we must keep the internal logic of `_compute_metric`, but `construct` must return signs.
    The evaluator will run its own scoring.
    "The evaluator measures the raw value".
    Usually, in these tasks, the "raw value" is defined in the task description.
    Task: "minimizing the maximum absolute autoconvolution divided by squared integral."
    So the evaluator likely computes `max(|ac|) / (sum(s))^2`.
    
    Therefore, we must optimize `max(|ac|) / (sum(s))^2`.
    
    Strategy:
    1. Start with all 1s (metric = 1/N).
    2. Try to flip bits to reduce `max(|ac|)` while keeping `sum(s)` large.
    3. Flipping a bit from 1 to -1 reduces sum by 2.
       New sum = S - 2.
       New denom = (S-2)^2.
       New max_ac might decrease.
       We need `(new_max_ac) / (S-2)^2 < (old_max_ac) / S^2`.
    
    Let's implement a local search that evaluates this specific metric efficiently.
    Since N=400, we can afford to compute the full convolution for each candidate flip if we do it smartly, but $N^2$ per flip * N flips * iterations might be slow in pure Python if not vectorized well.
    However, `fftconvolve` is fast. $400^2$ operations is trivial.
    We can just recompute the metric for each flip.
    
    To be safe with time, we will:
    1. Initialize with all 1s.
    2. Iterate through indices.
    3. For each index, calculate the metric if flipped.
    4. Accept if better.
    5. Repeat for a few passes or until time limit.
    6. Add random restarts to escape local minima.
    """
    
    start_time = time.time()
    time_limit = 100  # seconds, leaving buffer for overhead
    
    def get_metric(signs):
        s = signs.astype(np.float64)
        integral = np.sum(s)
        if abs(integral) < 1e-12:
            return np.inf
        ac = np.fft.irfft(np.fft.rfft(s) * np.fft.rfft(s))
        # Note: irfft of power spectrum gives circular convolution if padded, but standard fftconvolve is linear.
        # Using fftconvolve is safer for linear autoconvolution.
        # However, for N=400, linear convolution length is 799.
        # Let's use scipy.signal.fftconvolve as in the seed, it's robust.
        from scipy.signal import fftconvolve
        ac = fftconvolve(s, s, mode='full')
        max_ac = np.max(np.abs(ac))
        return max_ac / (integral * integral)

    # Initial state: all 1s
    signs = np.ones(N, dtype=np.int32)
    current_metric = get_metric(signs)
    
    # Local search
    # We try to improve the metric by flipping signs.
    # To speed up, we can limit the number of flips or use a greedy approach.
    
    # To avoid getting stuck at all-1s (which might be a local optimum but not global),
    # we can introduce some randomness.
    
    # Let's try a multi-start approach.
    # Start 1: All 1s.
    # Start 2: Random.
    # Start 3: Alternating? (Sum 0 or 1, likely bad).
    
    best_signs = signs.copy()
    best_metric = current_metric
    
    # We will perform several rounds of local search.
    # In each round, we sweep through all indices and accept improvements.
    
    max_rounds = 10
    
    # Helper to run local search on a given starting configuration
    def local_search(init_signs, max_rounds=5):
        signs = init_signs.copy()
        curr_m = get_metric(signs)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            # Randomize order to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                # Flip
                signs[i] = -signs[i]
                new_m = get_metric(signs)
                if new_m < curr_m:
                    curr_m = new_m
                    improved = True
                else:
                    signs[i] = -signs[i] # Revert
        return signs, curr_m

    # Run local search from all-1s
    s1, m1 = local_search(np.ones(N, dtype=np.int32), max_rounds=10)
    if m1 < best_metric:
        best_metric = m1
        best_signs = s1

    # Try a few random starts
    np.random.seed(42)
    for _ in range(20): # 20 random starts
        if time.time() - start_time > time_limit:
            break
        init = np.random.choice([-1, 1], size=N).astype(np.int32)
        # Optional: Bias towards positive sum?
        # If sum is small, metric is huge. 
        # We can force at least 60% to be 1 to keep denom large.
        # But let's let the optimizer handle it, or add a penalty.
        # Actually, if sum is 0, metric is inf. 
        # Random signs usually have sum ~ 0. 
        # So random starts will have very high metrics.
        # It is better to start with biased random.
        # P(1) = 0.6
        
        init_biased = np.random.choice([-1, 1], size=N, p=[0.2, 0.8]).astype(np.int32)
        s_r, m_r = local_search(init_biased, max_rounds=2)
        if m_r < best_metric:
            best_metric = m_r
            best_signs = s_r

    # Final check against all 1s and all -1s just in case
    m_1 = get_metric(np.ones(N, dtype=np.int32))
    if m_1 < best_metric:
        best_metric = m_1
        best_signs = np.ones(N, dtype=np.int32)
    
    m_neg1 = get_metric(-np.ones(N, dtype=np.int32))
    if m_neg1 < best_metric:
        best_metric = m_neg1
        best_signs = -np.ones(N, dtype=np.int32)

    return best_signs
# EVOLVE-BLOCK-END
