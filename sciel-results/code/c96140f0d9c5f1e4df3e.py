"""Erdos minimum overlap: Continuous gradient-based optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    """Exact evaluator score."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """
    Adjust h to sum to N/2 while keeping values in [0, 1].
    Used for final projection or initialization.
    """
    target = N / 2.0
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _smooth_max(corr, beta):
    """
    Smooth approximation of max(corr) using log-sum-exp.
    beta controls the sharpness. Higher beta -> closer to max.
    """
    # Shift to avoid overflow
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta

def _objective_and_grad(h, beta, lambda_sum):
    """
    Objective: smooth_max(correlate(h, 1-h)) + lambda_sum * (sum(h) - N/2)^2
    Returns value and gradient.
    """
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    
    # Soft max value
    m = np.max(corr)
    exp_corr = np.exp((corr - m) * beta)
    sum_exp = np.sum(exp_corr)
    val_sm = m + np.log(sum_exp) / beta
    
    # Gradient of soft max w.r.t. corr
    # d/dcorr_i [ m + 1/beta * log(sum(exp(beta*(corr-m))) ) ]
    # Note: m = max(corr), but in the smooth version we treat m as constant for the derivative 
    # of the log-sum-exp part relative to the shift, or simply use the standard softmax gradient.
    # Actually, standard log-sum-exp gradient is softmax(corr * beta) / beta.
    # Let S = sum(exp(beta*(corr-m))). val = m + log(S)/beta.
    # d val / d corr_k = (1/beta) * (1/S) * exp(beta*(corr_k-m)) * beta * (delta_{k, argmax} - 1) ?
    # No, simpler: LSE(x) = log(sum(exp(x))). d LSE / dx = softmax(x).
    # Here x = beta * (corr - m). But m depends on corr.
    # It is safer to just use the standard LSE on corr directly with a shift for stability.
    # LSE(corr) = log(sum(exp(corr - max(corr))) + max(corr)).
    # Gradient is softmax(corr - max(corr)).
    
    # Let's stick to the standard stable LSE gradient.
    shifted = corr - m
    exp_shifted = np.exp(shifted * beta)
    softmax_weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of LSE(corr) w.r.t. corr is softmax_weights * beta? 
    # d/dcorr_k [ log(sum(exp(beta*(corr-m)))) + m ]
    # If m is fixed, grad is softmax * beta.
    # If m varies, it's slightly different, but for large beta, the argmax term dominates.
    # For optimization, using the standard softmax gradient is a common and effective heuristic.
    grad_corr = softmax_weights * beta
    
    # Chain rule: corr = correlate(h, 1-h)
    # corr_k = sum_i h[i] * (1-h)[i+k] (with appropriate offsets for 'full' mode)
    # np.correlate(h, g, 'full')[k] = sum_i h[i] * g[i+k-N+1] ? 
    # Let's derive the gradient numerically or analytically.
    # Analytical is complex due to the 'full' mode indexing.
    # However, we can compute the gradient of the objective w.r.t. h using the chain rule:
    # grad_h = grad_corr convolved with g and h appropriately.
    # 
    # corr[k] = sum_i h[i] * g[i + k - (N-1)]  (0-indexed, k in 0..2N-2)
    # Let j = i + k - (N-1) => i = j - k + (N-1)
    # This is a convolution-like operation.
    # Actually, np.correlate(a, v) computes sum a[i] * v[i+k].
    # Here a=h, v=g.
    # corr[k] = sum_i h[i] * g[i+k].
    # d corr[k] / d h[j] = g[j+k] if 0 <= j+k < N, else 0.
    # d corr[k] / d g[j] = h[j+k] if 0 <= j+k < N, else 0.
    #
    # We want grad_h = sum_k grad_corr[k] * d corr[k] / d h[j]
    # grad_h[j] = sum_k grad_corr[k] * g[j+k] (where valid)
    # This is equivalent to a convolution of grad_corr with g.
    # Specifically, (grad_corr * g)[j] = sum_k grad_corr[k] * g[j+k].
    # Wait, standard convolution (a*b)[n] = sum_k a[k] b[n-k].
    # Correlation is sum_k a[k] b[k+n].
    # So grad_h is the correlation of grad_corr with g.
    # grad_h = np.correlate(grad_corr, g, mode='valid' or 'same'?)
    # The indices of h are 0..N-1.
    # k ranges 0..2N-2.
    # For a fixed j (0..N-1), the sum is over k such that 0 <= j+k < N.
    # This is exactly the 'valid' part of the correlation if we align them correctly?
    # Let's use a simpler approach: 
    # grad_h[j] = sum_{k} grad_corr[k] * g[j+k]
    # This is the cross-correlation of grad_corr and g, evaluated at lag j?
    # No, it's the convolution of grad_corr and reversed g?
    # Let's just compute it via FFT or explicit loops if N is small (100 is small enough for O(N^2) if optimized, but FFT is O(N log N)).
    # Actually, N=100 is tiny. O(N^2) = 10,000 operations. Python loops are slow, but vectorized numpy is fast.
    # Let's vectorize:
    # grad_h[j] = sum_{k=0}^{2N-2} grad_corr[k] * g[j+k]
    # We can create a matrix M where M[j, k] = g[j+k] if valid else 0.
    # Then grad_h = M @ grad_corr.
    
    # Constructing M explicitly for N=100 is 100x200 matrix. Very fast.
    # But we can do better with sliding window view or just simple indexing.
    
    # Let's try:
    # g is length N. grad_corr is length 2N-1.
    # For each j in 0..N-1:
    #   valid_k are those where 0 <= j+k < N => -j <= k < N-j.
    #   Since k >= 0, we need max(0, -j) <= k < N-j.
    #   So k ranges from 0 to N-j-1.
    #   grad_h[j] = sum_{k=0}^{N-j-1} grad_corr[k] * g[j+k]
    
    # This can be computed using a sliding dot product or FFT.
    # Let's use FFT for robustness and speed, although N is small.
    # Or just use np.convolve.
    # grad_h = np.convolve(grad_corr, g, mode='same')? 
    # Let's check the definition.
    # np.convolve(a, v, 'same') returns output of length max(M, N).
    # If we convolve grad_corr (len 2N-1) with g (len N), the 'valid' part would be length (2N-1) - N + 1 = N.
    # The 'valid' part of convolution (a, b) is sum_k a[k] b[i+k] ?? No.
    # Correlation (a, b) = sum a[k] b[k+i].
    # Convolution (a, b) = sum a[k] b[i-k].
    # So Correlation(a, b) = Convolution(a, reverse(b)).
    # We want sum_k grad_corr[k] * g[j+k]. This is Correlation(grad_corr, g)[j].
    # So grad_h = np.correlate(grad_corr, g, mode='valid')?
    # 'valid' mode returns only those convolutions that do not assume zeros.
    # Length of 'valid' for len(a)=2N-1, len(b)=N is (2N-1) - N + 1 = N.
    # The output index j corresponds to sum_{k=0}^{N-1} a[k] b[j+k]?
    # Let's verify with a small example.
    # a = [1, 2, 3] (len 3), b = [4, 5] (len 2).
    # Correlate(a, b, 'valid'):
    # j=0: a[0]*b[0] + a[1]*b[1] = 1*4 + 2*5 = 14
    # j=1: a[1]*b[0] + a[2]*b[1] = 2*4 + 3*5 = 23
    # Formula: sum_{k=0}^{len(b)-1} a[k+j] b[k].
    # My formula for grad_h[j] was sum_{k=0}^{N-j-1} grad_corr[k] g[j+k].
    # This is sum_{k} a[k] b[j+k].
    # This is NOT the standard correlation definition used by numpy.
    # Numpy correlate(a, v) computes sum a[i] v[i+k].
    # So if I want sum_k a[k] v[j+k], I can swap them?
    # sum_k a[k] v[j+k] = sum_k v[j+k] a[k].
    # Let m = j+k => k = m-j.
    # sum_m v[m] a[m-j].
    # This is Correlation(v, a)[j]?
    # Correlate(v, a)[j] = sum_i v[i] a[i+j].
    # If I set i=m, then sum_m v[m] a[m+j]. Close, but sign of j is flipped.
    # Correlate(a, v)[j] = sum_i a[i] v[i+j].
    # So sum_k a[k] v[j+k] is Correlate(a, v) with lag -j?
    # Let's just use the matrix approach or explicit loop for N=100 to avoid confusion.
    # N=100, 2N-1=199.
    # grad_h = np.zeros(N)
    # for j in range(N):
    #     # k from 0 to N-j-1
    #     grad_h[j] = np.dot(grad_corr[:N-j], g[j:])
    
    # This is 100 iterations of dot products of decreasing length. Total ops ~ N^2/2 = 5000. Very fast.
    grad_h = np.zeros(N)
    for j in range(N):
        grad_h[j] = np.dot(grad_corr[:N-j], g[j:])
        
    # Now account for d corr / d h via g.
    # We have grad_corr = d Obj / d corr.
    # d Obj / d h[j] = sum_k (d Obj / d corr[k]) * (d corr[k] / d h[j])
    # d corr[k] / d h[j] = g[j+k] (as derived).
    # So the loop above computes exactly sum_k grad_corr[k] * g[j+k].
    
    # Now, we also need the gradient w.r.t. the sum constraint penalty.
    # Penalty term: lambda_sum * (sum(h) - N/2)^2
    # d Penalty / d h[j] = 2 * lambda_sum * (sum(h) - N/2)
    sum_h = np.sum(h)
    penalty_grad = 2.0 * lambda_sum * (sum_h - N / 2.0)
    
    total_grad = grad_h + penalty_grad
    total_val = val_sm + lambda_sum * (sum_h - N / 2.0) ** 2
    
    return total_val, total_grad

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE

    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # We will run multiple optimizations from different starting points
    # with different beta (smoothness) and lambda_sum values.
    
    rng = np.random.RandomState(42)
    
    # Starting points:
    # 1. Uniform 0.5
    # 2. Random binary
    # 3. Random continuous
    # 4. Sawtooth
    # 5. Triangle
    
    starts = [np.full(N, 0.5)]
    
    # Random binary
    h_bin = rng.choice([0, 1], N)
    h_bin = _fix_sum(h_bin.astype(float))
    starts.append(h_bin)
    
    # Random continuous
    h_cont = rng.uniform(0, 1, N)
    h_cont = _fix_sum(h_cont)
    starts.append(h_cont)
    
    # Sawtooth
    x = np.arange(N)
    h_saw = (x / 10.0) % 1.0
    h_saw = _fix_sum(h_saw)
    starts.append(h_saw)
    
    # Triangle
    t = (x / 20.0) % 1.0
    h_tri = 1.0 - np.abs(2.0 * t - 1.0)
    h_tri = _fix_sum(h_tri)
    starts.append(h_tri)
    
    # Optimization parameters
    # Beta starts low (smooth) and increases (sharp)
    # Lambda_sum starts high (enforce constraint) and decreases? 
    # Actually, L-BFGS-B handles bounds [0,1]. We just need to handle sum.
    # A large lambda ensures sum is close to 50.
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    lambdas = [10.0, 50.0, 100.0]
    
    # To save time, we don't do a full grid. We prioritize higher betas (more accurate max)
    # and reasonable lambdas.
    
    # Strategy: 
    # 1. Start with a moderate beta and lambda, optimize.
    # 2. Use the result as start for higher beta.
    
    current_h = np.full(N, 0.5)
    
    # Phase 1: Coarse optimization
    for beta in [5.0, 10.0]:
        if deadline_check():
            break
        for lam in [10.0, 50.0]:
            if deadline_check():
                break
            res = minimize(
                _objective_and_grad,
                current_h,
                args=(beta, lam),
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success or res.fun < _score(current_h):
                current_h = res.x
                current_h = np.clip(current_h, 0, 1)
                current_h = _fix_sum(current_h)
                
    # Phase 2: Fine optimization with higher beta
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if deadline_check():
            break
        for lam in [50.0, 100.0]:
            if deadline_check():
                break
            res = minimize(
                _objective_and_grad,
                current_h,
                args=(beta, lam),
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < _score(current_h):
                candidate_h = res.x
                candidate_h = np.clip(candidate_h, 0, 1)
                candidate_h = _fix_sum(candidate_h)
                
                # Check validity and score
                if _is_valid(candidate_h):
                    s = _score(candidate_h)
                    if s < best_score:
                        best_score = s
                        best_h = candidate_h
                current_h = candidate_h
                
    # Try a few random restarts with high beta if time permits
    if not deadline_check():
        for _ in range(3):
            if deadline_check():
                break
            h_start = rng.uniform(0, 1, N)
            h_start = _fix_sum(h_start)
            
            res = minimize(
                _objective_and_grad,
                h_start,
                args=(100.0, 100.0),
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            candidate_h = res.x
            candidate_h = np.clip(candidate_h, 0, 1)
            candidate_h = _fix_sum(candidate_h)
            
            if _is_valid(candidate_h):
                s = _score(candidate_h)
                if s < best_score:
                    best_score = s
                    best_h = candidate_h
                    
    # Final safety check
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
