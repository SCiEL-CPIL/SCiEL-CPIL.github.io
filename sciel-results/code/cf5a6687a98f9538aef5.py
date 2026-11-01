import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(x, beta):
    """
    Smooth approximation of max(|x|).
    Uses log-sum-exp style for positive parts, but we need max of absolute values.
    max(|x|) approx exp( (1/beta) * log( sum exp(beta * |x|) ) )
    However, to avoid overflow, we can use:
    m = max(|x|)
    return m + (1/beta) * log( sum exp(beta * (|x| - m)) )
    """
    abs_x = np.abs(x)
    m = np.max(abs_x)
    if m == 0:
        return 0.0
    # For numerical stability
    return m + np.log(np.sum(np.exp(beta * (abs_x - m)))) / beta

def _objective_and_grad(x, beta):
    """
    Objective: smooth max of autocorrelation sidelobes.
    x is a continuous vector of length N.
    We want to minimize max_{k!=0} |R_k|.
    R_k = sum_i x[i] x[i+k].
    
    Gradient:
    Let M = max_k |R_k|.
    The subgradient of max |R_k| w.r.t x is a convex combination of gradients of |R_k| for active k.
    For a smooth approximation, we use softmax weights.
    
    Actually, for L-BFGS-B, we can just compute the gradient of the soft max.
    Let f(x) = soft_max(|R(x)|, beta).
    df/dx = sum_k w_k * d|R_k|/dx
    where w_k = exp(beta * |R_k|) / sum_j exp(beta * |R_j|).
    
    d|R_k|/dx:
    If R_k > 0: dR_k/dx
    If R_k < 0: -dR_k/dx
    
    dR_k/dx_i:
    R_k = sum_j x_j x_{j+k} (with appropriate bounds).
    dR_k/dx_i = x_{i+k} + x_{i-k} (if indices valid).
    Specifically, for autocorrelation R_k = sum_{j=0}^{N-1-k} x_j x_{j+k} for k>0.
    And R_{-k} = R_k for real x? No, R_k = sum x_j x_{j+k}. R_{-k} = sum x_j x_{j-k} = sum x_{j+k} x_j = R_k.
    So R is symmetric. We only need to consider k > 0.
    
    Let's compute R for k=1..N-1.
    R_k = sum_{j=0}^{N-1-k} x_j x_{j+k}.
    dR_k/dx_i = 
      If 0 <= i <= N-1-k: x_{i+k}
      If k <= i <= N-1: x_{i-k}
      Both can be true if i is in [k, N-1-k]. Then x_{i+k} + x_{i-k}.
    
    Let's vectorize.
    R = np.array([np.dot(x[:N-k], x[k:]) for k in range(1, N)])
    
    Gradient of R_k w.r.t x:
    g_k = np.zeros(N)
    # Term 1: x_{i+k} for i in 0..N-1-k
    g_k[:N-k] += x[k:]
    # Term 2: x_{i-k} for i in k..N-1
    g_k[k:] += x[:N-k]
    
    This is O(N^2) per gradient evaluation.
    N=400, so 160,000 ops.
    L-BFGS-B might call this ~100-500 times.
    500 * 160,000 = 80,000,000 ops.
    This is feasible in Python with NumPy if optimized.
    
    Let's check if we can do faster.
    R_k = sum x_j x_{j+k}.
    This is cross-correlation.
    Can compute all R_k via FFT.
    But gradient is still local.
    
    Let's try the direct method first and see if it's fast enough.
    """
    # Compute R_k for k=1..N-1
    R = np.array([np.dot(x[:N-k], x[k:]) for k in range(1, N)])
    
    # Soft max
    abs_R = np.abs(R)
    m = np.max(abs_R)
    if m == 0:
        f = 0.0
        w = np.ones(N-1) / (N-1)
    else:
        # Numerical stability
        shifted = beta * (abs_R - m)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        f = m + np.log(sum_exp) / beta
        w = exp_shifted / sum_exp
    
    # Gradient
    # g = sum_k w_k * sign(R_k) * dR_k/dx
    grad = np.zeros(N)
    
    # Vectorize the accumulation?
    # dR_k/dx_i = x_{i+k} + x_{i-k}
    # We need to accumulate w_k * sign(R_k) * (x_{i+k} + x_{i-k})
    
    # This loop is O(N^2).
    # Let's try to optimize.
    # For each i, grad[i] = sum_k w_k * sign(R_k) * (x_{i+k} + x_{i-k})
    # This looks like a convolution of w_k * sign(R_k) with x?
    # Let c_k = w_k * sign(R_k) for k=1..N-1.
    # c is symmetric? No, R_k is symmetric, but w_k depends on |R_k|, so w_k is symmetric.
    # sign(R_k) is symmetric. So c_k is symmetric.
    # grad[i] = sum_{k=1}^{N-1} c_k * (x_{i+k} + x_{i-k})
    # Note that x_{i+k} is only defined if i+k < N.
    # x_{i-k} is only defined if i-k >= 0.
    
    # This is exactly the convolution of c (padded) with x, but with boundary effects.
    # Let's define c_full of length 2N-1.
    # c_full[k+N-1] = c_k for k=1..N-1.
    # c_full[N-1] = 0 (k=0).
    # c_full[N-1-k] = c_k for k=1..N-1.
    
    c = np.zeros(2*N-1)
    # k=1..N-1
    # Index in c_full for lag k is N-1+k
    # Index in c_full for lag -k is N-1-k
    for k in range(1, N):
        val = w[k-1] * np.sign(R[k-1])
        c[N-1+k] = val
        c[N-1-k] = val
    
    # Now convolve c with x?
    # (c * x)[n] = sum_m c[m] x[n-m]
    # We want grad[i] = sum_k c_{N-1+k} x[i+k] + c_{N-1-k} x[i-k]
    # This is not a standard convolution because the kernel is centered at N-1.
    # Let's shift c.
    # Let d[m] = c[m + N-1]. So d[0] = c[N-1] = 0.
    # d[k] = c[N-1+k] = val_k.
    # d[-k] = c[N-1-k] = val_k.
    # So d is symmetric.
    # grad[i] = sum_{k} d[k] x[i+k]
    # This is a correlation of d and x?
    # Correlation (d ⋆ x)[i] = sum_k d[k] x[i+k].
    # Yes.
    
    # So we can compute grad via np.correlate(d, x, 'full')?
    # np.correlate(d, x) computes sum d[m] x[n-m]? No.
    # np.correlate(in1, in2) computes sum in1[m] in2[n-m].
    # We want sum d[k] x[i+k].
    # Let in1 = d, in2 = x.
    # We want sum_k d[k] x[i+k].
    # This is equivalent to convolution of d_rev and x?
    # Or just use FFT.
    
    # Simpler: just loop. N=400 is small.
    # But let's try the FFT method to be safe on time.
    
    # Pad d to length 2N-1?
    # d is defined for k in -(N-1)..(N-1).
    # Let's create an array D of length 2N-1.
    # D[N-1+k] = d[k].
    # Then grad = np.convolve(x, D, 'full')?
    # (x * D)[n] = sum_i x[i] D[n-i].
    # We want grad[i] = sum_k d[k] x[i+k].
    # Let j = i+k. k = j-i.
    # grad[i] = sum_j x[j] d[j-i].
    # This is (x * d_rev)[i] where d_rev[k] = d[-k] = d[k] (since symmetric).
    # So grad = np.convolve(x, d, 'full')?
    # Let's check indices.
    # (x * d)[n] = sum_j x[j] d[n-j].
    # We want grad[i] = sum_j x[j] d[j-i].
    # So grad[i] = (x * d)[i] if d is symmetric?
    # No. d[j-i] vs d[n-j].
    # If n=i, sum_j x[j] d[i-j].
    # We want sum_j x[j] d[j-i].
    # Since d is symmetric, d[i-j] = d[j-i].
    # So yes, grad[i] = (x * d)[i].
    # But the length?
    # x is length N. d is length 2N-1 (centered).
    # The convolution will be length N + 2N-1 - 1 = 3N-2.
    # We want grad of length N.
    # The valid part is in the middle.
    # Let's just do the loop for simplicity and correctness, N=400 is fast.
    
    # Loop version
    for k in range(1, N):
        val = w[k-1] * np.sign(R[k-1])
        if val == 0:
            continue
        # grad[:N-k] += val * x[k:]
        # grad[k:] += val * x[:N-k]
        grad[:N-k] += val * x[k:]
        grad[k:] += val * x[:N-k]
        
    return f, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Initial random guesses
    num_initial = 20
    for _ in range(num_initial):
        if time.time() > deadline - 5:
            break
        h = np.random.choice([-1, 1], size=N)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    # Gradient-based optimization
    # Start from best_h
    x0 = best_h.astype(float)
    
    beta = 5.0 # Temperature for soft max
    
    # L-BFGS-B
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 200}
        )
        if time.time() < deadline - 5:
            x_opt = res.x
            # Round to nearest +/- 1
            h_opt = np.sign(x_opt)
            # Ensure no zeros
            h_opt[h_opt == 0] = 1
            score_opt = _calc_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Local search on the result
    if time.time() < deadline - 5:
        h = best_h
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = h[j]
                h[j] = -h[j]
                new_score = _calc_score(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[j] = old_val
            
            if not improved:
                # Perturb
                num_perturb = 5
                for _ in range(num_perturb):
                    j = np.random.randint(N)
                    h[j] = -h[j]
                current_score = _calc_score(h)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                else:
                    # If perturbation didn't help, stop? Or continue?
                    # Let's stop if we've been stuck.
                    break

    return best_h
