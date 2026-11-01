"""
Continuous Spectral Shaping with Discrete Projection.
Strategy: 
1. Relax the discrete {+1, -1} constraint to continuous [-1, 1].
2. Use L-BFGS-B to minimize a smoothed maximum sidelobe objective.
   This navigates the non-convex landscape more globally than discrete search.
3. Project the continuous solution to the nearest discrete {+1, -1} sequence.
4. Perform a few steps of fast discrete local repair to polish the result.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _compute_acf(h):
    """Compute the circular autocorrelation of h using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return acf

def _objective_continuous(x):
    """
    Objective function for continuous optimization.
    We want to minimize the maximum absolute sidelobe.
    Since max is non-smooth, we use a soft-max approximation or just the max
    if we accept the kinks (L-BFGS-B can handle non-smooth to some extent,
    but a soft approximation is safer for gradient-based methods).
    However, for N=400, the max is often well-defined. 
    Let's try a simple max first, and if it fails, use a log-sum-exp.
    Actually, L-BFGS-B requires gradients. We can use finite differences,
    but that's slow. 
    Alternative: Minimize the sum of squares of sidelobes above a threshold?
    Or minimize the 95th percentile?
    
    Let's stick to minimizing the max absolute sidelobe, but use a 
    differentiable surrogate: 
    f(x) = log(sum(exp(beta * |acf[k]|))) / beta
    This approximates the max as beta -> infinity.
    We also need to penalize values going out of [-1, 1] if we don't use bounds.
    L-BFGS-B supports bounds.
    """
    # Ensure x is real
    acf = _compute_acf(x)
    sidelobes = np.abs(acf[1:])
    
    # Soft max approximation
    beta = 50.0
    # Clip to avoid overflow in exp
    shifted = sidelobes - np.max(sidelobes)
    soft_max = np.max(sidelobes) + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    return soft_max

def _gradient_continuous(x):
    """
    Numerical gradient for the objective.
    Since analytic gradient of soft-max-of-FFT is complex, we use finite differences.
    To save time, we use a sparse gradient or just let scipy handle it with '2-point'.
    However, 400 dimensions * 2 evaluations per step is 800 FFTs.
    Each FFT is ~1024 size. 800 * 1024^2 ~ 8e8 ops. This might be too slow for many iterations.
    
    Better approach: Use the fact that we are projecting to +/-1 at the end.
    Maybe we can just use a random initial point and let L-BFGS-B do its thing
    with a limited number of iterations.
    
    Actually, let's try a simpler objective: Minimize the sum of squared sidelobes.
    This is differentiable and smooth. It doesn't directly minimize the max,
    but it flattens the spectrum, which often correlates with low max sidelobes.
    Then we can do a final discrete repair for the max.
    
    Let's try minimizing the variance of the sidelobes or the sum of squares.
    """
    acf = _compute_acf(x)
    sidelobes = acf[1:]
    # Minimize sum of squares of sidelobes
    return np.sum(sidelobes**2)

def _gradient_sumsq(x):
    """
    Gradient of sum of squared sidelobes.
    acf[k] = sum_j x[j] x[j+k]
    d/dx[i] (sum_k acf[k]^2) = 2 * sum_k acf[k] * d(acf[k])/dx[i]
    d(acf[k])/dx[i] = x[i+k] + x[i-k] (for circular)
    
    This can be computed efficiently using FFT or direct convolution.
    Let's use FFT for speed.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft).real[:N]
    
    # We need sum_k acf[k] * (x[i+k] + x[i-k])
    # This is a convolution of acf with x?
    # Let g[i] = sum_k acf[k] * x[i+k] + sum_k acf[k] * x[i-k]
    # Note acf is even.
    # g[i] = 2 * sum_k acf[k] * x[i+k] (if we consider symmetric sum)
    # Actually, let's just compute it directly for N=400, it's fast enough.
    # Or use FFT:
    # conv(acf, x) gives sum_j acf[j] x[i-j].
    
    # Let's use the direct method for clarity and sufficient speed (400*400 = 160k ops)
    grad = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in range(1, N):
            # d(acf[k])/dx[i] = x[(i+k)%N] + x[(i-k)%N]
            # Note: acf[k] = acf[N-k]
            # The term for lag k and N-k are distinct in the sum k=1..N-1
            s += acf[k] * (x[(i+k) % N] + x[(i-k) % N])
            # Wait, the derivative of acf[k] w.r.t x[i] is x[i+k] + x[i-k].
            # The objective is sum_k acf[k]^2.
            # d/dx[i] sum_k acf[k]^2 = sum_k 2 * acf[k] * (x[i+k] + x[i-k])
            # So grad[i] = 2 * s
        grad[i] = 2.0 * s
        
    return grad

def _local_repair_fast(h, deadline):
    """
    Fast discrete local repair using incremental ACF updates.
    """
    h = h.copy()
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Initial ACF
    F = np.fft.rfft(h, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft).real[:N]
    
    best_val = np.max(np.abs(acf[1:]))
    best_h = h.copy()
    
    # Limit iterations to ensure we finish in time
    max_iters = 200
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        max_lag = np.argmax(np.abs(acf[1:])) + 1
        current_worst_val = np.abs(acf[max_lag])
        
        improved = False
        # Try flipping bits in a random order to avoid bias
        # But for speed, let's just scan 0..N-1
        for i in range(N):
            h_i = h[i]
            t1 = h[(i + max_lag) % N]
            t2 = h[(i - max_lag) % N]
            delta = -2.0 * h_i * (t1 + t2)
            
            new_acf_lag = acf[max_lag] + delta
            
            if abs(new_acf_lag) < current_worst_val:
                # Perform flip
                h[i] *= -1.0
                h_old_i = -h[i]
                
                # Update ACF
                for l in range(1, N):
                    t1 = h[(i + l) % N]
                    t2 = h[(i - l) % N]
                    delta_l = -2.0 * h_old_i * (t1 + t2)
                    acf[l] += delta_l
                    acf[N-l] += delta_l
                    
                improved = True
                break
        
        if not improved:
            # Stuck in local min. Perturb.
            num_flips = 5
            flip_indices = np.random.choice(N, num_flips, replace=False)
            h[flip_indices] *= -1.0
            
            # Recompute ACF
            F = np.fft.rfft(h, n=n_fft)
            acf = np.fft.irfft(F * np.conj(F), n=n_fft).real[:N]
            
        current_val = np.max(np.abs(acf[1:]))
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
    return best_h, best_val

def construct():
    """
    Construct a signed step function that minimizes the autoconvolution upper bound.
    """
    start_time = time.time()
    # Leave 10s for final steps and safety
    deadline = start_time + 140.0 
    
    np.random.seed(42)
    
    # 1. Continuous Relaxation
    # Start with a random continuous vector in [-1, 1]
    x0 = np.random.uniform(-1.0, 1.0, size=N)
    
    # Bounds
    bounds = [(-1.0, 1.0)] * N
    
    # Minimize sum of squared sidelobes
    # Use L-BFGS-B
    # maxiter is limited to ensure we don't run out of time
    # Each iteration involves gradient evaluation (O(N^2) or O(N log N))
    # N=400, O(N^2)=160k. 100 iters = 16M ops. Fast.
    
    try:
        res = minimize(
            _objective_continuous, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    # 2. Discrete Projection
    # Round to nearest +/-1
    h_discrete = np.sign(x_opt)
    # Handle zeros (unlikely with uniform init)
    h_discrete[h_discrete == 0] = 1.0
    
    # 3. Discrete Local Repair
    h_best, val_best = _local_repair_fast(h_discrete, deadline)
    
    return h_best

# EVOLVE-BLOCK-END
