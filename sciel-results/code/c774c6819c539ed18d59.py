import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # i ranges 0..m, i.e., 0..99 (since m=99), y[i] and y[i+1] valid for y of length 101
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
    """Faster computation using FFT-based autocorrelation."""
    h = np.asarray(h, dtype=float)
    m = len(h)
    n = 2 * m - 1
    # Next power of 2 >= n for FFT
    nfft = 1
    while nfft < n:
        nfft <<= 1
    H = np.fft.rfft(h, nfft)
    g = np.fft.irfft(H * np.conj(H), nfft)[:n]
    m = len(g)
    # l2 = (w/3)*(2*sum(g^2) + sum(g[:-1]*g[1:]))
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value_fast(h)

def neg_value_direct(h):
    return -compute_value_fast(h)

def construct():
    deadline = time.time() + 145.0  # 5s safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 1.0

    # 1. Global Optimization using Differential Evolution
    # DE works on the log-heights to maintain positivity
    # Bounds for log_h: [-10, 10] is a wide range (e^-10 ~ 4e-5, e^10 ~ 2e4)
    # We can tighten this if needed, but wide is safer for global search.
    
    def objective_de(x):
        return neg_value(x)

    bounds = [(-10.0, 10.0)] * N
    
    # Run DE with a reasonable number of iterations
    # maxiter controls the number of generations
    # popsize controls the population size (multiplier of N)
    # We need to be careful with time. 
    # Each function evaluation is ~ O(N log N) due to FFT.
    # Let's estimate: 1000 generations * 50 (popsize) * 50 (N) ~ 2.5M operations? No, FFT is faster.
    # Let's try maxiter=200, popsize=10 (500 individuals).
    # This is a structural change from local restarts.
    
    try:
        de_res = differential_evolution(
            objective_de,
            bounds,
            maxiter=100,
            popsize=10,
            seed=42,
            polish=False, # We will polish manually
            tol=1e-8,
            init='latinhypercube'
        )
        
        h_de = np.exp(de_res.x)
        val_de = compute_value_fast(h_de)
        
        if val_de > best_val:
            best_val = val_de
            best_h = h_de.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")

    # 2. Local Polishing with L-BFGS-B
    # Start from the best DE result and also from the seed (uniform)
    # and a few other diverse starts to ensure we don't miss a better local optima
    
    starts = [
        np.zeros(N), # log(1) = 0
        np.log(best_h), # Best from DE
        # Add a few random smooth starts
    ]
    
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.randn(N)
        # Smooth it
        kernel = np.ones(5)/5.0
        x = np.convolve(x, kernel, mode='same')
        x = x - x.mean()
        x = 0.5 * x
        starts.append(x)
        
    for start_log_h in starts:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(
                neg_value,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception as e:
            pass

    # 3. Final Polish with Nelder-Mead if time permits
    # Nelder-Mead can be good for final fine-tuning
    if time.time() < deadline - 10.0:
        try:
            nm_res = minimize(
                neg_value_direct,
                best_h.copy(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            h_cand = np.maximum(nm_res.x, 0.0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception as e:
            pass

    return list(np.asarray(best_h, dtype=float))
