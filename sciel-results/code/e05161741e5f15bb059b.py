import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps_fft(h):
    """Compute overlap of h with shifted (1-h) for all circular shifts using FFT.
    overlap[k] = sum_i h[i] * (1 - h[(i+k) % N]) = N/2 - autocorr[k]
    """
    h = np.asarray(h, dtype=np.float64)
    # Zero-pad to avoid circular aliasing issues (though we want circular here)
    # For circular autocorrelation: autocorr[k] = sum_i h[i] * h[(i+k) % N]
    H = np.fft.fft(h, n=N)
    autocorr = np.fft.ifft(H * np.conj(H)).real[:N]
    overlaps = N / 2.0 - autocorr
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps_fft(h)
    return np.max(overlaps)

def _objective_lse(x, beta, lambda_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps_fft(h)
    
    # LogSumExp for smooth max approximation
    max_ov = np.max(overlaps)
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    # Penalty for sum constraint
    sum_penalty = lambda_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) < 1e-12:
        return h
    # Iterative projection
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / N
        h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple random restarts with L-BFGS-B, using simulated annealing
    # on the temperature (beta) to escape local optima
    
    # Generate diverse initial points
    initial_points = []
    
    # 1. Uniform (baseline)
    initial_points.append(np.full(N, 0.5))
    
    # 2. Random points in [0.1, 0.9]
    for i in range(30):
        h_rand = rng.uniform(0.1, 0.9, N)
        h_rand = _project_to_feasible(h_rand)
        initial_points.append(h_rand)
    
    # 3. Structured patterns
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7]:
        h_pat = 0.5 + 0.4 * np.sin(freq * t)
        h_pat = _project_to_feasible(h_pat)
        initial_points.append(h_pat)
    
    # 4. Step patterns
    for split in [20, 40, 60, 80]:
        h_step = np.zeros(N)
        h_step[:split] = 0.8
        h_step[split:] = 0.2
        h_step = _project_to_feasible(h_step)
        initial_points.append(h_step)
    
    # 5. Binary-like patterns
    for seed in range(10):
        rng_local = np.random.default_rng(seed)
        h_bin = rng_local.choice([0.1, 0.9], N)
        h_bin = _project_to_feasible(h_bin)
        initial_points.append(h_bin)
    
    # Optimize each initial point with varying beta
    for idx, h_init in enumerate(initial_points):
        if time.time() > deadline:
            break
        
        h_init = np.array(h_init, dtype=float)
        h_init = _project_to_feasible(h_init)
        
        # Try multiple beta values (annealing schedule)
        betas = [5.0, 10.0, 20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            lambda_sum = 100.0
            obj_func = lambda x: _objective_lse(x, beta, lambda_sum)
            
            try:
                res = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                h_opt = _project_to_feasible(res.x)
                score = _max_overlap(h_opt)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt
                    # Use this as new starting point for higher beta
                    h_init = h_opt.copy()
                    
            except Exception:
                pass
        
        # Occasionally refresh h_init with a new random point
        if idx % 5 == 4 and time.time() < deadline - 10:
            h_init = rng.uniform(0.1, 0.9, N)
            h_init = _project_to_feasible(h_init)
    
    # Final refinement with very high beta
    if time.time() < deadline - 5.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline:
                break
            lambda_sum = 1000.0
            obj_func = lambda x: _objective_lse(x, beta, lambda_sum)
            
            try:
                res = minimize(
                    obj_func,
                    best_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = _project_to_feasible(res.x)
                score = _max_overlap(h_opt)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt
                    
            except Exception:
                pass
    
    return best_h
