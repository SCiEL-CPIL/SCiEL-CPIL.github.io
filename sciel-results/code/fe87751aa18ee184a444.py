"""
Strategy: Direct multi-start optimization with smooth surrogate for nonsmooth norm_inf.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so the scale of h
does not matter, only the shape. We use a smooth log-sum-exp approximation for the max
during optimization to help L-BFGS-B converge better, then compute the true value.

Key improvements:
1. Smooth surrogate objective using log-sum-exp for norm_inf during optimization.
2. More diverse and structured initializations.
3. Time-budgeted search with adaptive refinement.
4. High-precision polish pass on the best solution found.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the true value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    
    h = np.maximum(0, h)  # enforce non-negativity
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def compute_value_smooth(h, beta=100.0):
    """Compute a smooth surrogate of the value metric using log-sum-exp for norm_inf."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    
    h = np.maximum(0, h)  # enforce non-negativity
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Smooth approximation of max using log-sum-exp
    abs_g = np.abs(g)
    max_abs = np.max(abs_g)
    if max_abs < 1e-15:
        return 0.0
    norm_inf_smooth = max_abs * np.log(np.sum(np.exp((abs_g - max_abs) * beta))) / beta
    
    if norm_1 == 0 or norm_inf_smooth == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf_smooth)
    return value

def _generate_initializations():
    """Generate diverse initial height profiles."""
    x = np.linspace(0, N - 1, N)
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Gaussian centered
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/4))**2))
    
    # Bimodal: two peaks
    peak1 = np.exp(-0.5 * ((x - N*0.25) / (N/8))**2)
    peak2 = np.exp(-0.5 * ((x - N*0.75) / (N/8))**2)
    inits.append(peak1 + peak2)
    
    # Plateau
    plateau = np.zeros(N)
    plateau[N//4 : 3*N//4] = 1.0
    inits.append(plateau)
    
    # Cosine bump
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # Triangular
    inits.append(1 - 2 * np.abs(x - (N-1)/2) / (N-1))
    
    # Semicircle
    inits.append(np.sqrt(np.maximum(0, 1 - ((2*x/(N-1) - 1))**2)))
    
    # Sharp central peak
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/10))**2))
    
    # Broad flat top with edges
    inits.append(np.maximum(0, 1 - np.abs(x - (N-1)/2) / (N/3)))
    
    # Step-like: high in center, low at edges
    step = np.ones(N)
    step[:N//5] = 0.2
    step[-N//5:] = 0.2
    inits.append(step)
    
    # Double step
    ds = np.ones(N)
    ds[:N//8] = 0.1
    ds[-N//8:] = 0.1
    ds[N//3 : 2*N//3] = 1.5
    inits.append(ds)
    
    # Random smooth profiles (a few)
    rng = np.random.RandomState(42)
    for _ in range(15):
        # Generate random smooth profile via random Fourier coefficients
        freqs = rng.randint(1, 10, size=8)
        phases = rng.uniform(0, 2*np.pi, size=8)
        amps = rng.uniform(0.3, 1.0, size=8)
        profile = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            profile += a * np.cos(2 * np.pi * f * x / N + p)
        profile = np.maximum(0.05, profile)
        inits.append(profile)
    
    # More random smooth profiles
    for _ in range(15):
        # Piecewise linear random
        points = np.sort(rng.uniform(0, N, size=7))
        heights = rng.uniform(0.3, 2.0, size=7)
        profile = np.interp(x, points, heights)
        inits.append(profile)
    
    # Additional structured profiles
    # Narrow central peak with side lobes
    narrow = np.exp(-0.5 * ((x - N/2) / (N/12))**2)
    side1 = 0.3 * np.exp(-0.5 * ((x - N*0.2) / (N/10))**2)
    side2 = 0.3 * np.exp(-0.5 * ((x - N*0.8) / (N/10))**2)
    inits.append(narrow + side1 + side2)
    
    # Cosine window with different frequency
    inits.append(0.5 * (1 + np.cos(4 * np.pi * x / N)))
    
    # Exponential decay from center
    inits.append(np.exp(-np.abs(x - N/2) / (N/4)))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def objective_smooth(h):
        """Smooth surrogate objective using log-sum-exp for norm_inf."""
        h = np.maximum(0, h)
        val = compute_value_smooth(h, beta=100.0)
        if val <= 0:
            return 1e10
        return -val
    
    def objective_true(h):
        """True objective using exact max."""
        h = np.maximum(0, h)
        val = compute_value(h)
        if val <= 0:
            return 1e10
        return -val
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Evaluate the seed
    seed_val = compute_value(np.ones(N))
    best_val = seed_val
    best_h = np.ones(N).copy()
    
    inits = _generate_initializations()
    
    # First pass: evaluate all initializations directly
    for init in inits:
        if time.time() > deadline - 15:
            break
        init = np.asarray(init, dtype=float)
        val = compute_value(init)
        if val > best_val:
            best_val = val
            best_h = init.copy()
    
    # Second pass: refine each initialization with L-BFGS-B using smooth surrogate
    # Sort by initial value descending to prioritize promising ones
    init_vals = [(compute_value(np.asarray(init, dtype=float)), init) for init in inits]
    init_vals.sort(key=lambda x: -x[0])
    
    # Also add some perturbed versions of the best initializations
    extra_starts = []
    best_init = init_vals[0][1]
    rng = np.random.RandomState(123)
    for _ in range(20):
        perturbed = best_init + rng.normal(0, 0.3, N)
        perturbed = np.maximum(0, perturbed)
        extra_starts.append(perturbed)
    
    # Combine all starts for refinement
    all_starts = [iv[1] for iv in init_vals] + extra_starts
    
    for idx, init in enumerate(all_starts):
        if time.time() > deadline - 20:
            break
        
        init = np.asarray(init, dtype=float).copy()
        
        # If this start already exceeds best, refine it
        val_start = compute_value(init)
        
        try:
            res = minimize(
                objective_smooth,
                init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 400,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            h_opt = np.maximum(0, res.x)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Third pass: refine the best solution found further with true objective
    if time.time() < deadline - 10:
        try:
            # Try multiple random perturbations of the best solution
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Small perturbation
                perturbed = best_h + np.random.normal(0, 0.05, N)
                perturbed = np.maximum(0, perturbed)
                
                try:
                    res = minimize(
                        objective_true,
                        perturbed,
                        method='L-BFGS-B',
                        bounds=[(0, None)] * N,
                        options={
                            'maxiter': 300,
                            'ftol': 1e-12,
                            'gtol': 1e-8
                        }
                    )
                    
                    h_opt = np.maximum(0, res.x)
                    val_opt = compute_value(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Fourth pass: high-precision polish on the best solution
    if time.time() < deadline - 3:
        try:
            # Very tight tolerance polish
            res = minimize(
                objective_true,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-10
                }
            )
            
            h_opt = np.maximum(0, res.x)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: ensure non-negativity and finiteness
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(f"Score: {compute_value(h)}")
