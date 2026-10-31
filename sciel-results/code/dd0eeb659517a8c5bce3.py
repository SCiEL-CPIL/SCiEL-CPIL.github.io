import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max."""
    x_max = np.max(x)
    exp_terms = np.exp(alpha * (x - x_max))
    weights = exp_terms / np.sum(exp_terms)
    return weights

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(10):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Create initial two-block solution."""
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h = np.zeros(N)
    half = N // 2
    h[:half] = A
    h[half:] = B
    return h

def _sinusoidal_h(params):
    """Generate h from sinusoidal parameters: [A, freq, phase, offset]"""
    A, freq, phase, offset = params
    t = np.linspace(0, 2, N, endpoint=False)
    h = offset + A * np.sin(2 * np.pi * freq * t + phase)
    return h

def _sinusoidal_score(params):
    """Score for sinusoidal parameterization."""
    h = _sinusoidal_h(params)
    h = _project_to_feasible(h)
    return _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = _two_block_init()
    best_score = _score(best_h)
    
    # Strategy 1: Low-dimensional sinusoidal parameterization with global search
    # Parameters: [amplitude, frequency, phase, offset]
    # h(t) = offset + amplitude * sin(2*pi*freq*t + phase)
    
    def _sinusoidal_objective(params):
        if time.time() > deadline - 5.0:
            return 1.0
        A, freq, phase, offset = params
        # Bounds: A in [0, 0.5], freq in [0.5, 10], phase in [0, 2*pi], offset in [0, 1]
        A = np.clip(A, 0.0, 0.5)
        freq = np.clip(freq, 0.5, 10.0)
        phase = np.clip(phase, 0.0, 2*np.pi)
        offset = np.clip(offset, 0.0, 1.0)
        
        t = np.linspace(0, 2, N, endpoint=False)
        h = offset + A * np.sin(2 * np.pi * freq * t + phase)
        h = _project_to_feasible(h)
        
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=50.0)
        return smax
    
    # Multi-start sinusoidal optimization
    np.random.seed(42)
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        A0 = np.random.uniform(0.1, 0.4)
        freq0 = np.random.uniform(0.5, 5.0)
        phase0 = np.random.uniform(0, 2*np.pi)
        offset0 = np.random.uniform(0.2, 0.8)
        
        x0 = np.array([A0, freq0, phase0, offset0])
        bounds = [(0.0, 0.5), (0.5, 10.0), (0.0, 2*np.pi), (0.0, 1.0)]
        
        try:
            result = minimize(
                _sinusoidal_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.fun < best_score:
                h_opt = _sinusoidal_h(result.x)
                h_opt = _project_to_feasible(h_opt)
                current_score = _score(h_opt)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    
        except Exception:
            continue
    
    # Strategy 2: Two-block with smooth transition
    # Parameterize: [A, B, transition_center, transition_width]
    # h[i] = A for i < center - width/2, B for i > center + width/2
    # Smooth transition in between using sigmoid
    
    def _two_block_smooth_h(params):
        A, B, center, width = params
        A = np.clip(A, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)
        center = np.clip(center, 0.0, N)
        width = np.clip(width, 1.0, N)
        
        i = np.arange(N)
        # Sigmoid transition
        x = (i - (center - width/2)) / (width/4 + 1e-10)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        h = A + (B - A) * sigmoid
        return h
    
    def _two_block_smooth_objective(params):
        if time.time() > deadline - 5.0:
            return 1.0
        h = _two_block_smooth_h(params)
        h = _project_to_feasible(h)
        
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=50.0)
        return smax
    
    # Optimize two-block smooth from the golden ratio solution
    A_gold = (3.0 - math.sqrt(5.0)) / 2.0
    B_gold = 1.0 - A_gold
    
    x0_tb = np.array([A_gold, B_gold, N/2.0, 5.0])
    bounds_tb = [(0.0, 1.0), (0.0, 1.0), (0.0, float(N)), (1.0, float(N))]
    
    try:
        result_tb = minimize(
            _two_block_smooth_objective,
            x0_tb,
            method='L-BFGS-B',
            bounds=bounds_tb,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        
        if result_tb.fun < best_score:
            h_opt = _two_block_smooth_h(result_tb.x)
            h_opt = _project_to_feasible(h_opt)
            current_score = _score(h_opt)
            if current_score < best_score:
                best_score = current_score
                best_h = h_opt.copy()
                
    except Exception:
        pass
    
    # Strategy 3: Full optimization with multiple restarts from different periodic initializations
    periods = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 100]
    
    for P in periods:
        if time.time() > deadline - 15.0:
            break
        
        # Initialize period
        inits = []
        inits.append(np.full(P, 0.5))
        t = np.linspace(0, 2*np.pi, P, endpoint=False)
        inits.append(0.5 + 0.4 * np.sin(t))
        np.random.seed(42 + P)
        inits.append(np.random.rand(P))
        
        for h0_period in inits:
            if time.time() > deadline - 15.0:
                break
            
            def objective_period(params):
                if time.time() > deadline - 10.0:
                    return 1.0
                p = params
                reps = int(np.ceil(N / P))
                h_full = np.tile(p, reps)[:N]
                h_full = _project_to_feasible(h_full)
                
                shifted = np.zeros(N)
                shifted[:-1] = h_full[1:]
                overlap = h_full * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=50.0)
                return smax
            
            bounds_p = [(0.0, 1.0)] * P
            
            try:
                result_p = minimize(
                    objective_period,
                    h0_period,
                    method='L-BFGS-B',
                    bounds=bounds_p,
                    options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
                )
                
                if result_p.fun < best_score:
                    p_opt = result_p.x
                    reps = int(np.ceil(N / P))
                    h_full = np.tile(p_opt, reps)[:N]
                    h_full = _project_to_feasible(h_full)
                    current_score = _score(h_full)
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h_full.copy()
                        
            except Exception:
                continue
    
    # Final polish on the best solution
    if time.time() < deadline - 20.0:
        def objective_full(params):
            if time.time() > deadline - 10.0:
                return 1.0
            h_full = params
            h_full = _project_to_feasible(h_full)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_full[1:]
            overlap = h_full * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=100.0)
            return smax
        
        bounds_full = [(0.0, 1.0)] * N
        
        try:
            result_full = minimize(
                objective_full,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            h_opt = result_full.x
            h_opt = _project_to_feasible(h_opt)
            new_score = _score(h_opt)
            if new_score < best_score:
                best_h = h_opt
                best_score = new_score
        except Exception:
            pass
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    return best_h
