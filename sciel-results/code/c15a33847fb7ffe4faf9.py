"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _autoconv_max_abs(x):
    """Compute max absolute autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    conv = np.fft.irfft(f * f, n=2*n)[:n]
    return np.max(np.abs(conv))

def _squared_integral(x):
    """Compute squared integral (sum of squares for discrete)."""
    return np.sum(x**2)

def _objective(x):
    """Objective: max_abs_autoconv / squared_integral. Lower is better."""
    denom = _squared_integral(x)
    if denom < 1e-10:
        return 1e10
    return _autoconv_max_abs(x) / denom

def _construct_from_blocks(num_blocks, N):
    """
    Construct a signal from num_blocks rectangular blocks.
    Each block has: center, width, amplitude.
    Returns the constructed signal.
    """
    x = np.zeros(N)
    for i in range(num_blocks):
        center = params[i*3]
        width = params[i*3+1]
        amp = params[i*3+2]
        
        # Block spans [center - width/2, center + width/2]
        start = int(np.floor(center - width/2))
        end = int(np.ceil(center + width/2))
        
        # Clip to valid range
        start = max(0, start)
        end = min(N, end)
        
        if start < end:
            x[start:end] += amp
    return x

def _block_objective(params, N):
    """
    Objective function for block-based parameter optimization.
    params: [center0, width0, amp0, center1, width1, amp1, ...]
    """
    global _current_N
    num_blocks = len(params) // 3
    x = np.zeros(_current_N)
    for i in range(num_blocks):
        center = params[i*3]
        width = params[i*3+1]
        amp = params[i*3+2]
        
        start = int(np.floor(center - width/2))
        end = int(np.ceil(center + width/2))
        
        start = max(0, start)
        end = min(_current_N, end)
        
        if start < end:
            x[start:end] += amp
    
    denom = np.sum(x**2)
    if denom < 1e-10:
        return 1e10
    return _autoconv_max_abs(x) / denom

def construct():
    """
    Construct a signed step function minimizing max_abs_autoconv / squared_integral.
    Uses differential evolution to optimize block parameters.
    """
    global _current_N
    _current_N = N
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_score = 1.0  # Baseline: ones signal
    best_x = np.ones(N)
    
    # Try different numbers of blocks
    for num_blocks in [1, 2, 3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        
        # Bounds for each parameter: center in [0, N], width in [1, N], amp in [-2, 2]
        bounds = []
        for _ in range(num_blocks):
            bounds.append((0, N))       # center
            bounds.append((1, N))       # width
            bounds.append((-2, 2))      # amplitude
        
        # Initial population
        pop_size = 20
        init_pop = np.random.uniform(bounds[0][0], bounds[0][1], size=(pop_size, len(bounds)))
        for j in range(len(bounds)):
            init_pop[:, j] = np.random.uniform(bounds[j][0], bounds[j][1], size=pop_size)
        
        try:
            result = differential_evolution(
                _block_objective,
                bounds,
                maxiter=50,
                popsize=pop_size,
                tol=1e-6,
                seed=42,
                init=init_pop,
                polish=False,
                updating='deferred'
            )
            
            if result.success or result.fun < best_score:
                x = np.zeros(N)
                for i in range(num_blocks):
                    center = result.x[i*3]
                    width = result.x[i*3+1]
                    amp = result.x[i*3+2]
                    
                    start = int(np.floor(center - width/2))
                    end = int(np.ceil(center + width/2))
                    
                    start = max(0, start)
                    end = min(N, end)
                    
                    if start < end:
                        x[start:end] += amp
                
                score = _objective(x)
                if score < best_score:
                    best_score = score
                    best_x = x
        except Exception:
            continue
    
    # If we have time, try a local polish on the best solution
    if time.time() < deadline - 5 and best_score < 1.0:
        try:
            # Convert to continuous representation for local optimization
            # Use the block representation and refine with local search
            num_blocks = 3
            # Extract current best block parameters (approximate)
            # For simplicity, do a fine-grained local search around current best
            x_float = best_x.astype(float)
            
            # Local search: perturb the signal slightly
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                # Random perturbation
                perturbation = np.random.normal(0, 0.1, N)
                x_trial = best_x + perturbation
                score_trial = _objective(x_trial)
                if score_trial < best_score:
                    best_score = score_trial
                    best_x = x_trial
        except Exception:
            pass
    
    # Ensure the result is a valid sequence
    if not isinstance(best_x, np.ndarray) or len(best_x) != N:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
