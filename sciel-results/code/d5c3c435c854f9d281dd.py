"""Minimize the normalized max autoconvolution of a nonnegative 600-step function on [-1/4,1/4]."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 600
HALF = N // 2          # even, symmetric about 0
DT = (0.5) / N         # interval width

# EVOLVE-BLOCK-START

def _fft_peak(a):
    """Peak of normalized autoconvolution (a * a) / ||a||_2^2."""
    a = np.asarray(a, dtype=float)
    nrm = np.sqrt(np.dot(a, a))
    if nrm <= 0:
        return 0.0
    conv = fftconvolve(a, a, mode="full")
    return float(np.max(conv) / (nrm * nrm))


def _boxcar_param_to_signal(params):
    """
    Convert boxcar parameters to signal.
    params = [center1, width1, amp1, center2, width2, amp2, ...]
    Each boxcar is a rectangular pulse.
    """
    y = np.zeros(N)
    idx = np.arange(N)
    m = idx - HALF
    x = m * DT
    n_boxes = len(params) // 3
    for i in range(n_boxes):
        c = params[3*i]
        w = abs(params[3*i+1]) + 1e-10
        amp = params[3*i+2]
        mask = (x >= c - w/2) & (x <= c + w/2)
        y[mask] += amp
    y = np.maximum(y, 0.0)
    return y


def _boxcar_obj(params):
    y = _boxcar_param_to_signal(params)
    return _fft_peak(y)


def _optimize_boxcars(n_boxes, deadline):
    """Optimize boxcar parameters using differential evolution."""
    from scipy.optimize import differential_evolution
    
    bounds = []
    x0 = []
    for i in range(n_boxes):
        # center in [-0.125, 0.125]
        bounds.append((-0.125, 0.125))
        x0.append(0.0)
        # width in [0.001, 0.25]
        bounds.append((0.001, 0.25))
        x0.append(0.05)
        # amplitude in [0.0, 1.0]
        bounds.append((0.0, 1.0))
        x0.append(1.0)
    
    best_y = np.ones(N)
    best_score = _fft_peak(best_y)
    
    try:
        result = differential_evolution(
            _boxcar_obj, bounds, 
            maxiter=200, popsize=15, tol=1e-6,
            seed=42, polish=True,
            x0=x0
        )
        if result.fun < best_score:
            best_score = result.fun
            best_y = _boxcar_param_to_signal(result.x)
    except Exception:
        pass
    
    return best_y, best_score


def _binary_anneal(deadline):
    """
    Simulated annealing on binary (0/1) step functions.
    The idea: optimal may be sparse/binary.
    """
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Start with a random sparse binary signal
    y = np.zeros(N)
    # Place some random blocks
    n_blocks = 5
    for _ in range(n_blocks):
        start = random.randint(0, N-1)
        length = random.randint(1, 20)
        end = min(start + length, N)
        y[start:end] = 1.0
    
    best_y = y.copy()
    best_score = _fft_peak(best_y)
    current_score = best_score
    
    T = 0.1
    T_min = 0.001
    cooling = 0.95
    
    t = time.time()
    while T > T_min and time.time() < deadline - 20:
        # Generate a neighbor: flip a random block
        y_neighbor = best_y.copy()
        op = random.choice(['add', 'remove', 'move', 'resize'])
        
        if op == 'add':
            start = random.randint(0, N-1)
            length = random.randint(1, 15)
            end = min(start + length, N)
            y_neighbor[start:end] = 1.0
        elif op == 'remove':
            start = random.randint(0, N-1)
            length = random.randint(1, 15)
            end = min(start + length, N)
            y_neighbor[start:end] = 0.0
        elif op == 'move':
            start = random.randint(0, N-1)
            length = random.randint(1, 15)
            end = min(start + length, N)
            block = y_neighbor[start:end].copy()
            if np.any(block > 0):
                new_start = random.randint(0, max(0, N-length))
                new_end = min(new_start + length, N)
                y_neighbor[start:end] = 0.0
                y_neighbor[new_start:new_end] = block
        elif op == 'resize':
            start = random.randint(0, N-1)
            length = random.randint(1, 20)
            end = min(start + length, N)
            if np.any(y_neighbor[start:end] > 0):
                delta = random.randint(-5, 5)
                new_end = min(end + delta, N)
                new_end = max(new_end, start+1)
                y_neighbor[start:new_end] = 1.0
                if new_end < end:
                    y_neighbor[new_end:end] = 0.0
        
        neighbor_score = _fft_peak(y_neighbor)
        delta = neighbor_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            best_y = y_neighbor
            current_score = neighbor_score
            if neighbor_score < best_score:
                best_score = neighbor_score
        
        T *= cooling
    
    return best_y, best_score


def _smooth_polish(y, deadline):
    """
    Polish a signal using SLSQP with smooth constraints.
    """
    from scipy.optimize import minimize
    
    def obj_and_grad(params):
        y = _boxcar_param_to_signal(params)
        score = _fft_peak(y)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += eps
            p_minus = params.copy()
            p_minus[i] -= eps
            grad[i] = (_boxcar_param_to_signal_score(p_plus) - _boxcar_param_to_signal_score(p_minus)) / (2*eps)
        return score, grad
    
    def _boxcar_param_to_signal_score(params):
        y = _boxcar_param_to_signal(params)
        return _fft_peak(y)
    
    # Use the binary anneal result as starting point
    # Convert to boxcar representation approximately
    n_boxes = 8
    params0 = np.zeros(n_boxes * 3)
    # Extract approximate boxcars from the signal
    # This is a rough initialization
    for i in range(n_boxes):
        params0[3*i] = (i - n_boxes/2) * 0.03
        params0[3*i+1] = 0.02
        params0[3*i+2] = 1.0
    
    best_y = y.copy()
    best_score = _fft_peak(best_y)
    
    try:
        # Use L-BFGS-B with finite differences
        res = minimize(
            lambda p: _boxcar_param_to_signal_score(p),
            params0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_y = _boxcar_param_to_signal(res.x)
    except Exception:
        pass
    
    return best_y, best_score


def construct():
    start = time.time()
    deadline = start + 135.0   # safety margin under 150s kill
    
    # Initialize with a reasonable default
    best = np.ones(N)
    best_score = _fft_peak(best)
    
    # Strategy 1: Binary annealing (sparse structures)
    try:
        y_bin, score_bin = _binary_anneal(deadline)
        if score_bin < best_score:
            best, best_score = y_bin, score_bin
    except Exception:
        pass
    
    # Strategy 2: Boxcar optimization (structured rectangular pulses)
    try:
        y_box, score_box = _optimize_boxcars(5, deadline)
        if score_box < best_score:
            best, best_score = y_box, score_box
    except Exception:
        pass
    
    # Strategy 3: Smooth polish on best so far
    try:
        y_smooth, score_smooth = _smooth_polish(best, deadline)
        if score_smooth < best_score:
            best, best_score = y_smooth, score_smooth
    except Exception:
        pass
    
    best = np.maximum(best, 0.0)
    return list(best)

# EVOLVE-BLOCK-END
