"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

_start_time = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _score_h(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Uses FFT-based cross-correlation for efficiency.
    """
    sw = 2.0 / N
    # h is defined on [0, 2], 0 outside
    # g(x) = 1 - h(x) for x in [0, 2], 1 outside
    # C(s) = integral_0^2 h(x) * g(x+s) dx
    #      = integral_0^2 h(x) * (1 - h(x+s) * 1_{x+s in [0,2]}) dx
    #      = integral_0^2 h(x) dx - integral_0^2 h(x) h(x+s) 1_{x+s in [0,2]} dx
    #      = 1 - integral_0^2 h(x) h(x+s) 1_{x+s in [0,2]} dx
    
    # Let R(s) = integral_0^2 h(x) h(x+s) 1_{x+s in [0,2]} dx
    # This is the autocorrelation of h, but truncated.
    # Specifically, R(s) = sum_i h_i * (integral over x in step i, x+s in [0,2] of h(x+s) dx)
    
    # For discrete shifts k*sw, we can compute this exactly.
    # Let's use FFT to compute autocorrelation of h.
    # h is a sequence of step heights.
    # R(k) for shift s = k*sw is:
    # sum_{i=0}^{N-1} h_i * h_{i+k} * (length of intersection of step i and step i+k shifted)
    # Wait, h is constant on each step.
    # h(x) = h_i for x in [i*sw, (i+1)*sw)
    # h(x+s) = h_j for x+s in [j*sw, (j+1)*sw) => x in [j*sw - s, (j+1)*sw - s)
    # Intersection of [i*sw, (i+1)*sw) and [j*sw - s, (j+1)*sw - s)
    
    # Actually, for shift s = k*sw (integer shift in steps):
    # h(x+k*sw) = h_{i+k} if i+k < N, else 0.
    # x in [i*sw, (i+1)*sw) => x+k*sw in [(i+k)*sw, (i+k+1)*sw)
    # So h(x) = h_i, h(x+k*sw) = h_{i+k} (if i+k in [0, N), else 0)
    # The integral over step i is h_i * h_{i+k} * sw, for i such that i+k < N and i+k >= 0.
    # So R(k) = sw * sum_{i=0}^{N-1-k} h_i * h_{i+k} for k >= 0.
    # For k < 0, R(k) = R(-k) by symmetry? No, h is not necessarily symmetric.
    # But we only need max over s in [0, 2). s = k*sw for k=0,...,N-1.
    
    # So we need R(k) for k=0,...,N-1.
    # R(k) = sw * sum_{i=0}^{N-1-k} h_i * h_{i+k}
    
    # This is the autocorrelation of h.
    # We can compute it using np.correlate.
    autocorr = np.correlate(h, h, mode='full')
    # autocorr[N-1] is lag 0.
    # autocorr[N-1+k] is lag k.
    # R(k) = sw * autocorr[N-1+k] for k=0,...,N-1
    
    R = sw * autocorr[N-1 : 2*N-1]  # lags 0 to N-1
    # C(k) = 1 - R(k)
    C = 1.0 - R
    return np.max(C)

def _objective_lse(h, beta=100.0):
    """
    Smooth approximation of max overlap using Log-Sum-Exp.
    """
    sw = 2.0 / N
    autocorr = np.correlate(h, h, mode='full')
    R = sw * autocorr[N-1 : 2*N-1]
    C = 1.0 - R
    return logsumexp(beta * C) / beta

def _project(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    Simple projection: clip to [0,1], then adjust to meet sum constraint.
    """
    h = np.clip(h, 0, 1)
    diff = N / 2.0 - np.sum(h)
    # Distribute difference evenly, but must stay in [0,1]
    # If diff > 0, add diff/N to each, clip at 1, redistribute excess.
    # If diff < 0, subtract -diff/N from each, clip at 0, redistribute deficit.
    # This is an iterative water-filling.
    for _ in range(10):
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            # Need to add
            room = 1.0 - h
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            h += diff / total_room * room
        else:
            # Need to subtract
            room = h
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            h += diff / total_room * room
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
    return h

def construct():
    global _start_time
    deadline = _start_time + _TIME_LIMIT
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _score_h(best_h)
    
    # Try binary initializations
    binary_inits = []
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    binary_inits.append(h_alt)
    # Block
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    binary_inits.append(h_block)
    
    # Add random binary initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        h_rand = (rng.random(N) < 0.5).astype(float)
        # Adjust to meet sum constraint approximately
        # Just use it as a starting point for optimizer
        binary_inits.append(h_rand)
    
    # Optimize from each initialization
    bounds = [(0, 1)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    
    for i, h_init in enumerate(binary_inits):
        if time.time() > deadline - 20:
            break
        try:
            h_init = _project(h_init)
            # Use SLSQP with LSE objective
            result = minimize(
                _objective_lse, h_init, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            if result.success:
                h_opt = _project(result.x)
                val = _score_h(h_opt)
                if val < best_val:
                    best_h = h_opt.copy()
                    best_val = val
        except Exception:
            pass
    
    # Polish the best solution
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective_lse, best_h, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.success:
                h_opt = _project(result.x)
                val = _score_h(h_opt)
                if val < best_val:
                    best_h = h_opt
                    best_val = val
        except Exception:
            pass
    
    # Final projection and check
    best_h = _project(best_h)
    return best_h
# EVOLVE-BLOCK-END
