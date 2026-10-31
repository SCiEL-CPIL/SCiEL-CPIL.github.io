"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = TARGET_SUM - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    diff = TARGET_SUM - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _binomial_cdf(x, k, n):
    """
    Compute CDF of Binomial(n, p) at x, where p is derived from x's position.
    Here we use a fixed p=0.5 for simplicity, and scale x to [0, n].
    Returns P(X <= x) for X ~ Binomial(n, 0.5).
    """
    from scipy.stats import binom
    x_int = np.floor(x).astype(int)
    x_int = np.clip(x_int, 0, n)
    return binom.cdf(x_int, n, 0.5)

def _generate_binomial_transition(h_low, h_high, pos, width, order):
    """
    Generate a sequence with a single transition from h_low to h_high.
    The transition profile uses a cumulative binomial distribution.
    
    h_low: value before transition
    h_high: value after transition
    pos: position where transition is centered
    width: width of transition region
    order: number of binomial convolutions (smoothing order)
    """
    h = np.full(N, h_low)
    
    # Determine transition region
    left = int(pos - width / 2)
    right = int(pos + width / 2)
    
    # Clip to valid range
    left = max(0, left)
    right = min(N, right)
    
    if right > left:
        # Generate transition profile using binomial CDF
        x = np.arange(left, right)
        # Normalize position to [0, 1]
        x_norm = (x - left) / max(1, right - left - 1)
        
        # Use binomial CDF with order 'order'
        # Scale to get smooth transition
        profile = _binomial_cdf(x_norm * order, order, order)
        
        # Interpolate between h_low and h_high
        h[left:right] = h_low + (h_high - h_low) * profile
    
    return h

def _binomial_objective(params):
    """
    Objective for binomial transition optimization.
    params: [h_low, h_high, pos, width, order]
    """
    h_low, h_high, pos, width, order = params
    
    # Ensure valid parameters
    if h_low < 0 or h_low > 1 or h_high < 0 or h_high > 1:
        return 10.0
    if width < 1 or width > N:
        return 10.0
    if order < 1 or order > 20:
        return 10.0
    
    order_int = int(round(order))
    
    # Generate sequence
    h = _generate_binomial_transition(h_low, h_high, pos, width, order_int)
    
    # Enforce sum constraint
    h_final = _enforce_sum(h)
    
    # Check validity
    if not (np.all(h_final >= -1e-10) and np.all(h_final <= 1.0 + 1e-10)):
        return 10.0
    
    s = np.sum(h_final)
    if abs(s - TARGET_SUM) > 1e-8:
        return 10.0
    
    return _score(h_final)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - TARGET_SUM) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Global search using Differential Evolution on binomial transition parameters
    try:
        if time.time() < deadline - 60.0:
            # Parameters: [h_low, h_high, pos, width, order]
            bounds = [
                (0.0, 1.0),  # h_low
                (0.0, 1.0),  # h_high
                (5.0, N - 5),  # pos
                (2.0, 30.0),  # width
                (1.0, 20.0)   # order
            ]
            
            # Run differential evolution
            try:
                de_result = differential_evolution(
                    _binomial_objective,
                    bounds,
                    maxiter=300,
                    popsize=20,
                    tol=1e-12,
                    seed=42,
                    polish=True,
                    init='sobol'
                )
                
                if de_result.fun < current_score:
                    # Regenerate the best h
                    h_low, h_high, pos, width, order = de_result.x
                    order_int = int(round(order))
                    h_de = _generate_binomial_transition(h_low, h_high, pos, width, order_int)
                    h_de = _enforce_sum(h_de)
                    
                    if np.all(h_de >= -1e-10) and np.all(h_de <= 1.0 + 1e-10):
                        s = np.sum(h_de)
                        if abs(s - TARGET_SUM) < 1e-8:
                            sc = _score(h_de)
                            if sc < current_score:
                                _try_update(h_de)
            
            except Exception:
                pass
            
            # Try multiple initial positions for diversity
            for pos_init in [25, 40, 50, 60, 75]:
                if time.time() > deadline - 30.0:
                    break
                
                # Try a few different binomial orders
                for order in [3, 5, 8, 12]:
                    if time.time() > deadline - 25.0:
                        break
                    
                    # Grid search over width
                    for width in [4, 6, 8, 10, 14, 20]:
                        if time.time() > deadline - 20.0:
                            break
                        
                        # Try different h_low/h_high combinations
                        for h_low in [A * 0.8, A * 0.9, A, A * 1.1, A * 1.2]:
                            for h_high in [B * 0.8, B * 0.9, B, B * 1.1, B * 1.2]:
                                if h_low < 0 or h_low > 1 or h_high < 0 or h_high > 1:
                                    continue
                                
                                h_candidate = _generate_binomial_transition(h_low, h_high, pos_init, width, order)
                                h_candidate = _enforce_sum(h_candidate)
                                
                                if np.all(h_candidate >= -1e-10) and np.all(h_candidate <= 1.0 + 1e-10):
                                    s = np.sum(h_candidate)
                                    if abs(s - TARGET_SUM) < 1e-8:
                                        sc = _score(h_candidate)
                                        if sc < current_score:
                                            _try_update(h_candidate)
                
                if time.time() > deadline - 15.0:
                    break
            
    except Exception:
        pass

    # Phase 2: Local polish on best solution using SLSQP with smooth max
    try:
        if time.time() < deadline - 10.0:
            def objective_smooth(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # Use smooth max
                alpha = 500.0
                x_max = np.max(overlap)
                smax = x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))
                return smax

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - TARGET_SUM}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_smooth,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )

            if result.fun < current_score + 0.01:
                _try_update(result.x)

    except Exception:
        pass

    # Phase 3: Final tight polish
    try:
        if time.time() < deadline - 3.0:
            def objective_tight(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - TARGET_SUM}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_tight,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < current_score:
                _try_update(result.x)

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
