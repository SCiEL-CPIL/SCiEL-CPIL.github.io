"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
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

def _smooth_max(x, alpha=500.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(100):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment for non-clipped elements
    mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    if np.any(mask):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) > 1e-15:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _sinusoidal_h(params):
    """Generate h from sinusoidal parameters: [freq, phase, amp, offset]."""
    freq, phase, amp, offset = params
    x = np.linspace(0, 2, N)
    h = offset + amp * np.sin(freq * x + phase)
    h = np.clip(h, 0.0, 1.0)
    return h

def _sinusoidal_objective(params):
    """Objective for sinusoidal parameterization."""
    h = _sinusoidal_h(params)
    h_final = _enforce_sum(h)
    if not (np.all(h_final >= -1e-10) and np.all(h_final <= 1.0 + 1e-10)):
        return 10.0
    s = np.sum(h_final)
    if abs(s - N/2.0) > 1e-8:
        return 10.0
    return _score(h_final)

def _kkt_polish(h, max_iter=500, tol=1e-14):
    """KKT-based polish for the active set of constraints."""
    h = h.copy()
    n = len(h)
    
    for _ in range(max_iter):
        shifted = np.zeros(n)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        max_val = np.max(overlap)
        
        # Active constraints: where overlap is close to max
        active = overlap > max_val - 1e-10
        # Bound constraints: where h is at 0 or 1
        at_lower = h < 1e-12
        at_upper = h > 1.0 - 1e-12
        
        # Free variables: not at bounds and not in active overlap
        free = ~(at_lower | at_upper | active)
        
        if not np.any(free):
            break
        
        # Compute gradient of max overlap (approximate)
        grad = np.zeros(n)
        # d(overlap_i)/d(h_j) for j in {i, i-1}
        for i in range(n):
            if overlap[i] >= max_val - 1e-10:
                # overlap_i = h[i] * (1 - h[i+1]) if i < n-1, else h[i] * 1
                if i < n - 1:
                    grad[i] += (1.0 - h[i+1])
                    grad[i+1] += (-h[i])
                else:
                    grad[i] += 1.0
        
        # Project gradient to free variables
        grad_free = grad[free]
        
        if np.all(np.abs(grad_free) < tol):
            break
        
        # Line search with step size
        step = 1e-4
        improved = False
        for _ in range(50):
            h_trial = h.copy()
            h_trial[free] -= step * grad_free
            h_trial = _enforce_sum(h_trial)
            
            if np.all(h_trial >= -1e-12) and np.all(h_trial <= 1.0 + 1e-12):
                s = np.sum(h_trial)
                if abs(s - N/2.0) < 1e-12:
                    sc = _score(h_trial)
                    if sc < max_val - 1e-15:
                        h = h_trial
                        improved = True
                        break
            step *= 0.5
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Try sinusoidal parameterization
    try:
        if time.time() < deadline - 80.0:
            # Try several sinusoidal parameters
            best_sin_score = current_score
            best_sin_h = h_final.copy()
            
            for freq in [2.0, 3.0, 4.0, 5.0]:
                for phase in [0.0, math.pi/4, math.pi/2, 3*math.pi/4]:
                    for amp in [0.3, 0.4, 0.5]:
                        params = [freq, phase, amp, 0.5]
                        h_sin = _sinusoidal_h(params)
                        h_sin = _enforce_sum(h_sin)
                        if np.all(h_sin >= -1e-10) and np.all(h_sin <= 1.0 + 1e-10):
                            s = np.sum(h_sin)
                            if abs(s - N/2.0) < 1e-8:
                                sc = _score(h_sin)
                                if sc < best_sin_score:
                                    best_sin_score = sc
                                    best_sin_h = h_sin.copy()
            
            if best_sin_score < current_score:
                _try_update(best_sin_h)
                
    except Exception:
        pass

    # Phase 2: Local polish using SLSQP with smooth max
    try:
        if time.time() < deadline - 20.0:
            def objective_smooth(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return _smooth_max(overlap, alpha=500.0)

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_smooth,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False}
            )

            if result.fun < current_score + 0.01:
                _try_update(result.x)

    except Exception:
        pass

    # Phase 3: KKT polish
    try:
        if time.time() < deadline - 5.0:
            h_kkt = _kkt_polish(h_final, max_iter=1000, tol=1e-15)
            _try_update(h_kkt)
    except Exception:
        pass

    # Phase 4: Final tight polish
    try:
        if time.time() < deadline - 2.0:
            def objective_tight(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_tight,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 10000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < current_score:
                _try_update(result.x)

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
