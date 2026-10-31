"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second pass for elements not at boundary
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    mask2 = (h > 1e-12) & (h < 1.0 - 1e-12)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max(x, beta=100.0):
    """Soft max using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0
    shifted = x - max_x
    return max_x + math.log(np.sum(np.exp(beta * shifted))) / beta


def _overlap_smooth(h, beta=100.0):
    """Smooth surrogate for max overlap."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return _smooth_max(overlap, beta)


def _sinusoidal_overlap(params, n=N):
    """
    Evaluate overlap for a sinusoidal h.
    params: [amp, freq, phase, offset]
    h[i] = clip(offset + amp * sin(freq * i / n * 2*pi + phase), 0, 1)
    """
    amp, freq, phase, offset = params
    i = np.arange(n)
    h = offset + amp * np.sin(freq * i / n * 2.0 * math.pi + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return _score(h), h


def _optimize_sigmoidal(deadline):
    """
    Optimize a sigmoidal transition profile.
    h is modeled as a transition from a low value to a high value (or vice versa)
    with a smooth sigmoidal step.
    Parameters: [start_idx, transition_width, low_val, high_val]
    """
    n = N
    best_score = float('inf')
    best_h = None
    
    def build_h(params):
        start, width, low, high = params
        i = np.arange(n)
        # Sigmoid centered at start, with width controlling steepness
        # x = (i - start) / width
        # If width is small, it's a sharp step.
        # We use a tanh or sigmoid. Let's use a standard logistic/sigmoid shape.
        # To avoid division by zero or extreme values, clip width.
        w = max(abs(width), 1.0)
        x = (i - start) / w
        # Clip x to avoid overflow in exp if used, but tanh is safer
        # Let's use 0.5 * (1 + tanh(x)) which goes 0 -> 1
        # We want to interpolate between low and high.
        # If start is in the middle, we might want a symmetric shape or just a step.
        # The problem is "shifted 1-h". A step function is often optimal.
        # Let's allow the transition to be anywhere.
        t = 0.5 * (1.0 + np.tanh(x))
        h = low + (high - low) * t
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return h

    def obj_sigmoidal(params):
        h = build_h(params)
        return _overlap_smooth(h, beta=200.0)
    
    # Bounds:
    # start: 0 to N
    # width: 1 to N (steepness)
    # low: 0 to 1
    # high: 0 to 1
    bounds = [(0, n), (1, n), (0, 1), (0, 1)]
    
    try:
        # Use Differential Evolution for global search
        # Limit iterations based on time
        maxiter = 50
        if time.time() < deadline - 20:
            maxiter = 100
            
        result = differential_evolution(
            obj_sigmoidal, bounds,
            seed=42, maxiter=maxiter, tol=1e-8,
            popsize=15, mutation=(0.3, 1.0), recombination=0.7,
            polish=True, init='sobol'
        )
        
        if result.fun < 1.0:
            h = build_h(result.x)
            actual = _score(h)
            if actual < best_score:
                best_score = actual
                best_h = h.copy()
                
    except Exception:
        pass
        
    # Also try a "two-block" variant which is a limit of sigmoidal with width->0
    # But DE might find that. Let's also try optimizing a 2-block structure directly
    # as a fallback or alternative.
    
    return best_h, best_score


def _optimize_full_vector(deadline, initial_guess=None):
    """
    Optimize the full vector h directly using L-BFGS-B with smooth objective.
    This is a structural change: instead of block or sinusoidal parametric families,
    we optimize all N degrees of freedom directly.
    """
    n = N
    best_score = float('inf')
    best_h = None
    
    inits = []
    
    if initial_guess is not None:
        inits.append(initial_guess.copy())
    
    # Add other standard inits if time permits or if initial guess is not provided
    if initial_guess is None or time.time() < deadline - 30:
        # 1. Golden ratio 2-block
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h_init1 = np.zeros(n)
        h_init1[:n//2] = A
        h_init1[n//2:] = B
        h_init1 = _fix_sum(h_init1)
        inits.append(h_init1)
        
        # 2. Random
        np.random.seed(42)
        h_init2 = np.random.uniform(0.2, 0.8, n)
        h_init2 = _fix_sum(h_init2)
        inits.append(h_init2)
    
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        def obj_full(x, beta=300.0):
            h = np.clip(x, 0.0, 1.0)
            s = h.sum()
            penalty = (s - n/2.0) ** 2 * 50.0
            return _overlap_smooth(h, beta) + penalty
        
        def grad_full(x, beta=300.0):
            h = np.clip(x, 0.0, 1.0)
            s = h.sum()
            # Gradient of penalty
            grad_penalty = 2.0 * (s - n/2.0) * 50.0 / n
            # Gradient of smooth max overlap
            shifted = np.zeros(n)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            max_x = np.max(overlap)
            if max_x < 1e-12:
                grad_smooth = np.zeros(n)
            else:
                shifted_x = overlap - max_x
                weights = np.exp(beta * shifted_x)
                weights /= np.sum(weights)
                # d(overlap)/dh: overlap[i] = h[i] * (1 - shifted[i])
                # shifted[i] = h[i+1] for i < n-1, shifted[n-1] = 0
                # d(overlap[i])/dh[i] = (1 - shifted[i])
                # d(overlap[i])/dh[i+1] = -h[i] for i < n-1
                d_oh_dh = np.zeros(n)
                d_oh_dh[:-1] = 1.0 - h[1:]  # d(overlap[i])/dh[i]
                d_oh_dh[1:] -= h[:-1]        # d(overlap[i])/dh[i+1]
                grad_smooth = weights * d_oh_dh
            return grad_penalty + grad_smooth
        
        try:
            res = minimize(obj_full, h_init, jac=grad_full,
                          method='L-BFGS-B',
                          bounds=[(0, 1)] * n,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1.0:
                h = _fix_sum(res.x)
                actual = _score(h)
                if actual < best_score:
                    best_score = actual
                    best_h = h.copy()
        except Exception:
            pass
    
    return best_h, best_score


def _local_search(h, deadline, steps=2000):
    """
    Binary local search: try flipping individual elements or small neighborhoods
    to reduce the max overlap.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    improved = True
    step = 0
    
    while improved and step < steps and time.time() < deadline:
        improved = False
        step += 1
        
        # Try perturbing each element
        for i in range(n):
            if time.time() > deadline:
                break
            
            orig_val = best_h[i]
            
            # Try a few perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                new_val = orig_val + delta
                if new_val < 0 or new_val > 1:
                    continue
                
                candidate = best_h.copy()
                candidate[i] = new_val
                candidate = _fix_sum(candidate)
                
                score = _score(candidate)
                if score < best_score - 1e-12:
                    best_h = candidate
                    best_score = score
                    improved = True
                    break  # Move to next i
    
    return best_h, best_score


def _optimize_2block(deadline):
    """
    Optimize 2-block construction: h = [a]*k + [b]*(N-k)
    where k*a + (N-k)*b = N/2.
    This is a simpler case of the sigmoidal model (width -> 0).
    """
    n = N
    best_score = float('inf')
    best_h = None
    
    def build_2block(params):
        k = int(round(params[0]))
        a = params[1]
        # b is determined by sum constraint: k*a + (n-k)*b = n/2
        # b = (n/2 - k*a) / (n-k)
        if k <= 0 or k >= n:
            return None
        b = (n/2.0 - k*a) / (n - k)
        if b < 0 or b > 1:
            return None
        h = np.zeros(n)
        h[:k] = a
        h[k:] = b
        h = _fix_sum(h)
        return h

    def obj_2block(params):
        h = build_2block(params)
        if h is None:
            return 1.0
        return _overlap_smooth(h, beta=200.0)
    
    bounds = [(1, n-1), (0, 1)]
    
    try:
        result = differential_evolution(
            obj_2block, bounds,
            seed=101, maxiter=100, tol=1e-8,
            popsize=20, mutation=(0.3, 1.0), recombination=0.8,
            polish=True, init='sobol'
        )
        
        if result.fun < 1.0:
            h = build_2block(result.x)
            if h is not None:
                actual = _score(h)
                if actual < best_score:
                    best_score = actual
                    best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: 2-block optimization (fast, robust baseline)
    if time.time() < deadline - 80:
        h2, s2 = _optimize_2block(deadline - 80)
        if h2 is not None and s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Phase 2: Sigmoidal optimization (structural change: smooth transition)
    if time.time() < deadline - 40:
        hs, ss = _optimize_sigmoidal(deadline - 40)
        if hs is not None and ss < best_score:
            best_score = ss
            best_h = hs
    
    # Phase 3: Full vector optimization (polish from best found)
    if time.time() < deadline - 20 and best_h is not None:
        hf, sf = _optimize_full_vector(deadline - 20, initial_guess=best_h)
        if hf is not None and sf < best_score:
            best_score = sf
            best_h = hf
    
    # Phase 4: Local search polish on best found
    if time.time() < deadline - 5 and best_h is not None:
        hl, sl = _local_search(best_h, deadline - 5, steps=3000)
        if sl < best_score:
            best_score = sl
            best_h = hl
    
    # Fallback: golden ratio 2-block
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
