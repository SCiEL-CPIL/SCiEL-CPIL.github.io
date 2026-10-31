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
        # First pass: spread across all
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second pass: only non-boundary
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


def _overlap_for_2block(k, a, b):
    """Compute max overlap for 2-block with first k elements = a, rest = b."""
    n = N
    if k <= 0 or k >= n:
        return float('inf')
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    return max(vals)


def _optimize_2block(deadline):
    """Optimize 2-block construction using exact objective."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(1, n):
        # Constraint: k*a + (N-k)*b = N/2
        # b = (N/2 - k*a) / (N-k)
        
        def obj(a_val, k=k):
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                return 100.0
            b_val = np.clip(b_val, 0.0, 1.0)
            return _overlap_for_2block(k, a_val, b_val)
        
        # Bounds on a
        a_min = max(0.0, 1.5 - 1.0/k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # Sample starting points
        for a0 in [a_min + (a_max - a_min) * t for t in [0.1, 0.3, 0.5, 0.7, 0.9]]:
            a0 = np.clip(a0, a_min + 1e-9, a_max - 1e-9)
            try:
                res = minimize(obj, [a0], method='L-BFGS-B', bounds=[(a_min + 1e-9, a_max - 1e-9)])
                if res.success and res.fun < best_score:
                    a_opt = res.x[0]
                    b_opt = (N / 2.0 - k * a_opt) / (N - k)
                    b_opt = np.clip(b_opt, 0.0, 1.0)
                    score_val = _overlap_for_2block(k, a_opt, b_opt)
                    if score_val < best_score:
                        best_score = score_val
                        h = np.zeros(N)
                        h[:k] = a_opt
                        h[k:] = b_opt
                        h = _fix_sum(h)
                        best_h = h.copy()
            except Exception:
                pass
        
        if time.time() > deadline - 60:
            break
    
    return best_h, best_score


def _optimize_multiblock_smooth(deadline):
    """Optimize multi-block construction using smooth objective with SLSQP."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Try 3-block and 4-block
    for n_blocks in [3, 4]:
        if time.time() > deadline - 30:
            break
        
        # Parameterize: n_blocks-1 boundary positions + n_blocks heights
        # Use differential evolution for global search, then polish with SLSQP
        
        def de_obj(params, n_blocks=n_blocks):
            # params: [b1, b2, ..., b_{n-1}, h1, h2, ..., h_{n}]
            boundaries = np.zeros(n_blocks + 1)
            boundaries[0] = 0
            boundaries[-1] = n
            for i in range(n_blocks - 1):
                boundaries[i+1] = int(round(params[i]))
            # Ensure sorted
            for i in range(1, n_blocks):
                if boundaries[i+1] <= boundaries[i]:
                    boundaries[i+1] = boundaries[i] + 1
            heights = params[n_blocks-1:]
            
            # Check sum
            total = 0
            for i in range(n_blocks):
                length = boundaries[i+1] - boundaries[i]
                total += length * heights[i]
            penalty = max(0, abs(total - n/2.0) - 0.05) * 5
            
            h = np.zeros(n)
            for i in range(n_blocks):
                start = int(boundaries[i])
                end = int(boundaries[i+1])
                if end > start:
                    h[start:end] = heights[i]
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=100.0) + penalty
        
        de_bounds = [(0, n)] * (n_blocks - 1) + [(0, 1)] * n_blocks
        
        try:
            result = differential_evolution(
                de_obj, de_bounds,
                seed=42, maxiter=80, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sobol'
            )
            
            if result.fun < best_score + 0.1:
                # Extract and polish
                boundaries = np.zeros(n_blocks + 1)
                boundaries[0] = 0
                boundaries[-1] = n
                for i in range(n_blocks - 1):
                    boundaries[i+1] = int(round(result.x[i]))
                for i in range(1, n_blocks):
                    if boundaries[i+1] <= boundaries[i]:
                        boundaries[i+1] = boundaries[i] + 1
                heights = result.x[n_blocks-1:]
                
                # Polish with SLSQP
                def polish_obj(x, n_blocks=n_blocks, boundaries=boundaries):
                    h = np.zeros(n)
                    for i in range(n_blocks):
                        start = int(boundaries[i])
                        end = int(boundaries[i+1])
                        if end > start:
                            h[start:end] = x[i]
                    h = _fix_sum(h)
                    return _overlap_smooth(h, beta=500.0)
                
                res2 = minimize(polish_obj, heights,
                               method='SLSQP',
                               bounds=[(0, 1)] * n_blocks,
                               options={'maxiter': 500, 'ftol': 1e-12})
                
                if res2.fun < 1.0:
                    h = np.zeros(n)
                    for i in range(n_blocks):
                        start = int(boundaries[i])
                        end = int(boundaries[i+1])
                        if end > start:
                            h[start:end] = res2.x[i]
                    h = _fix_sum(h)
                    actual = _score(h)
                    if actual < best_score:
                        best_score = actual
                        best_h = h.copy()
        except Exception:
            pass
    
    return best_h, best_score


def _optimize_free_params(deadline):
    """Optimize all N parameters directly using smooth objective."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Start from a reasonable initial guess
    # Use the 2-block solution as starting point
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(n)
    h0[:n//2] = A
    h0[n//2:] = B
    h0 = _fix_sum(h0)
    
    def obj(h_flat, beta=200.0):
        h = h_flat.copy()
        h = _fix_sum(h)
        return _overlap_smooth(h, beta)
    
    try:
        res = minimize(obj, h0, method='L-BFGS-B', bounds=[(0, 1)] * n,
                      options={'maxiter': 500, 'ftol': 1e-12})
        if res.fun < 1.0:
            h = _fix_sum(res.x)
            actual = _score(h)
            if actual < best_score:
                best_score = actual
                best_h = h.copy()
    except Exception:
        pass
    
    # Try a few different starting points
    for frac in [0.3, 0.5, 0.7]:
        if time.time() > deadline - 20:
            break
        
        h_init = np.zeros(n)
        # Create a smoother transition
        x = np.linspace(0, 1, n)
        h_init = A + (B - A) * ((x - frac) ** 2)
        h_init = np.clip(h_init, 0, 1)
        h_init = _fix_sum(h_init)
        
        try:
            res = minimize(obj, h_init, method='L-BFGS-B', bounds=[(0, 1)] * n,
                          options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < 1.0:
                h = _fix_sum(res.x)
                actual = _score(h)
                if actual < best_score:
                    best_score = actual
                    best_h = h.copy()
        except Exception:
            pass
    
    return best_h, best_score


def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Optimize 2-block (fast, exact)
    h2, s2 = _optimize_2block(deadline)
    if h2 is not None and s2 < best_score:
        best_score = s2
        best_h = h2
    
    if time.time() > deadline - 40:
        if best_h is None:
            A = (3.0 - math.sqrt(5.0)) / 2.0
            B = 1.0 - A
            h = np.zeros(N)
            h[:N//2] = A
            h[N//2:] = B
            best_h = _fix_sum(h)
        return best_h
    
    # Phase 2: Optimize multi-block (3-4 blocks) with smooth objective
    h3, s3 = _optimize_multiblock_smooth(deadline)
    if h3 is not None and s3 < best_score:
        best_score = s3
        best_h = h3
    
    if time.time() > deadline - 20:
        if best_h is None:
            A = (3.0 - math.sqrt(5.0)) / 2.0
            B = 1.0 - A
            h = np.zeros(N)
            h[:N//2] = A
            h[N//2:] = B
            best_h = _fix_sum(h)
        return best_h
    
    # Phase 3: Optimize all N parameters directly
    hf, sf = _optimize_free_params(deadline)
    if hf is not None and sf < best_score:
        best_score = sf
        best_h = hf
    
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
