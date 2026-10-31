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


def _construct_from_params(params):
    """Given block boundaries and heights, construct h and fix sum."""
    n = N
    h = np.zeros(n)
    num_blocks = len(params) // 2
    for i in range(num_blocks):
        start = int(params[2 * i])
        end = int(params[2 * i + 1])
        height = params[2 * i + 1]  # placeholder, will fix
        # Actually params structure: [b0_start, b0_end, b0_height, b1_start, b1_end, b1_height, ...]
        # Let me restructure: params = [start_0, end_0, height_0, start_1, end_1, height_1, ...]
        pass
    
    # Simpler: params = [boundaries, heights] where boundaries are block edges
    # Let's use a different approach: params = [k1, h1, k2, h2, ..., k_m, h_m]
    # where k_i are block lengths and h_i are heights
    total_len = 0
    for i in range(0, len(params), 2):
        k = int(round(params[i]))
        h_val = params[i + 1]
        k = max(0, min(k, n - total_len))
        h[total_len:total_len + k] = h_val
        total_len += k
    # Fill remaining with 0
    if total_len < n:
        h[total_len:] = 0.0
    
    # Fix sum to be exactly n/2
    target = n / 2.0
    current_sum = h.sum()
    diff = target - current_sum
    
    if abs(diff) > 1e-12:
        # Distribute correction across all elements, but keep them in [0,1]
        # Add diff/n to each element
        h += diff / n
        # Clip and re-fix
        h = np.clip(h, 0.0, 1.0)
        # Second pass: adjust elements that weren't clipped
        s = h.sum()
        diff2 = target - s
        # Find elements not at boundary
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff2 / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            # Final fix
            s = h.sum()
            diff3 = target - s
            if abs(diff3) > 1e-15:
                h[0] += diff3
                h = np.clip(h, 0.0, 1.0)
    
    return h


def _multi_block_overlap(h):
    """Compute max overlap for a given h."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _objective_4block(params):
    """Objective for 4-block construction: params = [k1,k2,k3,k4, h1,h2,h3,h4] with sum(ki)=N."""
    n = N
    k1, k2, k3, k4 = params[0], params[1], params[2], params[3]
    h1, h2, h3, h4 = params[4], params[5], params[6], params[7]
    
    # Ensure integer block sizes summing to n
    k1 = int(round(k1))
    k2 = int(round(k2))
    k3 = int(round(k3))
    k4 = n - k1 - k2 - k3
    
    if k1 < 0 or k2 < 0 or k3 < 0 or k4 < 0:
        return 1.0
    
    # Constraint: k1*h1 + k2*h2 + k3*h3 + k4*h4 = n/2
    current_sum = k1*h1 + k2*h2 + k3*h3 + k4*h4
    if abs(current_sum - n/2.0) > 0.5:
        return 1.0 + abs(current_sum - n/2.0)
    
    # Build h
    h = np.zeros(n)
    pos = 0
    for k, hv in [(k1, h1), (k2, h2), (k3, h3), (k4, h4)]:
        if k > 0:
            h[pos:pos+k] = hv
            pos += k
    
    # Fix sum precisely
    diff = n/2.0 - h.sum()
    if abs(diff) > 1e-12:
        h += diff / n
        h = np.clip(h, 0.0, 1.0)
        s = h.sum()
        diff2 = n/2.0 - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    
    return _multi_block_overlap(h)


def _optimize_4block(deadline):
    """Optimize 4-block construction using differential evolution."""
    n = N
    
    # Bounds: k_i in [0, n], h_i in [0, 1]
    bounds = [(0, n), (0, n), (0, n), (0, n), 
              (0, 1), (0, 1), (0, 1), (0, 1)]
    
    # We need k1+k2+k3+k4 = n, so let's parameterize with k1,k2,k3 and set k4 = n-k1-k2-k3
    # And enforce sum constraint via the objective
    
    def obj(params):
        k1, k2, k3, _, h1, h2, h3, h4 = params
        k4 = n - int(round(k1)) - int(round(k2)) - int(round(k3))
        if k4 < 0:
            return 1.0
        current_sum = int(round(k1))*h1 + int(round(k2))*h2 + int(round(k3))*h3 + k4*h4
        if abs(current_sum - n/2.0) > 1.0:
            return 1.0 + abs(current_sum - n/2.0)
        
        h = np.zeros(n)
        pos = 0
        for k, hv in [(int(round(k1)), h1), (int(round(k2)), h2), (int(round(k3)), h3), (k4, h4)]:
            if k > 0:
                h[pos:pos+k] = hv
                pos += k
        
        diff = n/2.0 - h.sum()
        if abs(diff) > 1e-12:
            h += diff / n
            h = np.clip(h, 0.0, 1.0)
            s = h.sum()
            diff2 = n/2.0 - s
            if abs(diff2) > 1e-15:
                mask = (h > 1e-9) & (h < 1.0 - 1e-9)
                if np.any(mask):
                    h[mask] += diff2 / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        
        return _multi_block_overlap(h)
    
    # Initial best from 2-block
    best_score = float('inf')
    best_h = None
    
    # Try 2-block as starting point
    for k in range(1, n):
        a_min = max(0.0, (n/2.0 - n + k) / k)
        a_max = min(1.0, n / (2.0 * k))
        if a_min >= a_max:
            continue
        # Sample
        for frac in [0.2, 0.4, 0.5, 0.6, 0.8]:
            a = a_min + frac * (a_max - a_min)
            b = (n/2.0 - k*a) / (n-k)
            if b < 0 or b > 1:
                continue
            h = np.zeros(n)
            h[:k] = a
            h[k:] = b
            diff = n/2.0 - h.sum()
            h += diff / n
            h = np.clip(h, 0.0, 1.0)
            s = _multi_block_overlap(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    if time.time() > deadline - 30:
        return best_h, best_score
    
    # Now try differential evolution for 4-block
    try:
        # Parameterize: k1, k2, k3 (k4 determined), h1, h2, h3, h4
        # With constraint k1+h1*k1 + ... = n/2
        # Use soft constraint
        
        def de_obj(x):
            k1, k2, k3, h1, h2, h3, h4 = x
            k1_i = int(round(k1))
            k2_i = int(round(k2))
            k3_i = int(round(k3))
            k4_i = n - k1_i - k2_i - k3_i
            if k4_i < 0 or k1_i < 0 or k2_i < 0 or k3_i < 0:
                return 1.0
            sum_h = k1_i*h1 + k2_i*h2 + k3_i*h3 + k4_i*h4
            penalty = max(0, abs(sum_h - n/2.0) - 0.01) * 10
            h = np.zeros(n)
            pos = 0
            for k, hv in [(k1_i, h1), (k2_i, h2), (k3_i, h3), (k4_i, h4)]:
                if k > 0:
                    h[pos:pos+k] = hv
                    pos += k
            diff = n/2.0 - h.sum()
            if abs(diff) > 1e-12:
                h += diff / n
                h = np.clip(h, 0.0, 1.0)
                s2 = h.sum()
                diff2 = n/2.0 - s2
                if abs(diff2) > 1e-15:
                    mask = (h > 1e-9) & (h < 1.0 - 1e-9)
                    if np.any(mask):
                        h[mask] += diff2 / np.sum(mask)
                        h = np.clip(h, 0.0, 1.0)
            return _multi_block_overlap(h) + penalty
        
        de_bounds = [(0, n), (0, n), (0, n), (0, 1), (0, 1), (0, 1), (0, 1)]
        
        result = differential_evolution(
            de_obj, de_bounds, 
            seed=42, maxiter=200, tol=1e-10,
            popsize=20, mutation=(0.3, 1.0), recombination=0.8,
            polish=True, init='sobol'
        )
        
        if result.fun < best_score:
            k1_i = int(round(result.x[0]))
            k2_i = int(round(result.x[1]))
            k3_i = int(round(result.x[2]))
            k4_i = n - k1_i - k2_i - k3_i
            h1, h2, h3, h4 = result.x[3], result.x[4], result.x[5], result.x[6]
            
            h = np.zeros(n)
            pos = 0
            for k, hv in [(k1_i, h1), (k2_i, h2), (k3_i, h3), (k4_i, h4)]:
                if k > 0:
                    h[pos:pos+k] = hv
                    pos += k
            diff = n/2.0 - h.sum()
            if abs(diff) > 1e-12:
                h += diff / n
                h = np.clip(h, 0.0, 1.0)
                s2 = h.sum()
                diff2 = n/2.0 - s2
                if abs(diff2) > 1e-15:
                    mask = (h > 1e-9) & (h < 1.0 - 1e-9)
                    if np.any(mask):
                        h[mask] += diff2 / np.sum(mask)
                        h = np.clip(h, 0.0, 1.0)
            
            actual_score = _multi_block_overlap(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def _optimize_3block(deadline):
    """Optimize 3-block construction."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k1 in range(1, n-1):
        for k2 in range(1, n-k1):
            k3 = n - k1 - k2
            if k3 < 1:
                continue
            
            # Optimize h1, h2, h3 with constraint k1*h1 + k2*h2 + k3*h3 = n/2
            # Minimize max overlap
            
            def obj3(x):
                h1, h2, h3 = x
                if h1 < 0 or h1 > 1 or h2 < 0 or h2 > 1 or h3 < 0 or h3 > 1:
                    return 1.0
                s = k1*h1 + k2*h2 + k3*h3
                if abs(s - n/2.0) > 0.5:
                    return 1.0 + abs(s - n/2.0)
                
                h = np.zeros(n)
                h[:k1] = h1
                h[k1:k1+k2] = h2
                h[k1+k2:] = h3
                
                diff = n/2.0 - h.sum()
                if abs(diff) > 1e-12:
                    h += diff / n
                    h = np.clip(h, 0.0, 1.0)
                    s2 = h.sum()
                    diff2 = n/2.0 - s2
                    if abs(diff2) > 1e-15:
                        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
                        if np.any(mask):
                            h[mask] += diff2 / np.sum(mask)
                            h = np.clip(h, 0.0, 1.0)
                
                return _multi_block_overlap(h)
            
            # Try a few starting points
            for frac in [0.2, 0.5, 0.8]:
                h1_init = frac
                h2_init = frac * 0.8
                h3_init = (n/2.0 - k1*h1_init - k2*h2_init) / k3
                if h3_init < 0 or h3_init > 1:
                    continue
                try:
                    res = minimize(obj3, np.array([h1_init, h2_init, h3_init]),
                                   method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_score:
                        h1, h2, h3 = res.x
                        h = np.zeros(n)
                        h[:k1] = h1
                        h[k1:k1+k2] = h2
                        h[k1+k2:] = h3
                        diff = n/2.0 - h.sum()
                        if abs(diff) > 1e-12:
                            h += diff / n
                            h = np.clip(h, 0.0, 1.0)
                            s2 = h.sum()
                            diff2 = n/2.0 - s2
                            if abs(diff2) > 1e-15:
                                mask = (h > 1e-9) & (h < 1.0 - 1e-9)
                                if np.any(mask):
                                    h[mask] += diff2 / np.sum(mask)
                                    h = np.clip(h, 0.0, 1.0)
                        actual = _multi_block_overlap(h)
                        if actual < best_score:
                            best_score = actual
                            best_h = h.copy()
                except Exception:
                    pass
            
            if time.time() > deadline - 20:
                return best_h, best_score
    
    return best_h, best_score


def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill
    
    best_h = None
    best_score = float('inf')
    
    # Start with 2-block optimization (fast)
    for k in range(1, N):
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # Fine grid
        n_samples = 200
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, n_samples)
        
        best_local_score = float('inf')
        best_local_a = None
        
        for a_val in a_vals:
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            
            h = np.zeros(N)
            h[:k] = a_val
            h[k:] = b_val
            
            diff = N/2.0 - h.sum()
            if abs(diff) > 1e-15:
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
            
            s = _multi_block_overlap(h)
            if s < best_local_score:
                best_local_score = s
                best_local_a = a_val
        
        if best_local_a is not None and best_local_score < best_score:
            best_score = best_local_score
            a_opt = best_local_a
            b_opt = (N / 2.0 - k * a_opt) / (N - k)
            b_opt = np.clip(b_opt, 0.0, 1.0)
            h = np.zeros(N)
            h[:k] = a_opt
            h[k:] = b_opt
            diff = N/2.0 - h.sum()
            if abs(diff) > 1e-15:
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
            best_h = h.copy()
        
        if time.time() > deadline - 60:
            break
    
    # Try 3-block
    if time.time() < deadline - 30:
        h3, s3 = _optimize_3block(deadline - 30)
        if h3 is not None and s3 < best_score:
            best_score = s3
            best_h = h3
    
    # Try 4-block with differential evolution
    if time.time() < deadline - 15:
        h4, s4 = _optimize_4block(deadline - 10)
        if h4 is not None and s4 < best_score:
            best_score = s4
            best_h = h4
    
    if best_h is None:
        # Fallback
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = h
    
    return best_h
