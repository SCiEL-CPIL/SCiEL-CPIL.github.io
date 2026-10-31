import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 138.0  # safety margin before 150s kill

def _all_overlaps(h):
    """Compute overlap_k for k=1..N-1 using FFT.
    overlap_k = sum_i h[i] * (1 - h[(i-k) mod N])
              = sum_i h[i] - sum_i h[i]*h[(i-k) mod N]
    The second term is circular cross-correlation of h with itself.
    """
    n = len(h)
    total = np.sum(h)
    # Circular autocorrelation: corr[k] = sum_i h[i] * h[(i-k) mod N]
    # Use FFT: corr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    corr = np.fft.ifft(H * np.conj(H)).real
    # corr[k] for k=0..n-1, where corr[k] = sum_i h[i]*h[(i-k) mod n]
    overlaps = total - corr[1:]  # k=1..n-1
    return overlaps

def _max_overlap(h):
    """Maximum overlap over all shifts 1..N-1."""
    overlaps = _all_overlaps(h)
    return np.max(overlaps)

def _smooth_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = _all_overlaps(h)
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _project_to_bounds(h):
    """Project h to [0,1] and fix sum to N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    s = np.sum(h)
    if abs(s - target) > 1e-12:
        diff = target - s
        # Distribute adjustment to elements with room
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        # Final tiny correction
        diff2 = target - np.sum(h)
        if abs(diff2) > 1e-14:
            for i in range(N):
                if 0 < h[i] + diff2 <= 1:
                    h[i] += diff2
                    break
    return h

def _direct_objective(x, beta=50.0):
    """Objective: smooth max overlap with bounds and sum constraint handled by projection."""
    h = _project_to_bounds(x)
    return _smooth_max_overlap(h, beta)

def _direct_gradient(x, beta=50.0, eps=1e-6):
    """Numerical gradient of smooth max overlap."""
    h = _project_to_bounds(x)
    grad = np.zeros_like(x)
    for j in range(N):
        xp = x.copy()
        xp[j] += eps
        hp = _project_to_bounds(xp)
        fp = _smooth_max_overlap(hp, beta)
        
        xm = x.copy()
        xm[j] -= eps
        hm = _project_to_bounds(xm)
        fm = _smooth_max_overlap(hm, beta)
        
        grad[j] = (fp - fm) / (2 * eps)
    return grad

def _optimize_direct(n_restarts=10, timeout=30.0):
    """Optimize directly in coordinate space using L-BFGS-B with smooth max."""
    t_start = time.time()
    best_h = None
    best_score = float('inf')
    
    bounds = [(0.0, 1.0)] * N
    
    for restart in range(n_restarts):
        if time.time() - t_start > timeout:
            break
        
        # Different initializations
        if restart == 0:
            x0 = np.full(N, 0.5)
            # Add small sinusoidal perturbation
            x0 += 0.1 * np.sin(2 * np.pi * np.arange(N) / N * 2)
        elif restart == 1:
            x0 = np.full(N, 0.5)
            x0 += 0.15 * np.cos(2 * np.pi * np.arange(N) / N * 3)
        elif restart == 2:
            x0 = np.full(N, 0.5)
            x0 += 0.1 * np.sin(2 * np.pi * np.arange(N) / N * 5)
        elif restart == 3:
            x0 = np.full(N, 0.5)
            # Block pattern: alternating high/low
            x0[::2] += 0.2
            x0[1::2] -= 0.2
        elif restart == 4:
            x0 = np.full(N, 0.5)
            # Triangular wave
            x0 += 0.1 * np.abs(np.sin(np.pi * np.arange(N) / N))
        else:
            # Random perturbation
            np.random.seed(restart * 123)
            x0 = np.full(N, 0.5)
            x0 += np.random.uniform(-0.15, 0.15, N)
        
        x0 = _project_to_bounds(x0)
        
        try:
            result = minimize(
                _direct_objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 2000}
            )
            h = _project_to_bounds(result.x)
            s = _max_overlap(h)
            if s < best_score:
                best_score = s
                best_h = h
        except Exception:
            pass
    
    if best_h is None:
        return None, float('inf')
    return best_h, best_score

def _pairwise_descent(h, timeout=20.0):
    """Local refinement: adjust pairs (i,j) to maintain sum while reducing max overlap."""
    t_start = time.time()
    best_score = _max_overlap(h)
    step = 0.05
    
    while time.time() - t_start < timeout and step > 1e-6:
        improved = False
        # Try all pairs of indices
        for i in range(N):
            if time.time() - t_start > timeout:
                break
            for j in range(i + 1, N):
                if time.time() - t_start > timeout:
                    break
                # Move mass from i to j or vice versa
                for delta in [step, -step, step * 0.5, -step * 0.5]:
                    if h[i] + delta >= 0 and h[i] + delta <= 1 and \
                       h[j] - delta >= 0 and h[j] - delta <= 1:
                        h_new = h.copy()
                        h_new[i] += delta
                        h_new[j] -= delta
                        s = _max_overlap(h_new)
                        if s < best_score - 1e-12:
                            h = h_new
                            best_score = s
                            improved = True
                            break
        if not improved:
            step *= 0.6
        else:
            # Slowly reduce step
            step *= 0.95
    
    return h, best_score

def construct():
    t0 = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Strategy 1: Direct optimization with multiple restarts
    try:
        h_opt, s_opt = _optimize_direct(n_restarts=15, timeout=50.0)
        if h_opt is not None and s_opt < best_score:
            best_score = s_opt
            best_h = h_opt
        print(f"Direct opt: score={s_opt:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Direct opt failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 2: Pairwise descent refinement
    try:
        h_pd, s_pd = _pairwise_descent(best_h.copy(), timeout=40.0)
        if s_pd < best_score:
            best_score = s_pd
            best_h = h_pd
        print(f"Pairwise descent: score={s_pd:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Pairwise descent failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 3: More direct optimization from current best with perturbation
    if time.time() - t0 < DEADLINE - 30:
        try:
            # Perturb best solution and re-optimize
            for attempt in range(5):
                if time.time() - t0 > DEADLINE - 5:
                    break
                np.random.seed(1000 + attempt)
                x0 = best_h.copy()
                x0 += np.random.uniform(-0.05, 0.05, N)
                x0 = _project_to_bounds(x0)
                result = minimize(
                    _direct_objective, x0, method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'maxfun': 1500}
                )
                h = _project_to_bounds(result.x)
                s = _max_overlap(h)
                if s < best_score:
                    best_score = s
                    best_h = h
            print(f"Refined opt: score={best_score:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Refined opt failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Final pairwise refinement
    try:
        h_final, s_final = _pairwise_descent(best_h.copy(), timeout=15.0)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final
    except Exception:
        pass
    
    # Final cleanup: ensure valid output
    result = np.clip(best_h, 0.0, 1.0)
    target_sum = N / 2.0
    current_sum = np.sum(result)
    if abs(current_sum - target_sum) > 1e-10:
        result = _project_to_bounds(result)
    
    return result

# Quick self-test
if __name__ == "__main__":
    h = construct()
    print(f"Shape: {h.shape}, Sum: {np.sum(h):.6f}, Min: {np.min(h):.4f}, Max: {np.max(h):.4f}")
    print(f"Score: {_max_overlap(h):.6f}")
