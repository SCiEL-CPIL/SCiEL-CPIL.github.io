import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp

N = 100
DEADLINE = 135.0  # safety margin before 150s kill

def _max_overlap(h):
    """Maximum overlap over all shifts 1..N-1 for h of length N.
    overlap_k = sum_i h[i] * (1 - h[(i-k) mod N])
    """
    n = len(h)
    best = 0.0
    for k in range(1, n):
        shifted_1mh = 1.0 - np.roll(h, -k)
        ov = np.sum(h * shifted_1mh)
        if ov > best:
            best = ov
    return best

def _smooth_max_overlap(h, beta=80.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    n = len(h)
    overlaps = np.empty(n - 1)
    for k in range(1, n):
        shifted_1mh = 1.0 - np.roll(h, -k)
        overlaps[k-1] = np.sum(h * shifted_1mh)
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _fourier_to_h(coeffs, N):
    """Convert Fourier coefficients to h array of length N.
    coeffs[0] = DC term (should be 0.5 for integral constraint)
    coeffs[1:] = complex coefficients for frequencies 1..N/2
    We use a real-valued representation: coeffs = [a0, a1, b1, a2, b2, ...]
    where h[i] = a0 + sum_{k=1}^{K} (a_k * cos(2*pi*k*i/N) + b_k * sin(2*pi*k*i/N))
    """
    n = len(coeffs)
    K = (n - 1) // 2  # number of frequency components
    i = np.arange(N)
    h = np.zeros(N)
    h += coeffs[0]
    for k in range(1, K+1):
        a_k = coeffs[2*k - 1]
        b_k = coeffs[2*k]
        h += a_k * np.cos(2 * np.pi * k * i / N) + b_k * np.sin(2 * np.pi * k * i / N)
    return h

def _h_to_fourier(h, K):
    """Convert h array to Fourier coefficients [a0, a1, b1, ..., aK, bK]."""
    N = len(h)
    i = np.arange(N)
    coeffs = np.zeros(2*K + 1)
    coeffs[0] = np.mean(h)
    for k in range(1, K+1):
        coeffs[2*k - 1] = 2.0 / N * np.sum(h * np.cos(2 * np.pi * k * i / N))
        coeffs[2*k] = 2.0 / N * np.sum(h * np.sin(2 * np.pi * k * i / N))
    return coeffs

def _project_to_bounds(h, N):
    """Project h to [0,1] and fix sum to N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    s = np.sum(h)
    if abs(s - target) > 1e-12:
        # Try to adjust while staying in bounds
        if s > target:
            # Need to decrease sum by (s - target)
            excess = s - target
            # Find elements that can be decreased
            mask = h > 0
            if np.sum(mask) > 0:
                max_decrease = np.sum(h[mask])
                if max_decrease >= excess:
                    # Scale down proportionally
                    h[mask] -= excess / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
                else:
                    h[mask] = 0.0
                    # Still need to decrease more
                    remaining = excess - max_decrease
                    mask2 = h > 0
                    if np.sum(mask2) > 0:
                        h[mask2] -= remaining / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum by (target - s)
            deficit = target - s
            mask = h < 1
            if np.sum(mask) > 0:
                max_increase = np.sum(1 - h[mask])
                if max_increase >= deficit:
                    h[mask] += deficit / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
                else:
                    h[mask] = 1.0
                    remaining = deficit - max_increase
                    mask2 = h < 1
                    if np.sum(mask2) > 0:
                        h[mask2] += remaining / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h

def _optimize_fourier(K, n_restarts=5, timeout=25.0):
    """Optimize using Fourier representation with L-BFGS-B."""
    t_start = time.time()
    n_coeffs = 2*K + 1
    
    def objective(coeffs):
        if time.time() - t_start > timeout:
            return float('inf')
        h = _fourier_to_h(coeffs, N)
        h = _project_to_bounds(h, N)
        return _smooth_max_overlap(h, beta=80.0)
    
    def gradient(coeffs):
        if time.time() - t_start > timeout:
            return np.zeros_like(coeffs)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(coeffs)
        for j in range(len(coeffs)):
            c_plus = coeffs.copy()
            c_plus[j] += eps
            h_plus = _project_to_bounds(_fourier_to_h(c_plus, N), N)
            f_plus = _smooth_max_overlap(h_plus, beta=80.0)
            
            c_minus = coeffs.copy()
            c_minus[j] -= eps
            h_minus = _project_to_bounds(_fourier_to_h(c_minus, N), N)
            f_minus = _smooth_max_overlap(h_minus, beta=80.0)
            
            grad[j] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    best_h = None
    best_score = float('inf')
    
    bounds = [(0.0, 1.0)] + [(-0.5, 0.5)] * K * 2
    
    for restart in range(n_restarts):
        if time.time() - t_start > timeout:
            break
        
        # Initialize with sinusoidal patterns
        if restart == 0:
            # Start with 0.5 + small perturbation
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5
            x0[1] = 0.1  # cos component
            x0[2] = 0.05  # sin component
        elif restart == 1:
            # Higher frequency
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5
            if K >= 2:
                x0[3] = 0.15
                x0[4] = 0.1
        elif restart == 2:
            # Even higher
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5
            if K >= 3:
                x0[5] = 0.12
                x0[6] = 0.08
        else:
            # Random small coefficients
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5
            for k in range(1, min(K+1, 5)):
                x0[2*k-1] = np.random.uniform(-0.2, 0.2)
                x0[2*k] = np.random.uniform(-0.2, 0.2)
        
        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                h = _project_to_bounds(_fourier_to_h(result.x, N), N)
                best_h = h
        except Exception:
            pass
    
    if best_h is None:
        return None, float('inf')
    return best_h, best_score

def _coordinate_descent(h, n_iter=500, timeout=20.0):
    """Local refinement via coordinate descent."""
    t_start = time.time()
    best_score = _max_overlap(h)
    step = 0.02
    
    for iteration in range(n_iter):
        if time.time() - t_start > timeout:
            break
        improved = False
        for i in range(N):
            if time.time() - t_start > timeout:
                break
            # Try increasing and decreasing h[i]
            for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2]:
                h_new = h.copy()
                h_new[i] = np.clip(h[i] + delta, 0.0, 1.0)
                # Check sum constraint approximately
                if abs(np.sum(h_new) - N/2) < 0.01:
                    s = _max_overlap(h_new)
                    if s < best_score - 1e-10:
                        h = h_new
                        best_score = s
                        improved = True
                        break
        if not improved:
            step *= 0.7
            if step < 1e-5:
                break
    
    return h, best_score

def construct():
    t0 = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Strategy 1: Fourier optimization with increasing K
    for K in [2, 4, 6, 8]:
        if time.time() - t0 > DEADLINE - 5:
            break
        try:
            h_f, s_f = _optimize_fourier(K, n_restarts=4, timeout=20.0)
            if h_f is not None and s_f < best_score:
                best_score = s_f
                best_h = h_f
            print(f"Fourier K={K}: score={s_f:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"Fourier K={K} failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 2: Coordinate descent refinement
    try:
        h_cd, s_cd = _coordinate_descent(best_h, n_iter=300, timeout=25.0)
        if s_cd < best_score:
            best_score = s_cd
            best_h = h_cd
        print(f"Coord descent: score={s_cd:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Coord descent failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 3: DE on Fourier coefficients for broader search
    for K in [3, 5]:
        if time.time() - t0 > DEADLINE - 10:
            break
        try:
            n_coeffs = 2*K + 1
            bounds = [(0.0, 1.0)] + [(-0.5, 0.5)] * K * 2
            
            def de_obj(coeffs):
                h = _project_to_bounds(_fourier_to_h(coeffs, N), N)
                return _max_overlap(h)
            
            result = differential_evolution(
                de_obj, bounds, seed=42, maxiter=40,
                popsize=12, tol=1e-7
            )
            if result.fun < best_score:
                best_score = result.fun
                best_h = _project_to_bounds(_fourier_to_h(result.x, N), N)
            print(f"DE Fourier K={K}: score={result.fun:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"DE Fourier K={K} failed: {e}")
    
    # Final cleanup: ensure valid output
    result = np.clip(best_h, 0.0, 1.0)
    target_sum = N / 2.0
    current_sum = np.sum(result)
    if abs(current_sum - target_sum) > 1e-10:
        diff = target_sum - current_sum
        # Distribute adjustment
        mask = (result > 0) & (result < 1)
        if np.sum(mask) > 0:
            result[mask] += diff / np.sum(mask)
            result = np.clip(result, 0.0, 1.0)
        # Final correction
        diff = target_sum - np.sum(result)
        if abs(diff) > 1e-12:
            for i in range(N):
                if 0 < result[i] + diff <= 1:
                    result[i] += diff
                    break
    
    return result

# Quick self-test
if __name__ == "__main__":
    h = construct()
    print(f"Shape: {h.shape}, Sum: {np.sum(h):.6f}, Min: {np.min(h):.4f}, Max: {np.max(h):.4f}")
    print(f"Score: {_max_overlap(h):.6f}")
