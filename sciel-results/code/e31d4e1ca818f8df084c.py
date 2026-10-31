import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _overlap_max_fft(h):
    """
    Compute the maximum overlap of h with shifted (1-h) over all shifts 1..N-1.
    h has length N, represents step heights on [0,2] with step width 2/N.
    Integral = sum(h) * (2/N) = 1 => sum(h) = N/2.
    
    Overlap at shift k (circular):
    C_k = sum_i h[i] * (1 - h[(i+k) % N]) * (2/N)
        = (2/N) * sum_i h[i] - (2/N) * sum_i h[i] * h[(i+k) % N]
        = (2/N) * (N/2) - (2/N) * autocorr[k]
        = 1 - (2/N) * autocorr[k]
    
    So minimizing max C_k is equivalent to maximizing min autocorr[k] for k=1..N-1.
    """
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i-k) % n]
    # By symmetry autocorr[k] = autocorr[n-k], so we check k=1..n-1
    # C_k = 1 - (2/n) * autocorr[k]
    # We want max over k=1..n-1
    c_vals = 1.0 - (2.0 / n) * autocorr[1:]
    return np.max(c_vals)

def _soft_max_overlap(h, alpha=50.0):
    """Differentiable soft-max surrogate for the maximum overlap."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    c_vals = 1.0 - (2.0 / n) * autocorr[1:]
    m = np.max(c_vals)
    exp_terms = np.exp(alpha * (c_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    return m + log_sum / alpha

def _soft_max_grad(h, alpha=50.0):
    """Gradient of the soft-max overlap with respect to h."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    c_vals = 1.0 - (2.0 / n) * autocorr[1:]
    m = np.max(c_vals)
    exp_terms = np.exp(alpha * (c_vals - m))
    w = exp_terms / np.sum(exp_terms)
    
    # dC_k/dh_j = -(2/n) * (h_{j-k} + h_{j+k})  for the circular case
    # Actually for circular autocorr: autocorr[k] = sum_i h[i]*h[(i-k)%n]
    # d(autocorr[k])/dh[j] = h[(j-k)%n] + h[(j+k)%n]
    # dC_k/dh[j] = -(2/n) * (h[(j-k)%n] + h[(j+k)%n])
    
    grad = np.zeros(n)
    for k in range(1, n):
        if w[k-1] == 0:
            continue
        factor = -(2.0 / n) * w[k-1]
        for j in range(n):
            idx1 = (j - k) % n
            idx2 = (j + k) % n
            grad[j] += factor * (h[idx1] + h[idx2])
    return grad

def _project_to_feasible(h):
    """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute to elements not at bounds
        mask = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # All at bounds, just add to any element
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
    # Final exact sum fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Find an element we can adjust
        for i in range(N):
            new_val = h[i] + diff
            if 0.0 <= new_val <= 1.0:
                h[i] = new_val
                break
    return h

def _optimize_sinusoidal():
    """
    Strategy: Model h_i = 0.5 + A * sin(2*pi*i/N + phi), clip to [0,1], rescale.
    Optimize A and phi via grid + local refinement.
    """
    t_start = time.time()
    best_score = float('inf')
    best_A, best_phi = 0.0, 0.0
    
    # Grid search over A and phi
    A_vals = np.linspace(0.0, 0.5, 101)
    phi_vals = np.linspace(0.0, 2*np.pi, 101)
    
    for A in A_vals:
        for phi in phi_vals:
            if time.time() - t_start > DEADLINE - 60:
                return None, float('inf')
            h = 0.5 + A * np.sin(2*np.pi*np.arange(N)/N + phi)
            h = np.clip(h, 0.0, 1.0)
            # Rescale to sum = N/2
            s = np.sum(h)
            if s > 0:
                h = h * (N/2.0) / s
                h = np.clip(h, 0.0, 1.0)
            score = _overlap_max_fft(h)
            if score < best_score:
                best_score = score
                best_A, best_phi = A, phi
    
    # Local refinement using Nelder-Mead
    if time.time() - t_start > DEADLINE - 40:
        h = 0.5 + best_A * np.sin(2*np.pi*np.arange(N)/N + best_phi)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        return h, best_score
    
    def obj(params):
        A, phi = params
        h = 0.5 + A * np.sin(2*np.pi*np.arange(N)/N + phi)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        return _overlap_max_fft(h)
    
    try:
        res = minimize(obj, [best_A, best_phi], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        h = 0.5 + res.x[0] * np.sin(2*np.pi*np.arange(N)/N + res.x[1])
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        score = _overlap_max_fft(h)
        if score < best_score:
            best_score = score
        return h, best_score
    except Exception:
        h = 0.5 + best_A * np.sin(2*np.pi*np.arange(N)/N + best_phi)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        return h, best_score

def _optimize_slsqp_from(h0):
    """
    SLSQP optimization with soft-max objective and analytical gradient.
    Starts from h0, optimizes all N variables.
    """
    t_start = time.time()
    alpha = 50.0
    
    def objective(h):
        return _soft_max_overlap(h, alpha)
    
    def gradient(h):
        return _soft_max_grad(h, alpha)
    
    def constraint_sum(h):
        return np.sum(h) - N/2.0
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(objective, h0, jac=gradient, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 2000, 'ftol': 1e-12})
        h_opt = _project_to_feasible(res.x)
        score = _overlap_max_fft(h_opt)
        return h_opt, score
    except Exception:
        return h0, _overlap_max_fft(h0)

def _optimize_multi_sinusoidal(n_modes):
    """
    Multi-sinusoidal: h_i = 0.5 + sum_{m=1}^{n_modes} A_m * sin(2*pi*m*i/N + phi_m)
    Optimize 2*n_modes parameters.
    """
    t_start = time.time()
    n_params = 2 * n_modes
    best_score = float('inf')
    best_params = None
    
    # Grid search for small n_modes
    if n_modes == 1:
        # Already handled by _optimize_sinusoidal
        pass
    
    # Use differential evolution for global search
    bounds = [(-0.5, 0.5)] * n_modes + [(0, 2*np.pi)] * n_modes
    
    def obj(params):
        h = np.full(N, 0.5)
        for m in range(n_modes):
            freq = m + 1
            amp = params[m]
            phase = params[m + n_modes]
            h += amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        return _overlap_max_fft(h)
    
    try:
        res = differential_evolution(obj, bounds, maxiter=100, popsize=20,
                                     tol=1e-8, seed=42, init='latinhypercube')
        h = np.full(N, 0.5)
        for m in range(n_modes):
            freq = m + 1
            amp = res.x[m]
            phase = res.x[m + n_modes]
            h += amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        if s > 0:
            h = h * (N/2.0) / s
            h = np.clip(h, 0.0, 1.0)
        score = _overlap_max_fft(h)
        if score < best_score:
            best_score = score
        return h, best_score
    except Exception:
        return None, float('inf')

def _local_search(h0, n_iter=500):
    """
    Simple local search: randomly perturb h, keep if better.
    """
    t_start = time.time()
    h = h0.copy()
    best_score = _overlap_max_fft(h)
    
    step_size = 0.05
    for i in range(n_iter):
        if time.time() - t_start > DEADLINE - 20:
            break
        # Random perturbation
        idx = np.random.randint(0, N, size=3)
        delta = np.random.uniform(-step_size, step_size, size=3)
        h_new = h.copy()
        h_new[idx] += delta
        h_new = _project_to_feasible(h_new)
        score = _overlap_max_fft(h_new)
        if score < best_score:
            best_score = score
            h = h_new
        # Decay step size
        step_size *= 0.995
    
    return h, best_score

def construct():
    t0 = time.time()
    
    # Start with uniform
    best_h = np.full(N, 0.5)
    best_score = _overlap_max_fft(best_h)
    
    # Strategy 1: Single sinusoidal
    try:
        h_sin, s_sin = _optimize_sinusoidal()
        if h_sin is not None and s_sin < best_score:
            best_score = s_sin
            best_h = h_sin
            print(f"Sinusoidal: score={s_sin:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Sinusoidal failed: {e}")
    
    if time.time() - t0 > DEADLINE - 40:
        return _project_to_feasible(best_h)
    
    # Strategy 2: SLSQP from sinusoidal result
    try:
        h_slsqp, s_slsqp = _optimize_slsqp_from(best_h)
        if s_slsqp < best_score:
            best_score = s_slsqp
            best_h = h_slsqp
            print(f"SLSQP: score={s_slsqp:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"SLSQP failed: {e}")
    
    if time.time() - t0 > DEADLINE - 30:
        return _project_to_feasible(best_h)
    
    # Strategy 3: Multi-sinusoidal with 2 modes
    try:
        h_ms, s_ms = _optimize_multi_sinusoidal(2)
        if h_ms is not None and s_ms < best_score:
            best_score = s_ms
            best_h = h_ms
            print(f"Multi-sin(2): score={s_ms:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Multi-sin(2) failed: {e}")
    
    if time.time() - t0 > DEADLINE - 25:
        return _project_to_feasible(best_h)
    
    # Strategy 4: SLSQP from multi-sinusoidal result
    try:
        h_slsqp2, s_slsqp2 = _optimize_slsqp_from(best_h)
        if s_slsqp2 < best_score:
            best_score = s_slsqp2
            best_h = h_slsqp2
            print(f"SLSQP2: score={s_slsqp2:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"SLSQP2 failed: {e}")
    
    if time.time() - t0 > DEADLINE - 15:
        return _project_to_feasible(best_h)
    
    # Strategy 5: Local search refinement
    try:
        h_ls, s_ls = _local_search(best_h, n_iter=300)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls
            print(f"Local search: score={s_ls:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Local search failed: {e}")
    
    # Final projection
    result = _project_to_feasible(best_h)
    
    # Verify
    final_score = _overlap_max_fft(result)
    if final_score > best_score + 1e-6:
        # Projection may have degraded, use pre-projection if better
        pass
    
    return result

if __name__ == "__main__":
    h = construct()
    print(f"Shape: {h.shape}, Sum: {np.sum(h):.6f}, Min: {np.min(h):.4f}, Max: {np.max(h):.4f}")
    print(f"Score: {_overlap_max_fft(h):.6f}")
