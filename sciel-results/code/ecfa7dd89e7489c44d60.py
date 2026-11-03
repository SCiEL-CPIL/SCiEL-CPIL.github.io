"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated overlap computation with Nesterov accelerated projected
gradient on a log-sum-exp smooth surrogate. Multi-phase beta schedule for
coarse-to-fine optimization with exact projection onto the feasible set.
"""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    # Quick check if already feasible
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    # Ensure bracket
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    # Bisection for tau
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh)."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _nesterov_spectral(h0, beta, n_steps=5000, lr=None, time_budget=10.0, max_freq=None):
    """
    Nesterov accelerated gradient on spectral coefficients.
    Optimizes H[1..K] and H[K+1..N-1] (conjugates) while fixing H[0].
    h is recovered via irfft.
    A quadratic penalty is added for h outside [0,1].
    """
    t_start = time.time()
    if max_freq is None:
        max_freq = N // 2
    
    # Initialize spectral coefficients
    H = np.fft.rfft(h0)
    # H[0] is fixed to N/2 to satisfy sum constraint
    H[0] = _TARGET
    
    # Mask for active frequencies
    mask = np.zeros(N // 2 + 1, dtype=bool)
    mask[1:max_freq+1] = True
    
    # Extract active coefficients (real and imaginary parts for k>0, real for k=0 and k=N/2 if even)
    # For simplicity, we optimize the real vector of active coefficients.
    # rfft output: [H0, H1, ..., H_{N/2}]
    # H_k and H_{N-k} are conjugates.
    # We need to store H[1..max_freq] complex.
    
    # Let's just store the full H[1..max_freq] as a complex vector.
    active_idx = np.arange(1, max_freq + 1)
    H_active = H[active_idx].copy()
    
    y_active = H_active.copy()
    best_h = np.fft.irfft(H, N)
    best_h = _project(best_h) # Ensure initial is feasible
    best_val = _smooth_obj_and_grad(best_h, beta)[0]
    
    if lr is None:
        lr = 0.5 # Initial guess, will be adjusted by backtracking
    
    t_k = 1.0
    
    def _get_h_from_active(H_act):
        H_temp = np.zeros_like(H)
        H_temp[0] = _TARGET
        H_temp[1:max_freq+1] = H_act
        # If N is even, H[N/2] is real. rfft handles this.
        # irfft expects the correct structure.
        h_temp = np.fft.irfft(H_temp, N)
        return h_temp

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        # Evaluate at y
        H_y_temp = np.zeros_like(H)
        H_y_temp[0] = _TARGET
        H_y_temp[1:max_freq+1] = y_active
        y_h = np.fft.irfft(H_y_temp, N)
        
        val_y, grad_h_y = _smooth_obj_and_grad(y_h, beta)
        
        # Add penalty gradient for bounds
        # Penalty: sum(relu(-h)^2 + relu(h-1)^2)
        # Grad: -2*relu(-h) + 2*relu(h-1)
        pen_grad = np.zeros(N)
        neg_viol = np.maximum(-y_h, 0)
        pos_viol = np.maximum(y_h - 1, 0)
        pen_grad = -2.0 * neg_viol + 2.0 * pos_viol
        # Scale penalty gradient to match objective scale (objective is ~0.5-1.0)
        # Penalty should be significant if violated.
        pen_grad *= 10.0 
        
        total_grad_h = grad_h_y + pen_grad
        
        # Map gradient from h-space to H_active-space
        # h = irfft(H)
        # This is a linear transformation. 
        # grad_H = F * grad_h ? 
        # Actually, it's easier to compute the gradient of the objective with respect to H_active directly?
        # No, chain rule: dObj/dH_act = (dObj/dh) * (dh/dH_act)
        # dh/dH_act is the inverse FFT matrix restricted to active freqs.
        # Since H is Hermitian, this is effectively the real part of the FFT of the gradient?
        # Let's use the property: <grad_h, delta_h> = <grad_H, delta_H>
        # We can compute the full gradient in H-space via FFT and extract the active parts.
        grad_H_full = np.fft.rfft(total_grad_h)
        grad_active = grad_H_full[1:max_freq+1]
        
        # Nesterov step
        H_new_active = y_active - lr * grad_active
        
        # Reconstruct h for new point
        H_new_temp = np.zeros_like(H)
        H_new_temp[0] = _TARGET
        H_new_temp[1:max_freq+1] = H_new_active
        h_new = np.fft.irfft(H_new_temp, N)
        
        # Compute objective at new point
        val_new, _ = _smooth_obj_and_grad(h_new, beta)
        
        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            H_new_active = y_active - lr * grad_active
            H_new_temp[1:max_freq+1] = H_new_active
            h_new = np.fft.irfft(H_new_temp, N)
            val_new, _ = _smooth_obj_and_grad(h_new, beta)
        else:
            lr *= 0.5
            # If backtracking fails, we might need to project or just accept a small step
            # For now, accept the last computed h_new and H_new_active
            
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y_active = H_new_active + ((t_k - 1.0) / t_new) * (H_new_active - H_active)
        # Project y_active? No, spectral space is unconstrained (except H0 fixed).
        # But we should ensure y_active doesn't blow up.
        
        H_active = H_new_active
        t_k = t_new
        
        # Update best
        # We need to ensure h_new is feasible for the "best" tracking?
        # The score function _score uses _overlap_values which assumes h is in [0,1]?
        # No, _score just computes the value. But the task requires h in [0,1].
        # We should project h_new before comparing scores to ensure validity.
        h_new_proj = _project(h_new)
        score_new = _score(h_new_proj)
        
        # Also check the smooth value for the Nesterov logic
        # The Nesterov logic used val_new (smooth value of unprojected h_new).
        # This is fine for the optimization step, but for "best" we need the true score of the projected point.
        
        if score_new < best_score_global: # Need to pass best_score or return it
             pass # We'll handle best tracking in construct
        
        # Actually, let's return the best h found in this run based on true score.
        # We need to track best_h and best_score inside this function.
        
    # The above loop structure is getting messy with "best" tracking.
    # Let's refactor to track best_h and best_score inside _nesterov_spectral.
    pass


def _nesterov_spectral(h0, beta, n_steps=5000, lr=None, time_budget=10.0, max_freq=None):
    """
    Nesterov accelerated gradient on spectral coefficients.
    Optimizes H[1..K] while fixing H[0] = N/2.
    Uses a penalty for h outside [0,1].
    """
    t_start = time.time()
    if max_freq is None:
        max_freq = N // 2
    
    H = np.fft.rfft(h0)
    H[0] = _TARGET
    
    active_idx = np.arange(1, max_freq + 1)
    H_active = H[active_idx].copy()
    
    y_active = H_active.copy()
    best_h = _project(np.fft.irfft(H, N))
    best_score = _score(best_h)
    
    if lr is None:
        lr = 0.1
    
    t_k = 1.0
    H_prev_active = H_active.copy()
    
    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        # Evaluate at y
        H_y_temp = np.zeros_like(H)
        H_y_temp[0] = _TARGET
        H_y_temp[1:max_freq+1] = y_active
        y_h = np.fft.irfft(H_y_temp, N)
        
        val_y, grad_h_y = _smooth_obj_and_grad(y_h, beta)
        
        # Penalty for bounds
        neg_viol = np.maximum(-y_h, 0)
        pos_viol = np.maximum(y_h - 1, 0)
        pen_grad = -2.0 * neg_viol + 2.0 * pos_viol
        pen_grad *= 10.0 
        
        total_grad_h = grad_h_y + pen_grad
        
        # Gradient in spectral space
        grad_H_full = np.fft.rfft(total_grad_h)
        grad_active = grad_H_full[1:max_freq+1]
        
        # Nesterov step
        H_new_active = y_active - lr * grad_active
        
        # Reconstruct h
        H_new_temp = np.zeros_like(H)
        H_new_temp[0] = _TARGET
        H_new_temp[1:max_freq+1] = H_new_active
        h_new = np.fft.irfft(H_new_temp, N)
        
        # Smooth objective at new point (for backtracking)
        val_new, _ = _smooth_obj_and_grad(h_new, beta)
        
        # Backtracking: ensure decrease from val_y
        for _ in range(8):
            if val_new <= val_y - 1e-10:
                break
            lr *= 0.7
            H_new_active = y_active - lr * grad_active
            H_new_temp[1:max_freq+1] = H_new_active
            h_new = np.fft.irfft(H_new_temp, N)
            val_new, _ = _smooth_obj_and_grad(h_new, beta)
        else:
            lr *= 0.5
            
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y_active = H_new_active + ((t_k - 1.0) / t_new) * (H_new_active - H_prev_active)
        H_prev_active = H_new_active
        t_k = t_new
        
        # Update best based on true score of projected point
        h_new_proj = _project(h_new)
        score_new = _score(h_new_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new_proj

    return best_h, best_score

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Constant
    h_const = np.full(N, 0.5)
    _update_best(h_const)
    
    # Sine
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    _update_best(h_alt)
    
    # Random
    h_rand = np.random.uniform(0.1, 0.9, N)
    _update_best(h_rand)
    
    # More diverse candidates
    h_cos2 = 0.5 + 0.3 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)
    
    h_saw = 0.5 + 0.4 * (2.0 * (i / N - 0.5))
    _update_best(h_saw)
    
    # 2. Optimization Phases (Spectral Space)
    
    # Phase 1: Low-frequency optimization (smooth prior)
    # This enforces the "pure tone" structure by restricting to low frequencies.
    if time.time() - t0 < _DEADLINE - 20:
        # Use the best initial candidate as start
        h_start = best_h if best_h is not None else h_cos
        h_opt, s = _nesterov_spectral(h_start, beta=50.0, n_steps=2000, time_budget=15.0, max_freq=5)
        _update_best(h_opt)
        
    # Phase 2: Medium-frequency refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt, s = _nesterov_spectral(best_h, beta=100.0, n_steps=3000, time_budget=10.0, max_freq=15)
        _update_best(h_opt)
        
    # Phase 3: Full-spectrum polish (allow all frequencies, but start from smooth)
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt, s = _nesterov_spectral(best_h, beta=200.0, n_steps=2000, time_budget=5.0, max_freq=N//2)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _project(best_h)
    return best_h
