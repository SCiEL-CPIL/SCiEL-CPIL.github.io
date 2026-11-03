import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _subgradient_of_max_overlap(h):
    """
    Returns the subgradient of the max overlap function.
    Specifically, it returns the gradient of the component C_k that is currently maximal.
    """
    n = len(h)
    cs = _linear_overlap_values(h)
    k_max = np.argmax(cs)
    
    # Gradient of C_k = (2/N) * (N/2 - sum(h_i * h_{i+k}))
    # d(C_k)/d(h_j) = -(2/N) * (h_{j+k} + h_{j-k})
    
    # We need h_{j+k} and h_{j-k} for all j.
    # Using np.roll for circular shifts
    h_shift_plus = np.roll(h, -k_max) # h_{j+k}
    h_shift_minus = np.roll(h, k_max) # h_{j-k}
    
    grad = -(2.0 / n) * (h_shift_plus + h_shift_minus)
    
    # Special case k=0: C_0 = (2/N)(N/2 - sum h^2). Grad = -(4/N)h.
    # The formula above gives -(2/N)(h+h) = -(4/N)h, which is correct.
    
    return cs[k_max], grad


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _polyak_subgradient_descent(h0, t_lb, max_iter=5000):
    """
    Subgradient method with Polyak step size.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    best_h = h.copy()
    best_val = _score(h)
    
    # t_lb is the lower bound on the optimal value (best found so far)
    # We use this for the Polyak step: alpha = (f(h) - t_lb) / ||g||^2
    # If f(h) <= t_lb, we are at a better point, update t_lb.
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _subgradient_of_max_overlap(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            t_lb = val # Update lower bound
            
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq < 1e-12:
            break
            
        # Polyak step size
        # If val <= t_lb, step is 0 or negative, which means we are at optimum or better.
        # But we just updated t_lb if val < best_val.
        # If val >= t_lb, step is positive.
        alpha = (val - t_lb) / grad_norm_sq
        
        # Cap the step size to prevent instability
        alpha = min(alpha, 0.1)
        
        h_new = h - alpha * grad
        h_new = _fix_sum(h_new)
        
        h = h_new
        
        # Stagnation check: if no improvement in best_val for a long time, break
        # (Polyak method can oscillate)
        if it % 100 == 99:
             # Optional: check if we are stuck
             pass

    return best_h, best_val


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(10):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))

    h_saw = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N) * np.sign(np.sin(4.0 * np.pi * i / N))
    starts.append(_fix_sum(h_saw))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Subgradient Method with Polyak Step
    # This is the structural change. It uses the global best score as the lower bound t_lb
    # and projects onto the feasible set after each step.
    
    if best_h is not None:
        # Run from the best start found in Phase 1
        h_candidate, s_candidate = _polyak_subgradient_descent(best_h, best_score, max_iter=20000)
        if s_candidate < best_score:
            _update_best(h_candidate)
            
        # Try from a few other promising starts
        for h0 in starts[:5]:
            if time.time() > _DEADLINE - 10:
                break
            # Use the current global best as t_lb to guide the search
            h_candidate, s_candidate = _polyak_subgradient_descent(h0, best_score, max_iter=10000)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 3: Local Polish with Smooth Max (Nesterov)
    # Use the result from Subgradient as a high-quality starting point for a smooth local optimizer.
    # This combines the global reach of Subgradient with the local precision of Nesterov.
    
    def _accelerated_smooth_max(h0, beta, max_iter=500, lr_init=0.05):
        h = np.array(h0, dtype=np.float64, copy=True)
        h = _fix_sum(h)
        h_prev = h.copy()
        t_k = 1.0
        lr = lr_init

        best_val = _score(h)
        best_h = h.copy()

        for it in range(max_iter):
            if time.time() > _DEADLINE - 5:
                break

            # Smooth max objective and gradient
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            val = m + np.log(sum_exp) / beta

            n_fft = 2 * N
            w_pad = np.zeros(n_fft)
            w_pad[:N] = w
            W = np.fft.rfft(w_pad, n=n_fft)
            H = np.fft.rfft(h, n=n_fft)

            conv_wh = np.fft.irfft(W * H, n=n_fft)[:N]

            h_flip = np.zeros(N)
            h_flip[0] = h[0]
            h_flip[1:] = h[1:][::-1]
            h_flip_pad = np.zeros(n_fft)
            h_flip_pad[:N] = h_flip
            HF = np.fft.rfft(h_flip_pad, n=n_fft)
            conv_whf = np.fft.irfft(W * HF, n=n_fft)[:N]

            grad = -(2.0 / N) * (conv_wh + conv_whf)

            h_new = h - lr * grad
            h_new = _fix_sum(h_new)

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
            h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
            h_next = _fix_sum(h_next)

            h_prev = h
            h = h_next
            t_k = t_new

            if it % 100 == 99:
                lr = max(0.001, min(0.1, lr * 0.95))

            if it % 50 == 49:
                current_score = _score(h)
                if current_score < best_val:
                    best_val = current_score
                    best_h = h.copy()

        return best_h, best_val

    if best_h is not None:
        for beta in [100.0, 300.0]:
            if time.time() > _DEADLINE - 15:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta, max_iter=500, lr_init=0.02)
            if s_candidate < best_score:
                _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
