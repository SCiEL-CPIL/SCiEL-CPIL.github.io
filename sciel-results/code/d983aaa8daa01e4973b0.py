"""
Zadoff-Chu CAZAC base + direct subgradient optimization for the
third autocorrelation inequality task.

Strategy:
  1. Build a real-valued Zadoff-Chu sequence (CAZAC) of length 400.
     CAZAC sequences have zero non-zero cyclic autocorrelation, so the
     autoconvolution peak is isolated at lag 0.
  2. Use direct subgradient (sign-based) optimization on the 400-dim
     vector to minimize  max(conv(h,h)) / (sum(h))^2.
  3. Keep an explicit wall-clock deadline guard.
"""
import numpy as np
import time

N = 400


def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def _zadoff_chu(m, n):
    """
    Real-valued Zadoff-Chu sequence of length n with parameter m.
    z[k] = exp(-i*pi*m*k*(k+1)/n) for k=0..n-1
    We take the real part to get a real sequence.
    For CAZAC property, m should be coprime to n.
    """
    k = np.arange(n)
    phase = -np.pi * m * k * (k + 1) / n
    return np.cos(phase)  # real part


def _zadoff_chu_complex(m, n):
    """Complex Zadoff-Chu sequence."""
    k = np.arange(n)
    phase = -np.pi * m * k * (k + 1) / n
    return np.exp(1j * phase)


def _autoconv_max_subgrad(h):
    """
    Compute the autoconvolution, find the max element, and return
    (value, gradient-like subgradient vector).
    For the objective f(h) = max_k (conv(h,h))_k / (sum(h))^2,
    at the maximizing index k*, the subgradient w.r.t. h is:
      d/dh_j [conv_k*(h)] / (sum)^2  -  2*conv_k*(h)/(sum^3) * 1
    where conv_k*(h)_j = h_{k*-j} for valid indices (full convolution).
    """
    conv = np.convolve(h, h)
    idx = np.argmax(conv)
    s = np.sum(h)
    s2 = s * s
    val = abs(2.0 * N * conv[idx] / s2)
    
    # Subgradient of max(conv)/s^2 w.r.t. h
    # d(conv[idx])/d(h[j]) = h[idx - j] for j where idx-j is in [0, N-1]
    # i.e., j in [idx-(N-1), idx] intersect [0, N-1]
    grad = np.zeros(N)
    lo = max(0, idx - (N - 1))
    hi = min(N - 1, idx)
    for j in range(lo, hi + 1):
        grad[j] = h[idx - j]
    
    # Chain rule for 1/s^2: d(1/s^2)/d(h[j]) = -2/s^3
    grad = grad / s2 - 2.0 * conv[idx] / (s ** 3)
    
    return val, grad


def _subgradient_step(h, grad, lr):
    """Take a subgradient step and return new h."""
    h_new = h - lr * grad
    return h_new


def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t_start

    def remaining():
        return DEADLINE - elapsed()

    # --- Phase 1: Try multiple Zadoff-Chu parameters and pick the best ---
    best_h = None
    best_val = np.inf

    # Try various m values coprime to 400
    # 400 = 2^4 * 5^2, so m coprime to 400 means m not divisible by 2 or 5
    candidates_m = [1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49]
    
    for m in candidates_m:
        if remaining() < 5.0:
            break
        # Real part of Zadoff-Chu
        h_real = _zadoff_chu(m, N)
        # Also try with a linear phase shift (cyclic shift in frequency domain)
        # and try negation
        for sign in [1.0, -1.0]:
            h_try = sign * h_real
            v = _value(h_try)
            if v < best_val:
                best_val = v
                best_h = h_try.copy()
        
        # Try with a small linear phase to break symmetry
        k = np.arange(N)
        for delta_phase in [0.1, -0.1, 0.3, -0.3]:
            h_mod = _zadoff_chu(m, N) * np.cos(delta_phase * k / N)
            for sign in [1.0, -1.0]:
                h_try = sign * h_mod
                v = _value(h_try)
                if v < best_val:
                    best_val = v
                    best_h = h_try.copy()

    # Also try pure cosine with various frequencies as fallback
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        if remaining() < 3.0:
            break
        k = np.arange(N)
        h_cos = np.cos(2 * np.pi * freq * k / N)
        for sign in [1.0, -1.0]:
            h_try = sign * h_cos
            v = _value(h_try)
            if v < best_val:
                best_val = v
                best_h = h_try.copy()

    # Ensure we have a valid starting point
    if best_h is None or best_val >= 2.0:
        best_h = np.ones(N)
        best_val = 2.0

    # --- Phase 2: Direct subgradient optimization ---
    # The objective is piecewise smooth; subgradient method with decreasing step size
    # can drive the max autocorrelation down.
    
    h_current = best_h.copy()
    val_current = best_val

    # Adaptive learning rate schedule
    # Start with a moderate step, decrease over time
    n_steps_max = 5000  # cap on total steps
    step = 0
    lr_init = 0.05
    lr_decay = 0.999  # slow decay
    
    # Track best
    best_h_opt = best_h.copy()
    best_val_opt = best_val

    # Precompute: we'll do batches of steps with occasional re-evaluation
    while remaining() > 10.0 and step < n_steps_max:
        # Compute subgradient
        v, grad = _autoconv_max_subgrad(h_current)
        
        # Normalize gradient to control step size
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-15:
            break
        
        # Learning rate: scale by 1/sqrt(step) for subgradient convergence
        lr = lr_init / np.sqrt(step + 1)
        # Cap the step to avoid overshooting
        lr = min(lr, 0.5 / g_norm)
        
        h_new = h_current - lr * grad
        
        # Validate
        if not np.all(np.isfinite(h_new)):
            break
        
        # Evaluate every few steps to save time
        if step % 5 == 0:
            v_new = _value(h_new)
            if v_new < best_val_opt:
                best_val_opt = v_new
                best_h_opt = h_new.copy()
        
        h_current = h_new
        step += 1

    # --- Phase 3: Fine-tuning with smaller steps from best found ---
    if remaining() > 5.0:
        h_fine = best_h_opt.copy()
        lr_fine = 0.01
        for i in range(2000):
            if remaining() < 3.0:
                break
            v, grad = _autoconv_max_subgrad(h_fine)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-15:
                break
            lr = lr_fine / np.sqrt(i + 1)
            lr = min(lr, 0.1 / g_norm)
            h_fine = h_fine - lr * grad
            if not np.all(np.isfinite(h_fine)):
                break
            if i % 10 == 0:
                v_check = _value(h_fine)
                if v_check < best_val_opt:
                    best_val_opt = v_check
                    best_h_opt = h_fine.copy()

    # --- Phase 4: Try a few random perturbations + local refinement ---
    if remaining() > 5.0:
        rng = np.random.default_rng(12345)
        for _ in range(20):
            if remaining() < 3.0:
                break
            h_pert = best_h_opt + rng.normal(0, 0.1, N)
            # Quick subgradient refinement
            for i in range(200):
                if remaining() < 2.0:
                    break
                v, grad = _autoconv_max_subgrad(h_pert)
                g_norm = np.linalg.norm(grad)
                if g_norm < 1e-15:
                    break
                lr = 0.01 / np.sqrt(i + 1)
                lr = min(lr, 0.1 / g_norm)
                h_pert = h_pert - lr * grad
                if not np.all(np.isfinite(h_pert)):
                    break
            v_pert = _value(h_pert)
            if v_pert < best_val_opt:
                best_val_opt = v_pert
                best_h_opt = h_pert.copy()

    # --- Final validation ---
    h_final = best_h_opt
    s = np.sum(h_final)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h_final)):
        return np.ones(N)
    
    final_val = _value(h_final)
    if final_val >= 2.0:
        return np.ones(N)
    
    return h_final


if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
