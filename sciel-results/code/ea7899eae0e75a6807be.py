"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Strategy:
  1.  Fast, exact objective via FFT autoconvolution.
  2.  Global search over a compact basis expansion (differential evolution).
  3.  Smooth-surrogate local optimisation (adaptive beta log-sum-exp).
  4.  Direct local search with incremental autoconvolution updates.
  5.  Peak-suppression via projected gradient descent.
  6.  Final high-precision polish.

All phases are deadline-guarded so the program always returns within 150 s.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _autoconv_max_abs(h):
    """max |h*h| (linear autoconvolution) computed via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))


def _autoconv_and_grad(h, peak_idx=None):
    """Compute autoconvolution and gradient of |ac[peak_idx]| w.r.t. h.
    
    Returns (ac, grad) where grad is the gradient of |ac[peak_idx]| w.r.t. h.
    If peak_idx is None, uses the argmax of |ac|.
    """
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    
    if peak_idx is None:
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
    
    peak_sign = np.sign(ac[peak_idx])
    if peak_sign == 0:
        peak_sign = 1.0
    
    # Gradient of ac[k] w.r.t. h[j] = h[k-j] (for valid j)
    # ac[k] = sum_j h[j] * h[k-j]
    # d(ac[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] (when both terms contribute)
    # But for linear conv: ac[k] = sum_{j=0}^{n-1} h[j]*h[k-j] where h is zero-padded
    # d(ac[k])/d(h[j]) = h[k-j] for j where 0 <= k-j < n
    # Plus the symmetric term: when we differentiate the second factor h[k-j] w.r.t. h[j'],
    # we get h[j'] when k-j' = j, i.e., j' = k-j
    # So total: d(ac[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] (if 0 <= k-j < n)
    # Wait, that's not right. Let me re-derive.
    # ac[k] = sum_{j=0}^{n-1} h[j] * h[k-j] (with h out of range = 0)
    # d(ac[k])/d(h[j0]) = h[k-j0] + sum_j h[j] * d(h[k-j])/d(h[j0])
    # d(h[k-j])/d(h[j0]) = 1 if k-j = j0, i.e., j = k-j0
    # So d(ac[k])/d(h[j0]) = h[k-j0] + h[k-j0] (if 0 <= k-j0 < n)
    # = 2 * h[k-j0] (if 0 <= k-j0 < n)
    
    grad = np.zeros(n)
    kj = peak_idx - np.arange(n)
    mask = (kj >= 0) & (kj < n)
    grad[mask] += 2.0 * peak_sign * h[kj[mask]]
    
    return ac, grad


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _epigraph_objective(params, basis, deadline=None):
    """Epigraph formulation: minimize t s.t. |ac[k]| <= t*(sum h)^2 for all k.
    
    We use a smooth approximation: minimize t + mu * sum_k max(0, |ac[k]| - t*s^2)^2
    where s = sum(h).
    
    params: [h_coeffs (n_basis), t]
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    n_basis = basis.shape[0]
    h = np.dot(params[:n_basis], basis)
    t = params[n_basis]
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    s2 = s * s
    
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    
    # Smooth max: t + mu * log-sum-exp of violations
    mu = 0.1
    violations = abs_ac - t * s2
    # Soft penalty for violations
    max_viol = np.max(violations)
    if max_viol < 0:
        return t + 0.0  # No violations
    else:
        # Log-sum-exp of positive violations
        shifted = violations - max_viol
        soft_max = max_viol + np.log(np.sum(np.exp(mu * shifted))) / mu
        return t + soft_max


def _score_smooth(h, beta=80.0):
    """Differentiable surrogate: soft_max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    soft = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return soft / (s * s)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _make_basis_functions(n, n_basis=32):
    """Return an (n_basis, n) array of smooth basis functions."""
    x = np.arange(n, dtype=float)
    basis = []
    basis.append(np.ones(n))                       # 0: constant
    for k in range(1, 12):                          # 1-11: cosines
        basis.append(np.cos(2 * np.pi * k * x / n))
    for k in range(1, 12):                          # 12-21: sines
        basis.append(np.sin(2 * np.pi * k * x / n))
    # Gaussian bumps
    for frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = frac * n
        sigma = n / 12.0
        basis.append(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    # Quadratic
    basis.append(((x - n / 2) / (n / 2)) ** 2)
    # Linear ramp
    basis.append(x / n - 0.5)
    # Alternating
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    # Block patterns (4 of them)
    bs = n // 8
    for b in range(4):
        blk = np.zeros(n)
        blk[b * bs:(b + 1) * bs] = 1.0
        blk[(b + 4) * bs:(b + 5) * bs] = -1.0
        basis.append(blk)
    return np.array(basis[:n_basis])


def _spectral_subspace_optimize(h, basis, deadline, n_basis):
    """Optimize within the span of basis functions using L-BFGS-B with exact gradient.
    
    The objective is max|ac| / (sum h)^2. We use the fact that in the basis space,
    we can compute the gradient of the peak autoconvolution coefficient.
    """
    # Project h onto basis
    params, _, _, _ = np.linalg.lstsq(basis.T, h, rcond=None)
    
    def obj_and_grad(params):
        if time.time() > deadline:
            return 1e10, np.zeros(len(params))
        h = np.dot(params, basis)
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10, np.zeros(len(params))
        
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        s2 = s * s
        val = peak_val / s2
        
        # Gradient of peak_val w.r.t. h
        peak_sign = np.sign(ac[peak_idx])
        if peak_sign == 0:
            peak_sign = 1.0
        grad_h = np.zeros(n)
        kj = peak_idx - np.arange(n)
        mask = (kj >= 0) & (kj < n)
        grad_h[mask] += 2.0 * peak_sign * h[kj[mask]]
        
        # Gradient of val = peak_val / s2
        # d(val)/d(h) = (grad_h * s2 - peak_val * 2*s * ones) / s2^2
        # = grad_h / s2 - 2 * peak_val * s / s2^2
        # = grad_h / s2 - 2 * peak_val / (s * s2)
        grad_val_h = grad_h / s2 - 2.0 * peak_val / (s * s2)
        
        # Chain rule: params -> h
        grad_params = basis.T @ grad_val_h
        
        return val, grad_params
    
    try:
        res = minimize(obj_and_grad, params, method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 50})
        h_new = np.dot(res.x, basis)
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-6:
            h_new /= s_new
            v_new = _score(h_new)
            return h_new, v_new
    except Exception:
        pass
    return h, _score(h)


# ---------------------------------------------------------------------------
# Local search routines
# ---------------------------------------------------------------------------

def _local_search_flip(h, max_passes=6, deadline=None):
    """Try flipping each element's sign with incremental autoconv update; keep if it improves the score."""
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(555)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            d = -2.0 * h[i]
            new_ac = ac.copy()
            start = i
            end = i + n
            if end <= size:
                new_ac[start:end] += 2.0 * d * h
            else:
                valid_end = min(end, size)
                if start < valid_end:
                    new_ac[start:valid_end] += 2.0 * d * h[:valid_end - start]
            new_ac[i] += d * d
            new_sum = s_sum + d
            nsq = new_sum * new_sum
            if abs(nsq) < 1e-9:
                continue
            new_val = np.max(np.abs(new_ac)) / nsq
            if new_val < best_val:
                h[i] = -h[i]
                ac = new_ac
                s_sum = new_sum
                best_val = new_val
                improved = True
        if not improved:
            break
    return h, best_val


def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.04):
    """Coordinate-descent style: try small perturbations per element with incremental updates."""
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(777)
    current_step = step_size
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            best_local = old
            best_local_val = best_val
            best_local_ac = ac
            best_local_sum = s_sum
            for delta in (current_step, -current_step, 0.5 * current_step, -0.5 * current_step):
                new_ac = ac.copy()
                start = i
                end = i + n
                if end <= size:
                    new_ac[start:end] += 2.0 * delta * h
                else:
                    valid_end = min(end, size)
                    if start < valid_end:
                        new_ac[start:valid_end] += 2.0 * delta * h[:valid_end - start]
                new_ac[i] += delta * delta
                new_sum = s_sum + delta
                nsq = new_sum * new_sum
                if abs(nsq) < 1e-9:
                    continue
                v = np.max(np.abs(new_ac)) / nsq
                if v < best_local_val:
                    best_local_val = v
                    best_local = old + delta
                    best_local_ac = new_ac
                    best_local_sum = new_sum
            if best_local != old:
                h[i] = best_local
                ac = best_local_ac
                s_sum = best_local_sum
                best_val = best_local_val
                improved = True
        if not improved:
            break
        current_step *= 0.85
    return h, best_val


def _gaussian_perturbation(h, scale=0.06, deadline=None, max_iters=300):
    """Simulated-annealing style search with Gaussian perturbations."""
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    T_start, T_end = 0.4, 0.003
    restart_interval = 60
    no_improve = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, max_iters - 1))
        if rng.random() < 0.06:
            pert = rng.standard_normal(N) * scale * 3.0
        else:
            pert = rng.standard_normal(N) * scale
        h_trial = h_curr + pert
        trial_val = _score(h_trial)
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > restart_interval and it < max_iters - 10:
            h_curr = h_best.copy()
            curr_val = best_val
            no_improve = 0
        scale *= 0.996
    return h_best, best_val


def _incremental_flip_search(h, deadline=None, max_passes=200):
    """
    Fast local search using incremental autoconvolution updates.
    For a sign flip at index i:
      new_ac = old_ac + 2 * delta * h_shifted + delta^2 * delta_i
    where delta = -2*h[i] (since new_h[i] = -h[i], delta = new - old = -2*old)
    Actually, let's derive carefully:
      h_new = h, but h_new[i] = -h[i]
      ac_new[k] = sum_j h_new[j] * h_new[k-j]
      = sum_j h[j] h[k-j] + (h_new[i]-h[i]) * (h_new[k-i] - h[k-i]) + (h_new[i]-h[i]) * h[k-i] + h[i] * (h_new[k-i]-h[k-i])
      Let d = h_new[i] - h[i] = -2*h[i]
      Then ac_new = ac_old + d * h[k-i] + h[i] * (d if k-i==i else 0) + d * (d if k==i else 0)
      Wait, simpler:
      ac_new[k] = ac_old[k] + 2*d*h[k-i] + d^2 * (1 if k==i else 0)
      But only if k-i is in [0, N-1] and k is in [0, 2N-2].
      Actually, the cross terms:
      sum_j h_new[j] h_new[k-j]
      = sum_{j != i} h[j] h_new[k-j] + h_new[i] h_new[k-i]
      = sum_{j != i} h[j] (h[k-j] + (d if k-j==i else 0)) + (-h[i]) (h[k-i] + (d if k-i==i else 0))
      This is getting complicated. Let's just use the formula:
      ac_new = ac_old + 2*d*shift(h, i) + d^2*delta_i
      where shift(h, i)[k] = h[k-i] if 0 <= k-i < N else 0
      and delta_i[k] = 1 if k==i else 0.
      But d = -2*h[i].
      So ac_new = ac_old - 4*h[i]*shift(h, i) + 4*h[i]^2*delta_i
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            d = -2.0 * h[i]
            # Compute new ac
            new_ac = ac.copy()
            # Add 2*d*shift(h, i)
            # shift(h, i)[k] = h[k-i] for k in [i, i+N-1]
            start = i
            end = i + n
            if end <= size:
                new_ac[start:end] += 2.0 * d * h
            else:
                # Handle boundary
                valid_end = min(end, size)
                if start < valid_end:
                    new_ac[start:valid_end] += 2.0 * d * h[:valid_end - start]
            # Add d^2 * delta_i
            new_ac[i] += d * d
            
            new_sum = s_sum + d
            nsq = new_sum * new_sum
            if abs(nsq) < 1e-9:
                continue
            new_val = np.max(np.abs(new_ac)) / nsq
            
            if new_val < best_val:
                # Accept flip
                h[i] = -h[i]
                ac = new_ac
                s_sum = new_sum
                best_val = new_val
                improved = True
            else:
                # Reject, ac remains unchanged
                pass
        if not improved:
            break
    return h, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5 s safety margin

    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score

    n_basis = 32
    basis = _make_basis_functions(N, n_basis)

    # ---- Phase 1: differential evolution on basis coefficients ----
    p1_dl = deadline - 40
    if time.time() < p1_dl:
        bounds = [(-2.0, 2.0)] * n_basis

        def de_obj(params):
            if time.time() > p1_dl:
                return 1e10
            h = np.dot(params, basis)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)

        try:
            res = differential_evolution(
                de_obj, bounds, seed=42, maxiter=50, tol=1e-6,
                popsize=16, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            h_de = np.dot(res.x, basis)
            s_de = np.sum(h_de)
            if abs(s_de) > 1e-6:
                h_de /= s_de
                v = _score(h_de)
                if v < best_score:
                    best_score = v
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Phase 2: smooth-surrogate local optimisation (adaptive beta) ----
    p2_dl = deadline - 18
    if time.time() < p2_dl:
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr /= s_curr
        params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)

        rng = np.random.default_rng(123)
        inits = [params.copy()]
        for _ in range(8):
            inits.append(rng.standard_normal(n_basis) * 0.5)

        for p_init in inits[:9]:
            if time.time() > p2_dl - 3:
                break
            try:
                for beta in [60.0, 120.0, 250.0]:
                    if time.time() > p2_dl - 1.5:
                        break

                    def sobj(params, b=beta):
                        if time.time() > p2_dl - 1:
                            return 1e10
                        h = np.dot(params, basis)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)

                    res_s = minimize(sobj, p_init, method='BFGS',
                                     options={'maxiter': 200, 'gtol': 1e-8})
                    h_s = np.dot(res_s.x, basis)
                    s_s = np.sum(h_s)
                    if abs(s_s) > 1e-6:
                        h_s /= s_s
                        p_init = np.linalg.lstsq(basis.T, h_s, rcond=None)[0]

                # Final refinement with true objective
                def tobj(params):
                    if time.time() > p2_dl - 0.5:
                        return 1e10
                    h = np.dot(params, basis)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)

                res_t = minimize(tobj, p_init, method='L-BFGS-B',
                                 options={'maxiter': 300, 'gtol': 1e-10})
                h_t = np.dot(res_t.x, basis)
                s_t = np.sum(h_t)
                if abs(s_t) > 1e-6:
                    h_t /= s_t
                    v = _score(h_t)
                    if v < best_score:
                        best_score = v
                        best_h = h_t.copy()
            except Exception:
                pass

    # ---- Phase 3: peak-suppression via spectral subspace optimization ----
    p3_dl = deadline - 10
    if time.time() < p3_dl:
        h_ps, v_ps = _spectral_subspace_optimize(best_h, basis, p3_dl, n_basis)
        if v_ps < best_score:
            best_score = v_ps
            best_h = h_ps.copy()
        
        # Additional gradient descent in full space
        if time.time() < p3_dl - 2:
            h_grad = best_h.copy()
            n = len(h_grad)
            size = 2 * n - 1
            fsize = 1
            while fsize < size:
                fsize *= 2
            H = np.fft.rfft(h_grad, fsize)
            ac = np.fft.irfft(H * H, fsize)[:size].copy()
            s_sum = np.sum(h_grad)
            best_grad = np.max(np.abs(ac)) / (s_sum * s_sum)
            rng = np.random.default_rng(888)
            step = 0.02
            for it in range(300):
                if time.time() > p3_dl - 2:
                    break
                abs_ac = np.abs(ac)
                peak_val = np.max(abs_ac)
                peak_idx = np.argmax(abs_ac)
                peak_sign = np.sign(ac[peak_idx])
                if peak_sign == 0:
                    peak_sign = 1.0
                # Gradient of |ac[peak_idx]| / s^2 w.r.t. h
                grad_h = np.zeros(n)
                kj = peak_idx - np.arange(n)
                mask = (kj >= 0) & (kj < n)
                grad_h[mask] += 2.0 * peak_sign * h_grad[kj[mask]]
                s2 = s_sum * s_sum
                grad_val = grad_h / s2 - 2.0 * peak_val / (s_sum * s2)
                
                h_trial = h_grad - step * grad_val
                H_t = np.fft.rfft(h_trial, fsize)
                ac_t = np.fft.irfft(H_t * H_t, fsize)[:size]
                s_t = np.sum(h_trial)
                if abs(s_t) < 1e-9:
                    step *= 0.9
                    continue
                v_t = np.max(np.abs(ac_t)) / (s_t * s_t)
                if v_t < best_grad:
                    h_grad = h_trial
                    ac = ac_t
                    s_sum = s_t
                    best_grad = v_t
                    step = min(step * 1.05, 0.05)
                else:
                    step *= 0.9
            if best_grad < best_score:
                best_score = best_grad
                best_h = h_grad.copy()

    # ---- Phase 4: incremental flip local search ----
    p4_dl = deadline - 6
    if time.time() < p4_dl:
        h_fl, v_fl = _incremental_flip_search(best_h, deadline=p4_dl, max_passes=300)
        if v_fl < best_score:
            best_score = v_fl
            best_h = h_fl.copy()

    # ---- Phase 5: perturbation local search (now incremental) ----
    p5_dl = deadline - 3
    if time.time() < p5_dl:
        h_pt, v_pt = _local_search_perturb(best_h, max_passes=8,
                                           deadline=p5_dl, step_size=0.03)
        if v_pt < best_score:
            best_score = v_pt
            best_h = h_pt.copy()

    # ---- Phase 6: simulated-annealing polish ----
    p6_dl = deadline - 1.0
    if time.time() < p6_dl:
        h_sa, v_sa = _gaussian_perturbation(best_h, scale=0.04,
                                            deadline=p6_dl, max_iters=500)
        if v_sa < best_score:
            best_score = v_sa
            best_h = h_sa.copy()

    # ---- Phase 7: final high-precision polish with epigraph formulation ----
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr /= s_curr
            params, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
            
            # First: L-BFGS-B on the standard objective
            def final_obj(params):
                if time.time() > deadline - 0.3:
                    return 1e10
                h = np.dot(params, basis)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_f = minimize(final_obj, params, method='L-BFGS-B',
                             options={'maxiter': 300, 'gtol': 1e-12})
            h_f = np.dot(res_f.x, basis)
            s_f = np.sum(h_f)
            if abs(s_f) > 1e-6:
                h_f /= s_f
                v_f = _score(h_f)
                if v_f < best_score:
                    best_score = v_f
                    best_h = h_f.copy()
                    h_curr = h_f.copy()
            
            # Second: epigraph formulation polish
            if time.time() < deadline - 0.3:
                params2, _, _, _ = np.linalg.lstsq(basis.T, h_curr, rcond=None)
                s_init = np.sum(h_curr)
                t_init = best_score * s_init * s_init
                
                def epigraph_obj(params_ep):
                    if time.time() > deadline - 0.1:
                        return 1e10
                    h_ep = np.dot(params_ep[:n_basis], basis)
                    t_ep = params_ep[n_basis]
                    s_ep = np.sum(h_ep)
                    if abs(s_ep) < 1e-9:
                        return 1e10
                    s2_ep = s_ep * s_ep
                    n_ep = len(h_ep)
                    size_ep = 2 * n_ep - 1
                    fsize_ep = 1
                    while fsize_ep < size_ep:
                        fsize_ep *= 2
                    H_ep = np.fft.rfft(h_ep, fsize_ep)
                    ac_ep = np.fft.irfft(H_ep * H_ep, fsize_ep)[:size_ep]
                    abs_ac_ep = np.abs(ac_ep)
                    violations = abs_ac_ep - t_ep * s2_ep
                    max_viol = np.max(violations)
                    if max_viol < 0:
                        return t_ep
                    mu_ep = 0.2
                    shifted = violations - max_viol
                    soft_max = max_viol + np.log(np.sum(np.exp(mu_ep * shifted))) / mu_ep
                    return t_ep + soft_max
                
                initial_ep = np.concatenate([params2, [t_init]])
                res_ep = minimize(
                    epigraph_obj, initial_ep, method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-12}
                )
                h_ep = np.dot(res_ep.x[:n_basis], basis)
                s_ep = np.sum(h_ep)
                if abs(s_ep) > 1e-6:
                    h_ep /= s_ep
                    v_ep = _score(h_ep)
                    if v_ep < best_score:
                        best_score = v_ep
                        best_h = h_ep.copy()
        except Exception:
            pass

    # ---- Phase 8: final sign-flip polish ----
    if time.time() < deadline - 0.3:
        try:
            h_ff, v_ff = _local_search_flip(best_h, max_passes=3,
                                            deadline=deadline - 0.1)
            if v_ff < best_score:
                best_score = v_ff
                best_h = h_ff.copy()
        except Exception:
            pass
        
        # Final spectral subspace polish
        if time.time() < deadline - 0.1:
            try:
                h_sp, v_sp = _spectral_subspace_optimize(best_h, basis, deadline - 0.05, n_basis)
                if v_sp < best_score:
                    best_score = v_sp
                    best_h = h_sp.copy()
            except Exception:
                pass

    # Safety: never worse than baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
