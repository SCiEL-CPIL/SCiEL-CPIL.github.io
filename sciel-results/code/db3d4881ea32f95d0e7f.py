import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
DEADLINE = 130.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return DEADLINE - _elapsed()


def _autoconv_full(h):
    """Full autoconvolution: length 2N-1."""
    return np.convolve(h, h, mode='full')


def _ratio(h, beta=50.0):
    """
    Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) using L1 normalization.
    With h_norm = h/||h||_1: ratio = ||h_norm||_2^2 / ||h_norm*h_norm||_inf
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv_full(h_norm)
    # Soft max for differentiability
    cmax = np.max(conv)
    if cmax <= 0:
        return 0.0
    shifted = beta * (conv - cmax)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted)) / beta
    soft_max = cmax + log_sum_exp
    if soft_max <= 0:
        return 0.0
    return l2_sq / soft_max


def _ratio_grad(h, beta=50.0):
    """
    Compute ratio and gradient using L1 normalization.
    ratio = ||h_norm||_2^2 / ||h_norm * h_norm||_inf
    where h_norm = h / ||h||_1
    """
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0, np.zeros_like(h)
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = _autoconv_full(h_norm)
    cmax = np.max(conv)
    if cmax <= 0:
        return 0.0, np.zeros_like(h)
    
    # Soft max weights
    shifted = beta * (conv - cmax)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = cmax + np.log(np.sum(exp_shifted)) / beta
    
    ratio = l2_sq / soft_max
    
    # Gradient computation
    # h_norm = h / l1
    # d(h_norm_i)/d(h_j) = (delta_ij * l1 - h_i) / l1^2
    # Let u_i = d(h_norm_i)/d(h_j) = (delta_ij - h_i/l1) / l1 = (delta_ij - h_norm_i) / l1
    
    # l2_sq = sum(h_norm^2), d(l2_sq)/d(h_j) = 2 * sum_i h_norm_i * u_i = 2 * (2*h_norm_j - 2*h_norm_j) / l1 ... 
    # Actually: d(l2_sq)/d(h_j) = 2 * sum_i h_norm_i * d(h_norm_i)/d(h_j)
    # = 2 * sum_i h_norm_i * (delta_ij - h_norm_i) / l1
    # = 2 * (h_norm_j - sum_i h_norm_i^2) / l1
    # = 2 * (h_norm_j - l2_sq) / l1
    
    dl2 = 2.0 * (h_norm - l2_sq) / l1
    
    # d(soft_max)/d(h_j) = sum_k weights[k] * d(conv_k)/d(h_j)
    # conv_k = sum_i h_norm_i * h_norm_{k-i} (for valid indices)
    # d(conv_k)/d(h_j) = sum_i [u_i * h_norm_{k-i} + h_norm_i * u_{k-i}]
    # where u_i = (delta_ij - h_norm_i) / l1
    
    # This is complex; let's compute it numerically-ish but analytically:
    # d(conv_k)/d(h_j) = (1/l1) * [sum_i (delta_ij - h_norm_i) * h_norm_{k-i} + sum_i h_norm_i * (delta_{k-i,j} - h_norm_{k-i})]
    # = (1/l1) * [h_norm_{k-j} - sum_i h_norm_i * h_norm_{k-i} + h_norm_{k-j} - sum_i h_norm_i * h_norm_{k-i}]
    # Wait, this isn't quite right. Let me be more careful.
    
    # conv_k = sum_i h_norm[i] * h_norm[k-i] where indices are valid
    # d(conv_k)/d(h_j) = sum_i [d(h_norm_i)/d(h_j) * h_norm_{k-i} + h_norm_i * d(h_norm_{k-i})/d(h_j)]
    # = (1/l1) * sum_i [(delta_ij - h_norm_i) * h_norm_{k-i} + h_norm_i * (delta_{k-i,j} - h_norm_{k-i})]
    # = (1/l1) * [sum_i delta_ij * h_norm_{k-i} - sum_i h_norm_i * h_norm_{k-i} + sum_i h_norm_i * delta_{k-i,j} - sum_i h_norm_i * h_norm_{k-i}]
    # = (1/l1) * [h_norm_{k-j} - conv_k + h_norm_{k-j} - conv_k]  (when indices valid)
    # = (1/l1) * [2*h_norm_{k-j} - 2*conv_k]
    
    # But we need to handle boundary cases. Let's just compute d(conv)/dh numerically via the formula:
    # Actually, let's use a simpler approach: compute gradient of soft_max w.r.t. h_norm first, then chain rule.
    
    # d(soft_max)/d(h_norm_i) = sum_k weights[k] * d(conv_k)/d(h_norm_i)
    # conv_k = sum_m h_norm[m] * h_norm[k-m]
    # d(conv_k)/d(h_norm_i) = h_norm[k-i] + h_norm[i-k] (with valid indices)
    
    d_softmax_dn = np.zeros(N)
    for k in range(2 * N - 1):
        wk = weights[k]
        if wk < 1e-15:
            continue
        for i in range(N):
            # conv_k = sum_m h_norm[m]*h_norm[k-m]
            # d(conv_k)/d(h_norm_i): h_norm_i appears when m=i (then h_norm[k-i]) and when k-m=i i.e. m=k-i (then h_norm[i])
            # So d(conv_k)/d(h_norm_i) = h_norm[k-i] (if 0<=k-i<N) + h_norm[k-i] (if 0<=k-i<N)
            # Wait: conv_k = sum_m h_norm[m] * h_norm[k-m]
            # Term with h_norm[i]: when m=i, term is h_norm[i]*h_norm[k-i], derivative w.r.t h_norm[i] is h_norm[k-i]
            # Term with h_norm[k-i]: when m=k-i, term is h_norm[k-i]*h_norm[i], derivative w.r.t h_norm[i] is h_norm[k-i]
            # So total: h_norm[k-i] + h_norm[k-i] = 2*h_norm[k-i] if 0<=k-i<N, else h_norm[k-i] if only one term valid
            j = k - i
            if 0 <= j < N:
                # Both terms: h_norm[i]*h_norm[j] and h_norm[j]*h_norm[i]
                # But these are the same term counted twice only if i != j
                # Actually conv_k = sum_m h_norm[m]*h_norm[k-m], each pair (m, k-m) appears once
                # d(conv_k)/d(h_norm_i) = h_norm[k-i] if 0<=k-i<N (from m=i term)
                # + h_norm[k-i] if 0<=k-i<N (from m=k-i term, which is h_norm[k-i]*h_norm[i], deriv is h_norm[i]... wait)
                # Let me redo: conv_k = sum_m h_norm[m] * h_norm[k-m]
                # d/d(h_norm[i]) conv_k = sum_m [delta_{m,i} * h_norm[k-m] + h_norm[m] * delta_{k-m,i}]
                # = h_norm[k-i] (if 0<=k-i<N) + h_norm[k-i] (if 0<=k-i<N, since m=k-i gives h_norm[k-i]*h_norm[i], deriv w.r.t h_norm[i] is h_norm[k-i])
                # So = 2*h_norm[k-i] if 0<=k-i<N, else 0
                d_softmax_dn[i] += wk * 2.0 * h_norm[j]
            # Also need to check if i == k-i, i.e., k=2i, then it's h_norm[i]^2, derivative is 2*h_norm[i]
            # But the formula above already handles this: if j=i, we get 2*h_norm[i]
    
    # Now chain rule: d(ratio)/d(h_j) = (dl2_sq/d(h_j) * soft_max - l2_sq * d(soft_max)/d(h_j)) / soft_max^2
    # d(soft_max)/d(h_j) = sum_i d(soft_max)/d(h_norm_i) * d(h_norm_i)/d(h_j)
    # = sum_i d_softmax_dn[i] * (delta_ij - h_norm_i) / l1
    
    d_softmax_dh = np.zeros(N)
    for i in range(N):
        d_softmax_dh += d_softmax_dn[i] * (-h_norm[i]) / l1
    d_softmax_dh += d_softmax_dn / l1  # the delta_ij part: for each j, d_softmax_dn[j] / l1
    
    grad = (dl2 * soft_max - l2_sq * d_softmax_dh) / (soft_max * soft_max)
    
    return ratio, grad


def _objective_neg(x, beta=50.0):
    """Negative ratio for minimization."""
    x = np.maximum(x, 0.0)
    r, _ = _ratio_grad(x, beta)
    return -r


def _objective_neg_grad(x, beta=50.0):
    """Negative ratio and gradient."""
    x = np.maximum(x, 0.0)
    r, g = _ratio_grad(x, beta)
    return -r, -g


def _dft_parameterize(z):
    """h = |FFT(z)|^2, ensures h >= 0."""
    H = np.fft.fft(z)
    return np.abs(H) ** 2


def _dft_inverse(h):
    """Given h >= 0, find z such that |FFT(z)|^2 = h (approximately)."""
    H_mags = np.sqrt(np.maximum(h, 0.0) + 1e-30)
    H = H_mags.astype(complex)
    z = np.fft.ifft(H)
    return z


def _z_objective(xz, beta=50.0):
    """Objective in z-space."""
    z = xz[:N] + 1j * xz[N:]
    h = _dft_parameterize(z)
    return _objective_neg(h, beta)


def _z_objective_grad(xz, beta=50.0):
    """Gradient in z-space using chain rule."""
    z = xz[:N] + 1j * xz[N:]
    h = _dft_parameterize(z)
    r, gh = _ratio_grad(h, beta)
    
    # h = |FFT(z)|^2, h_k = |H_k|^2 where H_k = sum_j z_j * exp(-2*pi*i*j*k/N)
    # H_k = sum_j z_j * W_kj where W_kj = exp(-2*pi*i*j*k/N)
    # h_k = H_k * conj(H_k)
    # dh_k/d(Re z_j) = Re(conj(H_k) * W_kj)
    # dh_k/d(Im z_j) = Im(conj(H_k) * W_kj)
    
    H = np.fft.fft(z)
    # grad_Re[j] = sum_k gh[k] * Re(conj(H_k) * W_kj)
    # = Re(sum_k gh[k] * conj(H_k) * W_kj)
    # sum_k gh[k] * conj(H_k) * exp(-2*pi*i*j*k/N) = FFT(gh * conj(H))_j
    f = gh * np.conj(H)
    grad_complex = np.fft.fft(f)
    
    grad = np.zeros(2 * N)
    grad[:N] = np.real(grad_complex)
    grad[N:] = np.imag(grad_complex)
    
    return grad


def _optimize_z(z0, max_iter=300, beta=50.0):
    """Optimize in z-space."""
    if _elapsed() > DEADLINE - 5:
        h = _dft_parameterize(z0)
        return h, _ratio(h, beta)
    
    xz0 = np.concatenate([z0.real, z0.imag])
    try:
        res = minimize(_z_objective, xz0, jac=_z_objective_grad, args=(beta,),
                       method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        xz = res.x
        z = xz[:N] + 1j * xz[N:]
        h = _dft_parameterize(z)
        return h, _ratio(h, beta)
    except Exception:
        h = _dft_parameterize(z0)
        return h, _ratio(h, beta)


def _local_optimize(h0, max_iter=500, beta=50.0):
    """Direct optimization of h with non-negativity."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0, beta)
    
    h0 = np.array(h0, dtype=float)
    h0 = np.maximum(h0, 0.0)
    if np.sum(h0) <= 0:
        return h0, 0.0
    
    bounds = [(0.0, None)] * N
    try:
        res = minimize(_objective_neg, h0, jac=_objective_neg_grad, args=(beta,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        x = np.maximum(res.x, 0.0)
        f = _ratio(x, beta)
        f0 = _ratio(h0, beta)
        if f > f0:
            return x, f
    except Exception:
        pass
    
    return h0, _ratio(h0, beta)


def _initializations():
    """Generate diverse initializations."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    shapes = []
    
    # Uniform
    shapes.append(np.ones(N))
    
    # Gaussians
    for center in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        for width in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
            s = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            shapes.append(s)
    
    # Lorentzians
    for center in [-0.15, 0.0, 0.15]:
        for gamma in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
            s = gamma ** 2 / ((x - center) ** 2 + gamma ** 2)
            shapes.append(s)
    
    # Power laws
    for k in [1.5, 2, 3, 4, 5, 6]:
        for side in ['right', 'left', 'both']:
            s = np.zeros(N)
            if side == 'right':
                mask = x > 0
                s[mask] = x[mask] ** k
            elif side == 'left':
                mask = x < 0
                s[mask] = (-x[mask]) ** k
            else:
                s = np.abs(x) ** k
            if np.max(s) > 0:
                shapes.append(s)
    
    # Two Gaussians
    for c1, c2 in [(-0.15, 0.15), (-0.1, 0.1), (-0.2, 0.2), (-0.05, 0.05)]:
        for w in [0.03, 0.05, 0.08, 0.12]:
            s = np.exp(-((x - c1) ** 2) / (2 * w ** 2)) + np.exp(-((x - c2) ** 2) / (2 * w ** 2))
            shapes.append(s)
    
    # Sinc-like
    for width in [0.05, 0.1, 0.15]:
        for center in [0.0, 0.05, -0.05]:
            t = (x - center) / width
            s = np.sinc(t) ** 2
            s = np.maximum(s, 0)
            if np.max(s) > 0:
                shapes.append(s)
    
    # Triangles
    for width in [0.05, 0.1, 0.15, 0.2]:
        s = np.clip(1.0 - np.abs(x) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)
    
    # Boxes
    for width in [0.02, 0.05, 0.1, 0.15]:
        s = (np.abs(x) < width).astype(float)
        shapes.append(s)
    
    # Random smooth
    rng = np.random.default_rng(42)
    for _ in range(20):
        base = rng.random(N)
        kernel = np.exp(-np.arange(-8, 9) ** 2 / 8.0)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode='same')
        base = np.maximum(base, 0)
        if np.max(base) > 0:
            shapes.append(base)
    
    # Pure random
    for _ in range(20):
        shapes.append(rng.random(N))
    
    # Bimodal
    for d in [0.05, 0.1, 0.15, 0.2]:
        for sig in [0.01, 0.02, 0.03, 0.05]:
            s = np.exp(-(x - d) ** 2 / (2 * sig ** 2)) + np.exp(-(x + d) ** 2 / (2 * sig ** 2))
            if np.max(s) > 0:
                shapes.append(s)
    
    return shapes


def _coordinate_refine(h, max_rounds=5, beta=50.0):
    """Coordinate descent refinement."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h, beta)
    
    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 8:
            break
        improved = False
        for i in range(N):
            if _elapsed() > DEADLINE - 8:
                break
            orig = h[i]
            for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01,
                          0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                h[i] = max(0.0, orig + delta)
                f = _ratio(h, beta)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break
    
    return h, best_f


def _polish(h, beta=80.0):
    """High-precision final polish."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h, beta)
    
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.sum(h) <= 0:
        return h, 0.0
    
    bounds = [(0.0, None)] * N
    
    # L-BFGS-B tight
    try:
        res = minimize(_objective_neg, h, jac=_objective_neg_grad, args=(beta,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200})
        x = np.maximum(res.x, 0.0)
        f = _ratio(x, beta)
        f0 = _ratio(h, beta)
        if f > f0:
            h = x
    except Exception:
        pass
    
    # DFT polish
    if _elapsed() < DEADLINE - 3:
        z = _dft_inverse(h)
        h_dft, f_dft = _optimize_z(z, max_iter=500, beta=beta)
        f0 = _ratio(h, beta)
        if f_dft > f0:
            h = h_dft
    
    return h, _ratio(h, beta)


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h, 50.0)
    
    starts = _initializations()
    
    # Normalize all starts
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        s = np.maximum(s, 0.0)
        s1 = np.sum(s)
        if s1 > 0:
            norm_starts.append(s / s1)
    
    # Score initializations and sort
    scored = []
    for s in norm_starts:
        f = _ratio(s, 50.0)
        scored.append((f, s))
    scored.sort(key=lambda x: -x[0])
    
    if scored[0][0] > best_f:
        best_f = scored[0][0]
        best_h = scored[0][1].copy()
    
    # Phase 1: Optimize from top initializations
    top_k = min(20, len(scored))
    for i in range(top_k):
        if _time_left() < 15:
            break
        f_init, h_init = scored[i]
        
        # Direct optimization
        h_opt, f_opt = _local_optimize(h_init, max_iter=300, beta=50.0)
        if f_opt > best_f:
            best_f = f_opt
            best_h = h_opt.copy()
        
        # DFT-based optimization
        if _time_left() < 10:
            break
        z_init = _dft_inverse(h_init)
        h_dft, f_dft = _optimize_z(z_init, max_iter=200, beta=50.0)
        if f_dft > best_f:
            best_f = f_dft
            best_h = h_dft.copy()
    
    # Phase 2: Refine best with more iterations
    if _time_left() > 20:
        best_h, best_f = _local_optimize(best_h, max_iter=1000, beta=50.0)
    
    # Phase 3: DFT refinement on best
    if _time_left() > 15:
        z = _dft_inverse(best_h)
        h_dft, f_dft = _optimize_z(z, max_iter=500, beta=50.0)
        if f_dft > best_f:
            best_f = f_dft
            best_h = h_dft.copy()
    
    # Phase 4: Coordinate descent
    if _time_left() > 10:
        best_h, best_f = _coordinate_refine(best_h, max_rounds=5, beta=80.0)
    
    # Phase 5: Polish
    if _time_left() > 5:
        best_h, best_f = _polish(best_h, beta=80.0)
    
    # Phase 6: Random restarts from perturbed best
    if _time_left() > 15:
        rng = np.random.default_rng(999)
        for _ in range(10):
            if _time_left() < 8:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            s1 = np.sum(h_pert)
            if s1 <= 0:
                continue
            h_pert = h_pert / s1
            h_opt, f_opt = _local_optimize(h_pert, max_iter=300, beta=50.0)
            if f_opt > best_f:
                best_f = f_opt
                best_h = h_opt.copy()
    
    # Final polish
    if _time_left() > 3:
        best_h, best_f = _polish(best_h, beta=100.0)
    
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
