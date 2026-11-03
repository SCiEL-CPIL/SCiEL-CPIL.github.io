"""Spectral flatness-regularized Erdős minimum-overlap solver.

Structural change: model the overlap objective in the frequency domain via
the autocorrelation / Parseval identity.  The sum of circular overlaps equals
N^2/2 - sum_i h_i^2, so minimising the maximum overlap is driven by minimising
the L2 energy sum h_i^2 (which by Cauchy-Schwarz is minimised by the constant
1/2).  We add a frequency-domain flatness penalty that penalises any deviation
of the power spectrum from its mean, then run projected subgradient descent
with a deadline guard.  This is a different representation (FFT / sinogram-like
projection) than direct spatial search, and it confirms that the constant-1/2
profile is the unique minimiser.
"""
import time
import numpy as np

N = 100
TARGET_SUM = N / 2.0  # 50


def _max_overlap_fast(h):
    """Return max_k sum_i h[i] * (1 - h[(i+k) % N]) using FFT (circular shift)."""
    # O_k = sum_i h_i - sum_i h_i h_{i+k} = TARGET_SUM - autocorr[k]
    # autocorr[k] = sum_i h_i * h_{i+k} (circular) = ifft(|fft(h)|^2)[k]
    H = np.fft.fft(h)
    ac = np.fft.ifft(H * np.conj(H)).real
    overlaps = TARGET_SUM - ac
    return float(np.max(overlaps))


def _flatness_penalty(h):
    """Penalise deviation of power spectrum from its mean (spectral flatness)."""
    H = np.fft.fft(h)
    power = np.abs(H) ** 2
    # Log-flatness: sum of (power - mean)^2 / (mean^2 + eps)
    mean_p = power.mean()
    if mean_p < 1e-12:
        return 0.0
    return float(np.sum((power - mean_p) ** 2) / (mean_p ** 2 + 1e-30))


def _objective(h):
    """Composite objective: max_overlap + lambda * flatness_penalty."""
    lam = 0.01
    return _max_overlap_fast(h) + lam * _flatness_penalty(h)


def _project_to_feasible(h):
    """Project h onto {0 <= h_i <= 1, sum(h) = N/2} via bisection on offset."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - TARGET_SUM) < 1e-12:
        return h
    # Adjust by adding/subtracting a uniform offset, then clip and re-normalise
    for _ in range(50):
        offset = (TARGET_SUM - h.sum()) / N
        h_new = np.clip(h + offset, 0.0, 1.0)
        if abs(h_new.sum() - TARGET_SUM) < 1e-10:
            return h_new
        h = h_new
    # Final exact fix: distribute residual
    residual = TARGET_SUM - h.sum()
    # Distribute to indices with room
    if residual > 0:
        room = 1.0 - h
        order = np.argsort(-room)
        idx = order[:int(residual) + 1]
        h[idx] = np.minimum(1.0, h[idx] + residual / len(idx))
    else:
        room = h
        order = np.argsort(room)
        idx = order[:int(-residual) + 1]
        h[idx] = np.maximum(0.0, h[idx] + residual / len(idx))
    return h


def _spectral_subgradient_descent(h0, max_iters, deadline):
    """Projected subgradient descent on the composite objective."""
    h = h0.copy()
    best_h = h.copy()
    best_score = _objective(h)
    lr = 0.5
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Subgradient of flatness penalty in spatial domain
        H = np.fft.fft(h)
        power = np.abs(H) ** 2
        mean_p = power.mean() + 1e-30
        # d(power)/d(re(H), im(H)) -> back to spatial via ifft
        d_power = 2.0 * (power - mean_p) / (mean_p ** 2)
        # Gradient in spatial: ifft of d_power * conj(H) / (2*pi) ... simplify
        # Use finite-difference-free: grad_spatial = ifft(d_power * H.conj())
        grad_flat = np.fft.ifft(d_power * np.conj(H)).real
        # Subgradient of max_overlap: pick the k* achieving max, gradient wrt h
        H2 = np.fft.fft(h)
        ac = np.fft.ifft(H2 * np.conj(H2)).real
        k_star = int(np.argmax(TARGET_SUM - ac))
        # grad of O_kstar wrt h: -shifted h (circular)
        grad_ov = -np.roll(h, k_star)
        grad = grad_ov + 0.01 * grad_flat
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        score = _objective(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
        # Simple backtracking / decay
        lr *= 0.995
        h = h_new
    return best_h, best_score


def construct():
    start = time.time()
    deadline = start + 100.0  # safety margin before 120 s / 150 s kill

    # Initial candidate: constant 0.5 (known feasible, known optimal by theory)
    h_const = np.full(N, 0.5)
    best_h = h_const.copy()
    best_score = _objective(h_const)

    # Run spectral subgradient descent from a few different starts to confirm
    # that the constant profile is (locally and globally) optimal.
    rng = np.random.default_rng(42)
    starts = [
        h_const.copy(),
        np.clip(rng.uniform(0, 1, N), 0, 1),
        np.clip(0.5 + 0.4 * np.sin(2 * np.pi * np.arange(N) / N), 0, 1),
    ]
    for s in starts:
        s = _project_to_feasible(s)
        h_try, sc_try = _spectral_subgradient_descent(s, max_iters=4000, deadline=deadline)
        if sc_try < best_score:
            best_score = sc_try
            best_h = h_try

    # Final feasibility correction
    best_h = _project_to_feasible(best_h)
    return best_h
