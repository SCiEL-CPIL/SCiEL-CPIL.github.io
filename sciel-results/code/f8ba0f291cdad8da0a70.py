"""Erdos minimum overlap: spectral design + smooth-max optimization + local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
_start_time = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _deadline_ok(margin=2.0):
    return _elapsed() < _TIME_LIMIT - margin


def _h_from_params(params):
    """Build h from spectral parameters [m, A, phase]. A clipped to [0,0.5]."""
    m = int(round(params[0]))
    A = float(np.clip(params[1], 0.0, 0.5))
    phase = float(params[2])
    i = np.arange(N)
    h = 0.5 + A * np.cos(2.0 * np.pi * m * i / N + phase)
    return h


def _overlap_and_grad(params):
    """Smooth log-sum-exp of the overlap profile + analytic gradient wrt params.

    Overlap at shift k: O_k = sum_i h_i (1 - h_{i-k}) = N/2 - autocorr_k.
    autocorr_k = sum_i h_i h_{i-k}  =>  d(autocorr_k)/dtheta = sum_i (dh_i/dtheta) h_{i-k}
    So d(O_k)/dtheta = - sum_i (dh_i/dtheta) h_{i-k} = - ifft(fft(dh) * conj(fft(h)))_k
    """
    h = _h_from_params(params)
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr

    alpha = 60.0
    mmax = np.max(overlaps)
    exp_vals = np.exp(alpha * (overlaps - mmax))
    sum_exp = np.sum(exp_vals)
    val = (mmax + np.log(sum_exp)) / alpha
    weights = exp_vals / sum_exp
    W = np.fft.fft(weights)

    def d_overlap(dh):
        DH = np.fft.fft(dh)
        return -np.fft.ifft(DH * np.conj(H)).real

    i = np.arange(N)
    dm = (2.0 * np.pi / N) * i * (-A * np.sin(2.0 * np.pi * m * i / N + phase))
    dA = np.cos(2.0 * np.pi * m * i / N + phase)
    dph = -A * np.sin(2.0 * np.pi * m * i / N + phase)

    grad = np.array([
        np.sum(weights * d_overlap(dm)),
        np.sum(weights * d_overlap(dA)),
        np.sum(weights * d_overlap(dph)),
    ])
    return val, grad


def _max_overlap(h):
    """Exact max circular overlap (raw value, lower is better)."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return float(N / 2.0 - np.min(autocorr))


def _fix_sum(h):
    """Adjust h to sum exactly N/2, keeping values in [0,1]."""
    target = N / 2.0
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def _local_search(h, max_sweeps=300):
    """Coordinate descent / local search to refine the full vector."""
    h = _fix_sum(h.copy())
    best = _max_overlap(h)
    deltas = [0.05, 0.02, 0.01, 0.005]
    for _ in range(max_sweeps):
        if not _deadline_ok(6.0):
            break
        improved = False
        for i in range(N):
            if not _deadline_ok(6.0):
                break
            for j in range(i + 1, N):
                for d in deltas:
                    h2 = h.copy()
                    h2[i] -= d
                    h2[j] += d
                    if h2[i] >= 0 and h2[j] <= 1:
                        h2 = _fix_sum(h2)
                        ov = _max_overlap(h2)
                        if ov < best - 1e-11:
                            h = h2
                            best = ov
                            improved = True
                            break
                    h2 = h.copy()
                    h2[i] += d
                    h2[j] -= d
                    if h2[i] <= 1 and h2[j] >= 0:
                        h2 = _fix_sum(h2)
                        ov = _max_overlap(h2)
                        if ov < best - 1e-11:
                            h = h2
                            best = ov
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return h, best


def construct():
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    # Multi-start spectral optimization over frequency m and phase.
    for m in range(1, N // 2 + 1):
        if not _deadline_ok(8.0):
            break
        for phase in (0.0, np.pi / 2.0):
            if not _deadline_ok(4.0):
                break
            x0 = np.array([float(m), 0.25, phase])
            try:
                res = minimize(
                    _overlap_and_grad, x0, method='L-BFGS-B', jac=True,
                    bounds=[(0.0, N - 1.0), (0.0, 0.5), (-np.pi, np.pi)],
                    options={'maxiter': 200, 'maxls': 20},
                )
                h = _h_from_params(res.x)
                h = _fix_sum(h)
                sc = _max_overlap(h)
                if sc < best_score - 1e-12:
                    best_score = sc
                    best_h = h.copy()
            except Exception:
                pass

    # Local polish on the best full vector.
    if _deadline_ok(8.0):
        h, sc = _local_search(best_h, max_sweeps=300)
        if sc < best_score - 1e-12:
            best_score = sc
            best_h = h.copy()

    best_h = _fix_sum(best_h)
    return best_h
