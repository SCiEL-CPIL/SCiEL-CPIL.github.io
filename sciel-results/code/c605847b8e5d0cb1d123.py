"""Fresh design: smooth min-max of cross-correlation h vs (1-h) via projected subgradient descent."""
import time
import numpy as np

N = 100

_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin (seconds) before wall-clock limit

def _overlap_shifts(h):
    """Return overlap value for each circular shift t: sum_i h_i*(1-h_{i-t})."""
    # cross-correlation c[t] = sum_i h_i * h_{i-t}
    # overlap[t] = sum_i h_i - sum_i h_i*h_{i-t} = sum(h) - c[t]
    h = np.asarray(h)
    s = h.sum()
    # circular cross-correlation via FFT
    H = np.fft.fft(h)
    c = np.real(np.fft.ifft(H * np.conj(H)))
    return s - c

def _soft_max(vals, temp):
    """Smooth approximation of max."""
    m = vals.max()
    return m + temp * np.log(np.sum(np.exp((vals - m) / temp)))

def _objective(h, temp):
    """Smoothed max overlap."""
    vals = _overlap_shifts(h)
    return _soft_max(vals, temp)

def _project(h):
    """Project onto {0<=x<=1, sum = N/2} using water-filling."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(200):
        diff = h.sum() - target
        if abs(diff) < 1e-12:
            break
        # adjust only where we can move (interior points)
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if not mask.any():
            break
        step = diff / mask.sum()
        h = h.copy()
        h[mask] -= step
        h = np.clip(h, 0.0, 1.0)
    # final exact fix-up on a free coordinate
    diff = h.sum() - target
    if abs(diff) > 1e-12:
        # find a coordinate we can nudge
        if diff > 0:
            idx = int(np.argmin(h))  # smallest -> can increase
        else:
            idx = int(np.argmax(h))  # largest -> can decrease
        h = h.copy()
        h[idx] = np.clip(h[idx] - diff, 0.0, 1.0)
    return h

def _true_max_overlap(h):
    return float(_overlap_shifts(h).max())

def construct():
    t_start = time.time()
    best = np.full(N, 0.5)
    best_val = _true_max_overlap(best)

    rng = np.random.default_rng(12345)

    # multiple random restarts, each a projected subgradient descent run
    n_restarts = 6
    steps_per = 600
    base_temp = 0.05
    lr_start = 0.02
    lr_end = 0.0005

    for r in range(n_restarts):
        if time.time() - t_start > _DEADLINE - _MARGIN:
            break
        # random init in [0,1], then project
        h = rng.uniform(0.0, 1.0, size=N)
        h = _project(h)

        for it in range(steps_per):
            if time.time() - t_start > _DEADLINE - _MARGIN:
                break
            temp = base_temp * (0.98 ** it)
            # subgradient of soft max: softmax weights over shifts
            vals = _overlap_shifts(h)
            m = vals.max()
            w = np.exp((vals - m) / temp)
            w = w / w.sum()
            # d/dh of overlap[t] = 1 - h_{i-t}  (since overlap = sum h_i - sum h_i h_{i-t})
            # gradient = sum_t w[t] * (1 - h_{i-t})
            # h_{i-t} is a circular shift of h
            grad = np.zeros(N)
            for t in range(N):
                if w[t] < 1e-10:
                    continue
                grad += w[t] * (1.0 - np.roll(h, t))
            # anneal learning rate
            frac = it / max(1, steps_per - 1)
            lr = lr_start * (lr_end / lr_start) ** frac
            h = h - lr * grad
            h = _project(h)

        val = _true_max_overlap(h)
        if val < best_val - 1e-9:
            best_val = val
            best = h.copy()

    return best
