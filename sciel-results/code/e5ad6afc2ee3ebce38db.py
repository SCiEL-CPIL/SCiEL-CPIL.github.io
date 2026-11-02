"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Work in the Fourier domain. The overlap at shift k is
    O_k = sum_i h_i (1 - h_{i+k}) = N/2 - R_k
where R_k = sum_i h_i h_{i+k} is the circular autocorrelation. So minimizing
max_k O_k is equivalent to maximizing min_k R_k (and, ideally, making R flat).

The DFT of R is |H|^2 (H = DFT(h)). A flat autocorrelation R_k = N/4 for all k
is achieved by a signal whose amplitude spectrum |H_k| is constant for every
frequency (a "flat spectrum" / Golay-like signal). With zero phase, R is exactly
flat at N/4 and h is constant 1/2 (the seed). We therefore FIX the amplitude
spectrum to be flat and optimize only the PHASES so that the real time-domain
sequence h stays inside [0,1] with sum exactly N/2. This is a smooth problem on
the phase torus, which we solve with projected gradient descent on the phase
angles, using an epigraph (t) formulation to drive the minimum autocorrelation
up (the maximum overlap down).
"""
import time
import numpy as np

N = 100
TARGET = N / 2.0          # required sum
BASE = N / 4.0            # flat autocorrelation level -> overlap N/4


def _time_remaining(deadline):
    return deadline - time.time()


def _compute_overlaps(h):
    """O_k = sum_i h_i (1 - h_{i+k}) for all shifts k (circular)."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    # R_k = sum_i h_i h_{i+k}
    R = np.array([np.dot(h, np.roll(h, k)) for k in range(n)])
    return TARGET - R


def _max_overlap(h):
    return float(np.max(_compute_overlaps(h)))


def _score(h):
    return _max_overlap(h)


def _project_to_box(h, target=TARGET):
    """Project h onto {0<=h<=1, sum h = target} (Euclidean, via bisection on t)."""
    h = np.asarray(h, dtype=float).copy()
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        s = np.sum(np.clip(h - t, 0.0, 1.0))
        if s > target:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    return np.clip(h - t, 0.0, 1.0)


def _h_to_phases(h, amp):
    """Return phase angles whose flat-amplitude spectrum best matches h."""
    H = np.fft.fft(h)
    return np.angle(H / amp)


def _phases_to_h(ph, amp, target=TARGET):
    """Inverse flat-amplitude synthesis + projection to [0,1] with sum target."""
    H = amp * np.exp(1j * ph)
    h = np.fft.ifft(H).real
    return _project_to_box(h, target)


def _epigraph_obj_grad(ph, amp, t):
    """Objective  t + lam*sum_k max(0, t - R_k)  and its gradient w.r.t. phases."""
    h = _phases_to_h(ph, amp)
    n = len(h)
    # R_k = sum_i h_i h_{i+k}
    R = np.array([np.dot(h, np.roll(h, k)) for k in range(n)])
    viol = np.maximum(0.0, t - R)
    obj = t + 1.0 * np.sum(viol)

    # dR_k / dh = h * (roll(h, k) + roll(h, -k))
    # dO_k / dh = - dR_k / dh
    dRdh = np.zeros_like(h)
    for k in range(n):
        dRdh += h * (np.roll(h, k) + np.roll(h, -k))
    dObj_dh = -dRdh * (t - R > 0) * 1.0   # only violated shifts matter

    # dH/dph_j = i * H_j ;  dh/dph = ifft(i*H)
    H = amp * np.exp(1j * ph)
    dh_dph = np.fft.ifft(1j * H).real
    grad = np.sum(dObj_dh * dh_dph, axis=0)
    return obj, grad


def construct():
    start = time.time()
    deadline = start + 132.0          # safety margin under 150s kill
    n = N

    # Flat amplitude spectrum: |H_k| = sqrt(n * base) so that R_k = base for all k.
    amp = np.full(n, np.sqrt(n * BASE))

    # Start from the uniform (zero-phase) solution.
    ph = np.zeros(n)
    h = _phases_to_h(ph, amp)
    best_h = h.copy()
    best_ov = _max_overlap(h)

    # Warm restarts: random phases give diverse basin starts.
    rng = np.random.default_rng(1234)

    def run_phase_opt(ph0, t0, budget):
        nonlocal best_h, best_ov
        ph = ph0.copy()
        t = t0
        lr = 1.0
        for it in range(4000):
            if time.time() > budget:
                break
            obj, grad = _epigraph_obj_grad(ph, amp, t)
            h = _phases_to_h(ph, amp)
            ov = float(np.max(_compute_overlaps(h)))
            if ov < best_ov - 1e-12:
                best_ov = ov
                best_h = h.copy()
            # Adaptive step: scale so the phase update magnitude is ~ lr.
            gnorm = np.linalg.norm(grad)
            if gnorm > 1e-12:
                step = lr / gnorm
            else:
                break
            ph = np.angle(np.exp(1j * (ph - step * grad)))
            # Reduce t (the overlap cap) slowly to tighten the target.
            t = max(0.0, t - 0.02)
        return ph

    # Phase 1: single run from zero phase, moderate cap.
    t = best_ov
    run_phase_opt(ph, t, deadline - 30.0)

    # Phase 2: multiple random restarts with fresh phase vectors, decreasing caps.
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline - 15.0:
            break
        ph_r = rng.standard_normal(n)
        # initialize t near current best to encourage improvement
        t_r = max(0.0, best_ov - 0.05)
        run_phase_opt(ph_r, t_r, deadline - 15.0)

    # Final projection to guarantee exact constraints.
    h = _project_to_box(best_h, TARGET)
    return h
