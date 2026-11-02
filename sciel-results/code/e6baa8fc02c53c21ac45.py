"""Pulse-superposition design for the first autocorrelation inequality.

The step function is modelled as a small number of narrow, high-amplitude
rectangular pulses centred at distinct locations in [-1/4, 1/4].  Only the
pulse positions, widths and amplitudes are optimised (roughly 10-20
parameters) with a derivative-free Nelder-Mead search, which is far cheaper
than directly optimising 600 continuous heights.  The score is the normalised
maximum of the autoconvolution, so a lower value is better.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
# Interval [-1/4, 1/4] split into N equal sub-intervals.
DX = (0.5) / N          # width of each sub-interval
X = np.linspace(-0.25 + DX / 2.0, 0.25 - DX / 2.0, N)  # interval centres

# Deadline guard: kill the search well before the 150 s wall-clock limit.
_DEADLINE = time.time() + 130.0
_TIME_MARGIN = 10.0


def _time_left():
    return _DEADLINE - _TIME_MARGIN - time.time()


def _heights_from_params(params, n_pulses):
    """Build a nonnegative step-height vector from pulse parameters.

    params layout (per pulse): [center, width, amplitude]
    width is expressed in number of sub-intervals (>= 1).
    """
    heights = np.zeros(N, dtype=float)
    for i in range(n_pulses):
        c = params[3 * i]
        w = max(1.0, params[3 * i + 1])
        a = max(0.0, params[3 * i + 2])
        half = w / 2.0
        lo = c - half
        hi = c + half
        idx = np.where((X >= lo) & (X < hi))[0]
        if idx.size:
            heights[idx] += a
    return heights


def _score_from_heights(heights):
    """Normalised maximum of the autoconvolution of the step function."""
    # Autoconvolution on the discrete grid (linear, not circular).
    conv = np.convolve(heights, heights, mode='full')
    # Normalise so that the uniform function (all ones) has a known value.
    # For the uniform function the autoconvolution is triangular with peak N,
    # and the "normalised maximum" is defined so that uniform -> 1.0.
    # The natural normalisation: divide by the peak of the uniform autoconv.
    # uniform autoconv peak = N (at the centre).  So normalised max = max(conv)/N.
    peak = conv.max()
    return float(peak / N)


def _objective(params, n_pulses):
    if _time_left() <= 0:
        return float('inf')
    heights = _heights_from_params(params, n_pulses)
    if heights.max() <= 0:
        return float('inf')
    return _score_from_heights(heights)


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 130.0

    best_heights = np.ones(N, dtype=float)
    best_score = _score_from_heights(best_heights)  # = 1.0

    # Try several pulse counts; each uses a few Nelder-Mead restarts.
    pulse_counts = [3, 4, 5]
    rng = np.random.default_rng(0)

    for n_p in pulse_counts:
        if _time_left() <= 5:
            break
        n_params = 3 * n_p
        # Number of random restarts we can afford; keep it modest.
        max_restarts = 4
        for r in range(max_restarts):
            if _time_left() <= 5:
                break
            # Initial guess: spread pulses across the interval.
            centers = np.linspace(-0.18, 0.18, n_p)
            widths = rng.uniform(4.0, 12.0, n_p)
            amps = rng.uniform(0.5, 2.0, n_p)
            x0 = np.concatenate([centers, widths, amps])

            bounds_lo = np.concatenate([
                np.full(n_p, -0.25), np.full(n_p, 1.0), np.zeros(n_p)
            ])
            bounds_hi = np.concatenate([
                np.full(n_p, 0.25), np.full(n_p, 20.0), np.full(n_p, 5.0)
            ])

            # Nelder-Mead does not support bounds directly; we clip inside
            # the objective by re-using _heights_from_params (which already
            # clamps width and amplitude).  For centers we rely on the
            # initialisation and the geometry of the problem; a center far
            # outside simply produces a zero pulse, which is penalised.
            try:
                res = minimize(
                    _objective, x0, args=(n_p,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 400,
                        'maxfev': 2000,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True,
                    },
                )
            except Exception:
                continue

            if res.fun < best_score and np.isfinite(res.fun):
                best_score = float(res.fun)
                best_heights = _heights_from_params(res.x, n_p)

    # Return as a plain sequence of floats (nonnegative step heights).
    result = [float(h) for h in best_heights]
    # Ensure strict nonnegativity (clip tiny negatives from numerics).
    result = [max(0.0, h) for h in result]
    return result
