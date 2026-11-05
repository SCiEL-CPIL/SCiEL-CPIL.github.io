N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct


def _autoconv(h):
    """Full autoconvolution via FFT (c[k] for lag k = -(N-1)..(N-1))."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 130.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Structural change: Parameterize h as A + B * s_i
    # s_i is a square wave.
    # We iterate over possible periods P of the square wave.
    # For each P, we optimize A and B.
    
    # Possible periods: divisors of N or small integers.
    # Since N=400, we can try P from 2 to 400.
    # To be efficient, we can sample periods.
    periods = list(range(2, 401))
    
    # We can also consider the phase of the square wave.
    # A square wave of period P can be shifted.
    # However, due to the symmetry of the objective (if we flip signs of h, S flips sign, S^2 same, conv flips sign? No, conv(h,h) is same. max(conv) same. So h and -h give same value).
    # But shifting the square wave changes the alignment with the "constant" part?
    # Actually, the constant part A is uniform. The square wave part B*s_i has zero mean (if balanced).
    # So the sum S = N*A.
    # The autoconvolution will depend on the correlation of the square wave with itself.
    # For a square wave, the autocorrelation is a triangle wave.
    # The peak is at lag 0.
    # So max(conv) will likely be at lag 0 or near it.
    
    # Let's define the square wave s_i.
    # s_i = 1 if (i + phase) % P < P/2 else -1.
    # We can optimize phase as well, but maybe 0 is good enough or we can try a few.
    
    for P in periods:
        if time_left() < 5.0:
            break
            
        # Generate square wave s of length N with period P.
        # We try a few phases: 0, P/4, P/2 (same as 0 flipped).
        for phase_frac in [0.0, 0.25]:
            phase = int(phase_frac * P)
            s = np.zeros(N)
            for i in range(N):
                idx = (i + phase) % P
                if idx < P // 2:
                    s[i] = 1.0
                else:
                    s[i] = -1.0
            
            # Ensure s has zero mean?
            # If P is even, mean is 0. If P is odd, mean is not 0.
            # The model is h = A + B*s.
            # If mean(s) != 0, then A and B are not uniquely identifiable for the DC component.
            # But we optimize A and B directly.
            
            def obj(params):
                A, B = params
                h = A + B * s
                return _hard(h)
            
            # Initial guess for A, B
            # We want S = N*A to be large.
            # Let's try A=1, B=0.1, 0.5, 1.0
            inits = [[1.0, 0.1], [1.0, 0.5], [1.0, 1.0], [0.5, 0.5], [2.0, 0.5]]
            
            for init in inits:
                if time_left() < 2.0:
                    break
                try:
                    res = minimize(obj, init, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                    h_new = res.x[0] + res.x[1] * s
                    v_new = _hard(h_new)
                    if v_new < best_val:
                        best_val = v_new
                        best_h = h_new.copy()
                except Exception:
                    pass

    # Fallback to original DCT approach if time remains?
    # The directive says "Make a STRUCTURAL change".
    # The new direction is the square wave parameterization.
    # If it fails, we might want to keep some of the old logic as a fallback,
    # but the directive implies replacing the strategy.
    # However, to ensure we don't regress to seed (2.0) if the new strategy is bad,
    # we can keep the best found.
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
