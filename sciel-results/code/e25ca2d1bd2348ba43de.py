"""Spectral-domain windowed-sinc construction + sign-flip refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _autoconv_max_abs(h):
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _gen_windowed_sinc(M, w):
    """FIR windowed-sinc of length M, window w (length M). Real part."""
    n = np.arange(M)
    mid = (M - 1) / 2.0
    x = n - mid
    # sinc: sin(pi x)/(pi x)
    sinc = np.ones_like(x)
    nz = x != 0
    sinc[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    h = w * sinc
    return h

def _spectral_sidelobe_objective(w, M, f_lo=0.15, f_hi=0.85):
    """Max of |H|^2 over the stopband (sidelobe region) in normalized freq."""
    h = _gen_windowed_sinc(M, w)
    H = np.fft.rfft(h, 2 * M)
    psd = np.abs(H) ** 2
    # normalized freq index for rfft of length 2M
    nfreq = len(psd)
    # main lobe roughly around 0; sidelobes away. We penalize max in [f_lo, f_hi]
    idx_lo = int(f_lo * nfreq)
    idx_hi = int(f_hi * nfreq)
    if idx_hi <= idx_lo:
        idx_hi = idx_lo + 1
    return np.max(psd[idx_lo:idx_hi])

def _optimize_window(M, f_lo=0.15, f_hi=0.85, iters=200, seed=0):
    """Optimize window coefficients to minimize max sidelobe PSD, subject to sum(w)=M."""
    rng = np.random.RandomState(seed)
    w0 = np.ones(M) / M  # normalized-ish
    # We optimize w directly; constrain sum to 1 via penalty-free by fixing one.
    # Use scipy minimize with SLSQP, variable = w, constraint sum(w)=1, w>=0.
    def obj(w):
        return _spectral_sidelobe_objective(w, M, f_lo, f_hi)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    res = opt.minimize(obj, w0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': iters, 'ftol': 1e-8})
    if res.success or res.fun < obj(w0):
        return res.x
    return w0

def _gen_windowed_sinc_sequence(w, target_len):
    """Pad/truncate the windowed-sinc FIR to target_len, then sign."""
    h = _gen_windowed_sinc(len(w), w)
    if len(h) < target_len:
        h = np.pad(h, (0, target_len - len(h)))
    else:
        h = h[:target_len]
    # sign
    s = np.sign(h)
    s[s == 0] = 1.0
    return s

def _local_single_flip(h, max_passes=8, deadline=None):
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        best_i = -1
        best_new = best_val
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            nv = _score(h)
            h[i] = -h[i]
            if nv < best_new:
                best_new = nv
                best_i = i
        if best_i >= 0:
            h[best_i] = -h[best_i]
            best_val = best_new
        else:
            break
    return h, best_val

def _local_pair_flip(h, max_passes=4, deadline=None):
    """Random pair flips to escape local minima."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.RandomState(123)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for trial in range(200):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.choice(N, 2, replace=False)
            h[i] = -h[i]
            h[j] = -h[j]
            nv = _score(h)
            if nv < best_val:
                best_val = nv
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
        if not improved:
            break
    return h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0

    best_h = np.ones(N)
    best_val = _score(best_h)

    # Phase 1: spectral windowed-sinc constructions with several windows
    # Use a few fixed windows (Blackman, Hann, Hamming, Blackman-Harris) + one optimized
    windows = []
    M = N
    # Blackman
    n = np.arange(M)
    win_blackman = 0.42 - 0.5*np.cos(2*np.pi*n/(M-1)) + 0.08*np.cos(4*np.pi*n/(M-1))
    windows.append(win_blackman)
    # Hann
    win_hann = 0.5 - 0.5*np.cos(2*np.pi*n/(M-1))
    windows.append(win_hann)
    # Hamming
    win_hamming = 0.54 - 0.46*np.cos(2*np.pi*n/(M-1))
    windows.append(win_hamming)
    # Blackman-Harris (4-term)
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    win_bh = (a[0] - a[1]*np.cos(2*np.pi*n/(M-1)) + a[2]*np.cos(4*np.pi*n/(M-1))
              - a[3]*np.cos(6*np.pi*n/(M-1)))
    windows.append(win_bh)
    # Optimized window (minimax sidelobe)
    if time.time() < deadline - 20:
        try:
            w_opt = _optimize_window(M, f_lo=0.15, f_hi=0.85, iters=150, seed=0)
            windows.append(w_opt)
        except Exception:
            pass

    for w in windows:
        if time.time() > deadline - 10:
            break
        h = _gen_windowed_sinc_sequence(w, N)
        # ensure nonzero sum; if negative flip
        if np.sum(h) < 0:
            h = -h
        if abs(np.sum(h)) < 1e-6:
            h[0] = 1.0
        h, v = _local_single_flip(h, max_passes=6, deadline=deadline)
        if v < best_val:
            best_val = v
            best_h = h

    # Phase 2: pair-flip refinement to escape local minima
    if time.time() < deadline - 8:
        h, v = _local_pair_flip(best_h, max_passes=3, deadline=deadline)
        if v < best_val:
            best_val = v
            best_h = h

    # Phase 3: final single-flip polish
    if time.time() < deadline - 3:
        h, v = _local_single_flip(best_h, max_passes=6, deadline=deadline)
        if v < best_val:
            best_val = v
            best_h = h

    # Phase 4: one more pair-flip round if time
    if time.time() < deadline - 2:
        h, v = _local_pair_flip(best_h, max_passes=2, deadline=deadline)
        if v < best_val:
            best_val = v
            best_h = h

    return best_h
# EVOLVE-BLOCK-END
