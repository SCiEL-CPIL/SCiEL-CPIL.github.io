"""Autocorrelation inequality task: minimize 2*N*max(autocorr)/sum(h)^2 via sparse IDFT spectra."""
import numpy as np
import time

N = 400
DEADLINE = 150.0


def _value(h):
    s = h.sum()
    if s == 0:
        return np.inf
    ac = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(ac) / (s * s))


def _autocorr_fft(h):
    # full autocorrelation via FFT (matches np.convolve(h,h) for real h)
    n = 1
    while n < 2 * N - 1:
        n <<= 1
    H = np.fft.rfft(h, n)
    ac = np.fft.irfft(H * np.conj(H), n)[:2 * N - 1]
    return ac


def _value_fast(h):
    s = h.sum()
    if s == 0:
        return np.inf
    ac = _autocorr_fft(h)
    return abs(2.0 * N * np.max(ac) / (s * s))


def _from_spectrum(mags, phases, freqs):
    # build h as IDFT of sparse real spectrum
    spec = np.zeros(N, dtype=complex)
    for k, m, ph in zip(freqs, mags, phases):
        spec[k] = m * np.exp(1j * ph)
        spec[N - k] = np.conj(spec[k]) if k != 0 else spec[k]
    h = np.fft.irfft(spec)
    return h


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_v = _value(best_h)

    def time_left():
        return DEADLINE - 10.0 - (time.time() - t0)

    # ---- Phase 1: random sparse spectra search ----
    rng = np.random.default_rng(12345)
    freq_pool = list(range(1, N))  # nonzero frequencies
    n_iter = 0
    while time_left() > 20.0:
        n_iter += 1
        # pick a small number of frequency bins
        n_bins = int(rng.integers(2, 8))
        freqs = sorted(rng.choice(freq_pool, size=n_bins, replace=False).tolist())
        mags = rng.uniform(0.1, 5.0, size=n_bins)
        phases = rng.uniform(-np.pi, np.pi, size=n_bins)
        h = _from_spectrum(mags, phases, freqs)
        # enforce sum(h) != 0
        if abs(h.sum()) < 1e-6:
            continue
        v = _value_fast(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        # local polish via gradient-free on the last few
        if v < best_v + 0.05 and time_left() > 25.0:
            # simple coordinate descent on magnitudes
            cur = h.copy()
            cv = v
            for _ in range(300):
                if time_left() < 25.0:
                    break
                improved = False
                for j in range(n_bins):
                    for step in (0.1, -0.1, 0.02, -0.02):
                        mags2 = mags.copy()
                        mags2[j] = max(0.0, mags[j] + step)
                        h2 = _from_spectrum(mags2, phases, freqs)
                        if abs(h2.sum()) < 1e-6:
                            continue
                        v2 = _value_fast(h2)
                        if v2 < cv:
                            cv = v2
                            cur = h2
                            mags = mags2
                            improved = True
                            break
                if not improved:
                    break
            if cv < best_v:
                best_v = cv
                best_h = cur.copy()

    # ---- Phase 2: final polish with exact convolve value ----
    if time_left() > 5.0:
        # refine best_h with a small random perturbation search
        rng2 = np.random.default_rng(999)
        for _ in range(2000):
            if time_left() < 5.0:
                break
            h2 = best_h + rng2.normal(0, 0.01 * max(1.0, np.max(np.abs(best_h))), N)
            if abs(h2.sum()) < 1e-6:
                continue
            v2 = _value_fast(h2)
            if v2 < best_v:
                best_v = v2
                best_h = h2

    # verify with exact np.convolve at the end
    final_v = _value(best_h)
    if final_v < best_v:
        best_v = final_v

    return best_h
