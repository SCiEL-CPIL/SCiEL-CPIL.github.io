import numpy as np
import time

def _peak(c, N=4096, top_k=8, gs_iters=25):
    """
    Estimate the max |P(z)| on the unit circle.
    Dense FFT sampling + golden-section refinement of top peaks.
    """
    n = len(c)
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F = np.fft.ifft(cf) * N
    mags = np.abs(F)

    idx = np.argpartition(mags, -top_k)[-top_k:]
    best_peak = float(mags[idx].max())

    gr = (np.sqrt(5) - 1) / 2
    for k in idx:
        lo = max(0.0, (k - 3) / N)
        hi = min(1.0, (k + 4) / N)
        a, b = lo, hi
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = abs(c @ np.exp(2j * np.pi * c1))
        f2 = abs(c @ np.exp(2j * np.pi * c2))
        for _ in range(gs_iters):
            if f1 > f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = abs(c @ np.exp(2j * np.pi * c1))
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = abs(c @ np.exp(2j * np.pi * c2))
        v = max(f1, f2)
        if v > best_peak:
            best_peak = v
    return best_peak

def _local_search(c, deadline, N=8192, top_k=6):
    """
    Vectorized local search: for each single flip, compute the new FFT-based
    peak in one vectorized pass (no golden-section per flip).  Iterate until
    no improving flip is found or the deadline is hit.
    """
    c = c.astype(float).copy()
    n = len(c)
    cf = np.zeros(N, dtype=complex)
    cf[:n] = c
    F0 = np.fft.ifft(cf) * N
    mags0 = np.abs(F0)
    idx = np.argpartition(mags0, -top_k)[-top_k:]
    peak0 = float(mags0[idx].max())

    # Precompute the N roots of unity once
    roots = np.exp(2j * np.pi * np.arange(N) / N)
    # powers[k][j] = roots[j] ** k  for k in 0..n-1
    powers = np.vstack([roots ** k for k in range(n)])

    while time.time() < deadline:
        # For each flip i: new_F = F0 - 2*c[i]*powers[i]
        # new_peak_i = max over idx of |F0[idx] - 2*c[i]*powers[i][idx]|
        # Vectorize: diff shape (n, top_k)
        delta = 2.0 * c[:, None] * powers[:, idx][None, :]  # (n, top_k)
        new_mags = np.abs(F0[None, idx] - delta)  # (n, top_k)
        new_peaks = new_mags.max(axis=1)  # (n,)

        best_i = int(np.argmin(new_peaks))
        best_peak = float(new_peaks[best_i])

        if best_peak >= peak0:
            break  # no improving flip

        c[best_i] = -c[best_i]
        # Update F0: F0 -= 2*c_old*powers[best_i]  (c_old was c[best_i] before flip)
        c_old = -c[best_i]
        F0 -= 2.0 * c_old * powers[best_i]
        peak0 = best_peak

    return c, peak0

def construct():
    start = time.time()
    deadline = start + 140.0
    n = 24

    candidates = []

    # Known / structured starting points
    candidates.append(np.ones(n))
    candidates.append(np.array([1, -1] * 12, dtype=float))
    candidates.append(np.array([1, 1, -1, -1] * 6, dtype=float))
    candidates.append(np.array([1, -1, -1, 1] * 6, dtype=float))
    candidates.append(np.array([1, 1, 1, -1] * 6, dtype=float))
    candidates.append(np.array([1, 1, -1, 1] * 6, dtype=float))
    candidates.append(np.array([1, -1, 1, 1] * 6, dtype=float))
    candidates.append(np.array([1, 1, 1, 1, -1, -1, -1, -1] * 3, dtype=float))
    candidates.append(np.array([1, -1, -1, -1, 1, 1, 1, 1] * 3, dtype=float))

    # GRS (Golay-Riordan-Salem) construction
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:n], dtype=float))

    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=n))

    best_c = None
    best_score = float('inf')

    # Phase 1: quick screen all candidates with a fast peak (smaller N)
    for c in candidates:
        if time.time() > deadline - 30:
            break
        p = _peak(c, N=2048, top_k=4, gs_iters=15)
        if p < best_score:
            best_score = p
            best_c = c.copy()

    # Phase 2: local search on the most promising candidates
    # Re-rank with a slightly better peak estimate
    scored = []
    for c in candidates:
        if time.time() > deadline - 10:
            break
        p = _peak(c, N=4096, top_k=6, gs_iters=20)
        scored.append((p, c))
    scored.sort(key=lambda x: x[0])

    for p, c in scored:
        if time.time() > deadline - 5:
            break
        if p > best_score * 1.03:
            break
        c_opt, peak_opt = _local_search(c, deadline - 3)
        if peak_opt < best_score:
            best_score = peak_opt
            best_c = c_opt
        # Optional: refine the final answer with golden-section
        if time.time() < deadline - 5:
            refined = _peak(c_opt, N=8192, top_k=10, gs_iters=30)
            if refined < best_score:
                best_score = refined

    if best_c is None:
        best_c = candidates[0]

    return [int(x) for x in best_c]
