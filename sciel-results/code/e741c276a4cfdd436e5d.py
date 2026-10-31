import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Fast metric: max of linear autoconvolution (normalized)
    # We work with h normalized to sum=1, then metric = max(convolve(h, h, 'full'))
    # Using FFT for speed: circular convolution of length 2N
    size = 2 * N

    def compute_metric(h):
        """h: array of length N, nonnegative. Returns normalized max autoconvolution."""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        h_norm = h / total
        H = np.fft.rfft(h_norm, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        # acorr has length size=2N. The linear convolution of two length-N arrays
        # fits in 2N-1 points. The circular version wraps, but for our purposes
        # the peak is what matters. Actually irfft with n=size gives circular conv.
        # For linear conv we'd need n=2N-1, but the difference is negligible for
        # well-spread distributions. Let's use n=2N-1 for correctness.
        H = np.fft.rfft(h_norm, n=2 * N - 1)
        acorr = np.fft.irfft(H * np.conj(H), n=2 * N - 1)
        return np.max(acorr)

    # Strategy: IPF-style iterative proportional fitting to minimize L1 of autoconvolution
    # combined with parametric bump representation and local refinement.

    # Build a multi-bump parametric representation
    def build_bumps(params, K):
        """
        params: array of length 3*K: [center_0, width_0, amp_0, center_1, width_1, amp_1, ...]
        Returns nonnegative array of length N, normalized to sum 1.
        """
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N  # bin centers
        f = np.zeros(N)
        for k in range(K):
            c = params[3 * k]
            w = max(params[3 * k + 1], 1e-4)
            a = max(params[3 * k + 2], 0.0)
            # Gaussian bump
            f += a * np.exp(-0.5 * ((x - c) / w) ** 2)
        f = np.maximum(f, 0.0)
        total = np.sum(f)
        if total < 1e-15:
            return np.ones(N) / N
        return f / total

    def objective_bumps(params, K):
        f = build_bumps(params, K)
        return compute_metric(f)

    # IPF-style refinement: iteratively adjust h to reduce peak of autoconvolution
    # Key idea: if bin i has high autoconvolution contribution, reduce h[i]
    # and redistribute to bins with low contribution.
    def ipf_refine(h_init, max_iters=2000, tol=1e-10):
        """
        Iterative proportional fitting: minimize peak of autoconvolution.
        At each step, find the peak location in autoconvolution, then
        reduce the bins contributing most to that peak.
        """
        h = h_init.copy().astype(np.float64)
        total = np.sum(h)
        if total == 0:
            return h
        h = h / total

        best_h = h.copy()
        best_m = compute_metric(h)

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            # Compute autoconvolution
            H = np.fft.rfft(h, n=2 * N - 1)
            acorr = np.fft.irfft(H * np.conj(H), n=2 * N - 1)
            peak_val = np.max(acorr)

            if abs(peak_val - best_m) < tol:
                break

            # Find which lags have high autoconvolution
            # For lag k, acorr[k] = sum_i h[i]*h[i+k] (with wrap for circular)
            # For linear: acorr[k] = sum_i h[i]*h[i+k] for valid i
            # Strategy: reduce h where it contributes to high acorr values.
            # Use subgradient: for the peak lag, reduce h[j] where h[j]*h[j+lag] is large.

            # Find top few lags
            top_lags = np.argsort(acorr)[-min(5, len(acorr)):]
            top_lags = top_lags[top_lags > 0]  # skip lag 0 (always max for nonneg)

            # Build subgradient: for each h[j], how much does it contribute to peak lags?
            subgrad = np.zeros(N)
            for lag in top_lags:
                lag = int(lag)
                if lag < N:
                    # acorr[lag] = sum_{j=0}^{N-1-lag} h[j]*h[j+lag]
                    # d(acorr[lag])/d(h[j]) = h[j+lag] for j < N-lag, h[j-lag] for j >= lag
                    contrib = np.zeros(N)
                    contrib[:N - lag] += h[lag:]
                    contrib[lag:] += h[:N - lag]
                    subgrad += contrib * (acorr[lag] / (peak_val + 1e-15))

            # Normalize subgradient
            subgrad = subgrad / (np.max(subgrad) + 1e-15)

            # Update: reduce h where subgrad is high
            lr = 0.01
            h_new = h - lr * subgrad
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s < 1e-15:
                break
            h_new = h_new / s

            m_new = compute_metric(h_new)
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()

            h = h_new

            # Adaptive learning rate
            if m_new > best_m * (1 + 1e-6):
                lr *= 0.9

        return best_h

    # Local refinement via coordinate descent
    def local_refine(h_init, max_time=30.0):
        """Direct local search: try moving mass between pairs of bins."""
        h = h_init.copy().astype(np.float64)
        total = np.sum(h)
        if total == 0:
            return h
        h = h / total

        best_h = h.copy()
        best_m = compute_metric(h)
        local_deadline = time.time() + max_time

        np.random.seed(12345)

        while time.time() < local_deadline and time.time() < deadline - 5.0:
            # Pick two random indices
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue

            # Try moving a small fraction of mass
            for frac in [0.01, 0.05, 0.1, 0.2]:
                delta = h[j] * frac
                new_h = h.copy()
                new_h[j] -= delta
                new_h[i] += delta

                if new_h[j] < 0:
                    continue

                m_new = compute_metric(new_h)
                if m_new < best_m:
                    best_m = m_new
                    best_h = new_h.copy()
                    h = new_h.copy()
                    break
                elif m_new < compute_metric(h):
                    # Accept if it improves over current (not just best)
                    h = new_h.copy()

        return best_h

    # Main search: try different numbers of bumps, optimize, then refine
    best_h = np.ones(N) / N
    best_m = compute_metric(best_h)

    K_values = [2, 3, 4, 5, 6, 8]

    for K in K_values:
        if time.time() > deadline - 15:
            break

        # Multiple random initializations for K bumps
        n_inits = max(2, 6 - K // 2)
        for init_idx in range(n_inits):
            if time.time() > deadline - 10:
                break

            if init_idx == 0:
                # Symmetric initialization
                centers = np.linspace(-0.18, 0.18, K)
                widths = np.full(K, 0.04)
                amps = np.full(K, 1.0)
            else:
                centers = np.random.uniform(-0.22, 0.22, K)
                widths = np.random.uniform(0.01, 0.12, K)
                amps = np.random.uniform(0.5, 2.0, K)

            params0 = np.concatenate([centers, widths, amps])

            # Optimize using Nelder-Mead
            try:
                result = minimize(
                    lambda p: objective_bumps(p, K),
                    params0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True,
                    }
                )

                f_cand = build_bumps(result.x, K)
                m_cand = compute_metric(f_cand)

                if m_cand < best_m:
                    best_m = m_cand
                    best_h = f_cand.copy()
            except Exception:
                pass

    # IPF refinement on the best found
    if time.time() < deadline - 10:
        h_ipf = ipf_refine(best_h, max_iters=3000)
        m_ipf = compute_metric(h_ipf)
        if m_ipf < best_m:
            best_m = m_ipf
            best_h = h_ipf

    # Local refinement
    if time.time() < deadline - 8:
        remaining = min(25.0, deadline - 8 - time.time())
        if remaining > 2:
            h_local = local_refine(best_h, max_time=remaining)
            m_local = compute_metric(h_local)
            if m_local < best_m:
                best_m = m_local
                best_h = h_local

    # Final IPF pass if time allows
    if time.time() < deadline - 5:
        h_final = ipf_refine(best_h, max_iters=1000)
        m_final = compute_metric(h_final)
        if m_final < best_m:
            best_m = m_final
            best_h = h_final

    if best_h is None or np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N

    return list(best_h)
