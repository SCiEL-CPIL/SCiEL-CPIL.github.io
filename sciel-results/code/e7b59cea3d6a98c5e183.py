import numpy as np
import time

N = 600


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    i = np.arange(N)

    def compute_metric(h):
        """Normalized max of autoconvolution: max(|acorr[1:]|) / acorr[0]."""
        h = np.asarray(h, dtype=np.float64)
        if h.size == 0 or np.all(h == 0):
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = acorr[0]
        if peak <= 0:
            return float('inf')
        return float(np.max(np.abs(acorr[1:])) / peak)

    def spectral_init(alpha, beta):
        """Design h from a shaped spectrum: h = irfft(|H|^2 * mask), clipped >=0."""
        size = 2 * N
        freqs = np.fft.rfftfreq(size, d=1.0 / size)
        # window in freq domain: lowpass-ish, raised-cosine taper to suppress sidelobes
        w = np.cos(np.pi * np.abs(freqs) / (1.0 / (2.0 * N)))
        w = np.maximum(w, 0.0)
        # additional shaping
        w = w ** alpha
        H = w * np.exp(1j * beta * np.sin(2 * np.pi * freqs))
        h = np.fft.irfft(H * np.conj(H), n=size)[:N]
        h = np.maximum(h, 0.0)
        return h

    def subgradient_opt(h, max_iters, base_lr):
        """Subgradient descent on worst-sidelobe ratio with adaptive step."""
        h = h.copy()
        best_h = h.copy()
        best_m = compute_metric(h)
        lr = base_lr

        for it in range(max_iters):
            if time.time() > deadline:
                break
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak = acorr[0]
            if peak <= 0:
                break
            # worst sidelobe (signed)
            k = int(np.argmax(np.abs(acorr[1:]))) + 1
            # subgradient of acorr[k] w.r.t. h
            g = np.zeros(N)
            g[:N - k] += h[k:]
            g[k:] += h[:N - k]
            # subgradient of peak = 2h
            gp = 2.0 * h
            # d(acorr[k]/peak)
            grad = (g * peak - acorr[k] * gp) / (peak * peak)
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0.0)
            m_new = compute_metric(h_new)
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            # adaptive lr
            if m_new < best_m * 0.999:
                lr = min(lr * 1.2, 0.5)
            else:
                lr = max(lr * 0.9, 0.0005)
            h = h_new

        return best_h, best_m

    def local_refine(h, max_iters):
        """Coordinate-descent local refinement on the metric."""
        h = h.copy()
        best_m = compute_metric(h)
        step = 0.02
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # try a few random coordinates
            idx = np.random.randint(0, N, size=8)
            for j in idx:
                for delta in (step, -step):
                    h2 = h.copy()
                    h2[j] = max(0.0, h[j] + delta)
                    m2 = compute_metric(h2)
                    if m2 < best_m:
                        best_m = m2
                        h = h2
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
        return h, best_m

    best_h = None
    best_m = float('inf')

    # Phase 1: spectral initializations (cheap, structurally good)
    inits = []
    for alpha in (1.0, 1.5, 2.0, 3.0):
        for beta in (0.0, 0.5, 1.0):
            inits.append(spectral_init(alpha, beta))
    # also classic shapes
    inits.append(0.5 * (1.0 + np.cos(np.pi * (i - N / 2.0) / (N / 2.0))))
    inits.append(np.exp(-2.0 * ((i - N / 2.0) / (N / 4.0)) ** 2))

    for h0 in inits:
        if time.time() > deadline:
            break
        h0 = np.maximum(h0, 0.0)
        if np.all(h0 == 0):
            continue
        h_opt, m_opt = subgradient_opt(h0, max_iters=1500, base_lr=0.05)
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt.copy()

    # Phase 2: local refinement on the best found
    if best_h is not None and time.time() < deadline:
        best_h, best_m = local_refine(best_h, max_iters=400)

    if best_h is None:
        best_h = np.ones(N)

    return list(best_h)
