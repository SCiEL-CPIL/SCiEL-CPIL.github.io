import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def max_overlap(h):
        corr = compute_corr(h)
        min_corr = np.min(corr)
        return N / 2.0 - min_corr

    def normalize(h):
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 0:
            h = h / s * (N / 2.0)
        h = np.clip(h, 0, 1)
        # Fix any tiny numerical issues
        diff = N / 2.0 - h.sum()
        if abs(diff) > 1e-12:
            # Distribute correction evenly where possible
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        return h

    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)  # Should be N/4 = 25.0

    # Phase 1: Grid search over multi-frequency sinusoidal patterns
    # Try combinations of 2-3 frequencies with various amplitudes
    if time.time() < deadline - 60:
        # Single frequency search (coarse)
        for freq in range(1, N // 2 + 1):
            if time.time() > deadline - 60:
                break
            for amp in np.linspace(0.05, 0.49, 8):
                h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
                h = normalize(h)
                score = max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()

    # Phase 2: Multi-frequency search
    if time.time() < deadline - 40:
        # Try 2-frequency combinations
        freq_pairs = [(1, 2), (1, 3), (1, 5), (2, 3), (2, 5), (3, 4), (1, 4), (2, 4), (3, 5), (1, 7), (2, 7), (3, 7)]
        for f1, f2 in freq_pairs:
            if time.time() > deadline - 40:
                break
            for a1 in np.linspace(0.05, 0.35, 5):
                if time.time() > deadline - 40:
                    break
                for a2 in np.linspace(0.05, 0.35, 5):
                    if time.time() > deadline - 40:
                        break
                    phase2 = 0.0
                    h = 0.5 + a1 * np.sin(2 * np.pi * f1 * np.arange(N) / N) + \
                          a2 * np.sin(2 * np.pi * f2 * np.arange(N) / N + phase2)
                    h = normalize(h)
                    score = max_overlap(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()

    # Phase 3: Local gradient refinement
    if time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.003
        for step in range(5000):
            if time.time() > deadline - 5:
                break

            corr = compute_corr(h)
            k_max = np.argmin(corr)  # shift that gives max overlap

            # Gradient of overlap w.r.t. h[j]:
            # d/dh[j] (N/2 - corr[k]) = -(h[(j+k)%N] + h[(j-k)%N])
            idx = np.arange(N)
            grad = -(h[(idx + k_max) % N] + h[(idx - k_max) % N])

            h = h - lr * grad
            h = normalize(h)

            # Adaptive learning rate: reduce if oscillating
            if step % 100 == 0 and step > 0:
                lr = max(0.0005, lr * 0.95)

        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h

    # Phase 4: Final fine-tuning with smaller steps
    if time.time() < deadline - 5:
        h = best_h.copy()
        lr = 0.001
        for step in range(2000):
            if time.time() > deadline - 2:
                break
            corr = compute_corr(h)
            k_max = np.argmin(corr)
            idx = np.arange(N)
            grad = -(h[(idx + k_max) % N] + h[(idx - k_max) % N])
            h = h - lr * grad
            h = normalize(h)

        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h

    # Ensure exact constraints
    best_h = normalize(best_h)

    # Final safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
