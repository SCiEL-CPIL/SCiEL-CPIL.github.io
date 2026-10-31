import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        return h

    # Initial candidates: uniform and a few structured patterns
    candidates = []
    candidates.append(np.full(N, 0.5))
    # Add some structured starting points
    for freq in [1, 2, 3, 5, 7]:
        for amp in [0.2, 0.35, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = normalize(h)
            candidates.append(h)
    # Binary-like patterns
    for period in [3, 4, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            if (i // period) % 2 == 0:
                h[i] = 1.0
        h = normalize(h)
        candidates.append(h)

    best_score = float('inf')
    best_h = None
    for h in candidates:
        s = max_overlap(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Simulated Annealing
    rng = np.random.default_rng(42)
    current_h = best_h.copy()
    current_score = best_score
    best_h = current_h.copy()
    best_score = current_score

    T_start = 0.15
    T_end = 0.001
    total_steps = 80000
    # Adaptive: run as many steps as time allows
    t0 = time.time()
    step_times = []

    for step in range(total_steps):
        if time.time() > deadline - 5.0:
            break

        # Temperature schedule (geometric cooling)
        frac = step / total_steps
        T = T_start * (T_end / T_start) ** frac

        # Perturbation: modify a random subset of entries
        n_mod = rng.integers(3, 15)
        idx = rng.choice(N, size=n_mod, replace=False)
        delta = rng.normal(0, 0.08, size=n_mod)
        new_h = current_h.copy()
        new_h[idx] += delta
        new_h = normalize(new_h)

        new_score = max_overlap(new_h)

        # Accept if better, or with probability exp(-(new-old)/T)
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            delta_e = new_score - current_score
            if delta_e > 0:
                prob = np.exp(-delta_e / max(T, 1e-10))
                if rng.random() < prob:
                    current_h = new_h
                    current_score = new_score

        # Periodically check timing to estimate remaining budget
        if step % 5000 == 0 and step > 0:
            elapsed = time.time() - t0
            if elapsed > 0:
                # Estimate if we should reduce remaining steps
                pass

    # Final local gradient polish from the best SA result
    if time.time() < deadline - 2.0:
        h = best_h.copy()
        lr = 0.002
        for step in range(3000):
            if time.time() > deadline - 1.0:
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
