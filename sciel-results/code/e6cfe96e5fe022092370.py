import numpy as np
import time
from itertools import product

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant score: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l1 * conv_peak / l2_sq

def conv_peak(h):
    """Compute just the peak of the autoconvolution."""
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    return np.max(conv[:2*N-1])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)
    best_ratio = compute_ratio(best_h)

    # === Phase 1: Structured parametric family search ===
    # The key insight from inspiration 1 (score 1.1576) is that structured
    # parametric families find the optimal shape much more efficiently.
    # We exploit scale invariance: fix L1=1, then the ratio = L2_sq / conv_peak.
    # We want to maximize L2_sq / conv_peak with L1=1.
    #
    # Key families to search:
    # 1. Symmetric patterns (optimal is likely symmetric)
    # 2. Sparse patterns with specific spacing
    # 3. Geometric/arithmetic progressions
    # 4. Block patterns

    candidates = []

    # Uniform (baseline)
    candidates.append(np.ones(N) / N)

    # Two-point patterns: h[i]=a, h[j]=1-a
    for i in range(N):
        for j in range(i+1, N):
            for a in [0.5]:  # Symmetric is often optimal
                h = np.zeros(N)
                h[i] = a
                h[j] = 1 - a
                candidates.append(h)

    # Three-point patterns
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Only check a subset to save time
                if (j - i) % 5 == 0 and (k - j) % 5 == 0:
                    for a, b in [(1/3, 1/3), (0.5, 0.25), (0.7, 0.15)]:
                        h = np.zeros(N)
                        h[i] = a
                        h[j] = b
                        h[k] = 1 - a - b
                        if h[k] >= 0:
                            candidates.append(h)

    # Arithmetic progression supports with uniform weights
    for start in range(0, N):
        for step in range(1, 8):
            indices = list(range(start, N, step))
            if len(indices) < 2:
                continue
            k = len(indices)
            h = np.zeros(N)
            h[indices] = 1.0 / k
            candidates.append(h)
            # Also try geometric weights
            for r in [0.5, 0.7, 1.0, 1.4]:
                weights = np.array([r ** (k - 1 - i) for i in range(k)])
                h = np.zeros(N)
                h[indices] = weights
                h /= np.sum(h)
                candidates.append(h)

    # Symmetric pairs: h[i] = h[N-1-i]
    # Parameterize by choosing values for first half
    for trial in range(500):
        if time.time() > deadline:
            break
        # Random symmetric pattern
        half = N // 2
        h_half = np.random.dirichlet(np.ones(half))
        h = np.zeros(N)
        h[:half] = h_half
        h[half:] = h_half[::-1]
        if N % 2 == 1:
            h[half] /= 2
            h[half] += h_half[-1] / 2
            h[-1] = h_half[-1] / 2
        else:
            h[half:] = h_half[::-1]
        h /= np.sum(h)
        candidates.append(h)

    # Sparse random patterns
    for trial in range(300):
        if time.time() > deadline:
            break
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.dirichlet(np.ones(k))
        h /= np.sum(h)
        candidates.append(h)

    # Gaussian-like profiles
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]:
        for center in [N//4, N//2, 3*N//4]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h += 1e-12
            h /= np.sum(h)
            candidates.append(h)

    # Power law / Cauchy profiles
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for center in [N//4, N//2, 3*N//4]:
            x = np.arange(N)
            h = (1.0 + ((x - center) / 3.0) ** 2) ** (-p)
            h += 1e-12
            h /= np.sum(h)
            candidates.append(h)

    # Block patterns
    for k in range(1, 25):
        for start in range(0, max(1, N - k + 1), max(1, (N - k) // 8)):
            h = np.zeros(N)
            h[start:start+k] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # Alternating / periodic patterns
    for period in range(2, 12):
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = shape_score(h)
        if s < best_shape:
            best_shape = s
            best_h = h.copy()

    # === Phase 2: Local optimization via coordinate descent ===
    if time.time() < deadline - 30:
        h = best_h.copy()
        current_shape = shape_score(h)
        
        # Coordinate descent with golden section search on each coordinate
        for iteration in range(30):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 30:
                    break
                # For coordinate i, try moving mass to each other coordinate
                # Use a few trial fractions
                for j in range(N):
                    if i == j:
                        continue
                    if h[i] < 1e-15:
                        continue
                    # Try several fractions
                    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        delta = h[i] * frac
                        h_new = h.copy()
                        h_new[i] -= delta
                        h_new[j] += delta
                        if h_new[i] < -1e-12:
                            continue
                        new_shape = shape_score(h_new)
                        if new_shape < current_shape - 1e-12:
                            h = h_new
                            current_shape = new_shape
                            improved = True
                            if current_shape < best_shape:
                                best_shape = current_shape
                                best_h = h.copy()
                            break
            if not improved:
                break

    # === Phase 3: Random restarts with local search ===
    if time.time() < deadline - 20:
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break

            # Create diverse starting points
            start_type = r % 5
            if start_type == 0:
                # Perturb best
                h = best_h.copy()
                noise = np.random.dirichlet(np.ones(N) * 10)
                h = 0.7 * h + 0.3 * noise
                h = np.maximum(h, 0)
                h /= np.sum(h)
            elif start_type == 1:
                # Random sparse
                k = np.random.randint(2, 8)
                idx = np.random.choice(N, k, replace=False)
                h = np.zeros(N)
                h[idx] = np.random.dirichlet(np.ones(k))
            elif start_type == 2:
                # Random symmetric
                half = N // 2
                h_half = np.random.dirichlet(np.ones(half))
                h = np.zeros(N)
                h[:half] = h_half
                h[half:] = h_half[::-1]
                if N % 2 == 1:
                    h[half] = (h_half[-1] + h_half[-2]) / 2
                h /= np.sum(h)
            elif start_type == 3:
                # Arithmetic progression
                start = np.random.randint(0, N)
                step = np.random.randint(1, 6)
                indices = list(range(start, N, step))
                if len(indices) < 2:
                    continue
                h = np.zeros(N)
                weights = np.random.dirichlet(np.ones(len(indices)))
                h[indices] = weights
            else:
                # Gaussian random
                sigma = np.random.uniform(1, 8)
                center = np.random.uniform(0, N)
                x = np.arange(N)
                h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                h += 1e-12
                h /= np.sum(h)

            current_shape = shape_score(h)
            if current_shape < best_shape:
                best_shape = current_shape
                best_h = h.copy()

            # Local search from this point
            for _ in range(200):
                if time.time() > deadline - 20:
                    break
                improved = False
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                if h[i] < 1e-15:
                    continue
                delta = h[i] * np.random.uniform(0.05, 0.5)
                h_new = h.copy()
                h_new[i] -= delta
                h_new[j] += delta
                if h_new[i] < -1e-12:
                    continue
                new_shape = shape_score(h_new)
                if new_shape < current_shape - 1e-12:
                    h = h_new
                    current_shape = new_shape
                    improved = True
                    if current_shape < best_shape:
                        best_shape = current_shape
                        best_h = h.copy()

    # === Phase 4: High-precision polish ===
    if time.time() < deadline - 5:
        h = best_h.copy()
        current_shape = shape_score(h)

        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            # Very small perturbations
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            if h[i] < 1e-15:
                continue
            delta = h[i] * np.random.uniform(0.001, 0.01)
            h_new = h.copy()
            h_new[i] -= delta
            h_new[j] += delta
            if h_new[i] < -1e-12:
                continue
            new_shape = shape_score(h_new)
            if new_shape < current_shape - 1e-14:
                h = h_new
                current_shape = new_shape
                if current_shape < best_shape:
                    best_shape = current_shape
                    best_h = h.copy()

    # === Phase 5: Try specific structured patterns that might be optimal ===
    # The optimal pattern for autocorrelation inequalities often has
    # very specific structure. Let's try:
    # - Patterns where h*h has its peak at lag 0 (self-correlation)
    # - Patterns with specific symmetries

    # Try: h concentrated at positions that are "far apart" to reduce
    # cross-correlation peaks
    for trial in range(200):
        if time.time() > deadline - 2:
            break
        # Choose k positions that are evenly spaced
        k = np.random.randint(2, 10)
        base_step = N // k
        positions = [(i * base_step + np.random.randint(-base_step//3, base_step//3 + 1)) % N for i in range(k)]
        # Remove duplicates
        positions = list(set(positions))
        if len(positions) < 2:
            continue
        k = len(positions)
        h = np.zeros(N)
        weights = np.random.dirichlet(np.ones(k))
        h[positions] = weights
        h /= np.sum(h)
        s = shape_score(h)
        if s < best_shape:
            best_shape = s
            best_h = h.copy()

    # Normalize to L1=1 (the evaluator likely normalizes, but let's be safe)
    best_h = np.maximum(best_h, 0)
    best_h /= np.sum(best_h)

    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(f"Ratio: {compute_ratio(h)}")
    print(f"Shape score: {shape_score(h)}")
    print(f"L1: {np.sum(h)}, L2_sq: {np.sum(np.array(h)**2)}")
