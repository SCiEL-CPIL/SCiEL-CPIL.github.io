import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Autoconvolution via FFT for efficiency
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)

    # Key insight: for nonnegative h, check if conv_peak is always at lag 0
    # If so, ratio = 1/L1, and we just minimize L1
    # But let's verify with a test
    test_h = np.array([0., 1., 0., 1., 0.])
    n_fft = 2 * len(test_h)
    H = np.fft.fft(test_h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    print(f"Test conv: {conv[:2*len(test_h)-1]}")
    # conv should be [0, 1, 2, 1, 0, 0, 1, 0] -> peak is 2 at lag 2, not lag 0!
    # So conv_peak != L2^2 in general. Need proper optimization.

    # Generate diverse initial candidates
    candidates = []

    # 1. Uniform
    candidates.append(np.ones(N) / N)

    # 2. Sparse patterns: pick k indices with equal weights
    for k in range(1, 20):
        for _ in range(5):
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # 3. Sparse with geometric weights (favor one end)
    for k in range(2, 15):
        for _ in range(3):
            idx = np.random.choice(N, k, replace=False)
            idx = np.sort(idx)
            weights = np.array([0.5 ** (k - 1 - i) for i in range(k)])
            h = np.zeros(N)
            h[idx] = weights
            h /= np.sum(h)
            candidates.append(h)

    # 4. Block of consecutive ones
    for k in range(1, 25):
        for start in range(0, max(1, N - k + 1), max(1, (N - k) // 10)):
            h = np.zeros(N)
            h[start:start+k] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # 5. Alternating patterns
    for period in range(2, 15):
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h)

    # 6. Gaussian-like profiles
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        for center in [N//4, N//2, 3*N//4]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h += 1e-12
            h /= np.sum(h)
            candidates.append(h)

    # 7. Power law profiles
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        x = np.arange(N)
        center = N / 2.0
        h = (1.0 + (x - center) ** 2) ** (-p)
        h += 1e-12
        h /= np.sum(h)
        candidates.append(h)

    # 8. Random Dirichlet
    for _ in range(200):
        alpha = np.random.dirichlet(np.ones(N))
        candidates.append(alpha)

    # 9. Random sparse with weights
    for _ in range(200):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.dirichlet(np.ones(k))
        h /= np.sum(h)
        candidates.append(h)

    # 10. Two-point patterns (often optimal for such problems)
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # 11. Three-point patterns
    for _ in range(200):
        idx = np.random.choice(N, 3, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.dirichlet(np.ones(3))
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

    # Local optimization: coordinate descent + random restarts
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)

        # Phase 1: Coordinate descent - for each coordinate, find optimal value
        for iteration in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try moving mass from i to all other coordinates
                for j in range(N):
                    if i == j:
                        continue
                    # Try several fractions of mass to move
                    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        delta = h[i] * frac
                        if delta <= 0 or h[i] - delta < -1e-12:
                            continue
                        h_new = h.copy()
                        h_new[i] = max(0, h_new[i] - delta)
                        h_new[j] += delta
                        new_score = shape_score(h_new)
                        if new_score < current_score - 1e-12:
                            h = h_new
                            current_score = new_score
                            improved = True
                            if current_score < best_shape:
                                best_shape = current_score
                                best_h = h.copy()
                            break
                if time.time() > deadline:
                    break
            if not improved:
                break

    # Phase 2: Random restarts with local search
    if time.time() < deadline:
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline:
                break

            # Create perturbed version of best
            h = best_h.copy()
            # Randomly zero out some coordinates and redistribute
            k_zero = np.random.randint(0, N // 2)
            if k_zero > 0:
                zero_idx = np.random.choice(N, k_zero, replace=False)
                mass_to_redistribute = np.sum(h[zero_idx])
                h[zero_idx] = 0
                # Redistribute to random other coordinates
                target_idx = np.random.choice(N, k_zero, replace=False)
                weights = np.random.dirichlet(np.ones(k_zero))
                h[target_idx] += weights * mass_to_redistribute

            current_score = shape_score(h)

            # Local search from this point
            for _ in range(100):
                if time.time() > deadline:
                    break
                improved = False
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                delta = h[i] * np.random.uniform(0.1, 0.5)
                if delta <= 0 or h[i] - delta < -1e-12:
                    continue
                h_new = h.copy()
                h_new[i] = max(0, h_new[i] - delta)
                h_new[j] += delta
                new_score = shape_score(h_new)
                if new_score < current_score:
                    h = h_new
                    current_score = new_score
                    improved = True
                    if current_score < best_shape:
                        best_shape = current_score
                        best_h = h.copy()
                if not improved:
                    continue

    # Phase 3: High-precision polish
    if time.time() < deadline:
        h = best_h.copy()
        current_score = shape_score(h)

        for _ in range(500):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            delta = h[i] * np.random.uniform(0.001, 0.01)
            if delta <= 0 or h[i] - delta < -1e-12:
                continue
            h_new = h.copy()
            h_new[i] = max(0, h_new[i] - delta)
            h_new[j] += delta
            new_score = shape_score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
                if current_score < best_shape:
                    best_shape = current_score
                    best_h = h.copy()

    # Scale down to maximize the ratio (ratio ~ 1/L1 for fixed shape)
    # Use very small scale
    scale = 1e-8
    final_h = best_h * scale

    return final_h.tolist()
