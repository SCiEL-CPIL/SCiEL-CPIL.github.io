import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    hh = np.convolve(h, h, mode='full')
    inf_norm = float(np.max(hh))
    if inf_norm < 1e-15:
        return 0.0
    return l2_sq / (l1 * inf_norm)

def construct():
    """
    Returns a sequence of 50 nonnegative step heights that maximizes the ratio
    ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s kill

    # Initialize with single impulse (known score 1.0)
    best_h = np.zeros(N, dtype=float)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    # Try several structured configurations
    candidates = []

    # Uniform
    candidates.append(np.ones(N, dtype=float) / N)

    # Triangular
    candidates.append(np.linspace(0, 1, N))
    candidates.append(np.linspace(1, 0, N))

    # Two impulses at various gaps
    for gap in [1, 2, 5, 10, 25]:
        h_two = np.zeros(N, dtype=float)
        h_two[0] = 1.0
        h_two[gap] = 1.0
        candidates.append(h_two)

    # Block of k consecutive ones
    for k in [2, 3, 5, 10]:
        h_block = np.zeros(N, dtype=float)
        h_block[:k] = 1.0
        candidates.append(h_block)

    # Evaluate all candidates
    for h_cand in candidates:
        s = compute_ratio(h_cand)
        if s > best_score:
            best_score = s
            best_h = h_cand.copy()

    # Random search with local optimization
    rng = np.random.RandomState(42)
    for iteration in range(2000):
        if time.time() > deadline:
            break

        # Generate random candidate
        h_rand = rng.rand(N)
        # Occasionally try sparse candidates
        if iteration % 5 == 0:
            mask = rng.rand(N) < 0.2
            h_rand = np.where(mask, rng.rand(N), 0.0)

        s_rand = compute_ratio(h_rand)
        if s_rand > best_score:
            best_score = s_rand
            best_h = h_rand.copy()

        # Local refinement: try perturbing one element
        if best_score > 1.0 and time.time() < deadline - 5:
            h_opt = best_h.copy()
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    old_val = h_opt[i]
                    if old_val < 1e-10:
                        # Try turning on
                        for val in [0.01, 0.1, 0.5, 1.0]:
                            h_test = h_opt.copy()
                            h_test[i] = val
                            s_test = compute_ratio(h_test)
                            if s_test > best_score:
                                best_score = s_test
                                h_opt[i] = val
                                improved = True
                                break
                    else:
                        # Try scaling
                        for factor in [0.1, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 5.0]:
                            h_test = h_opt.copy()
                            h_test[i] = old_val * factor
                            s_test = compute_ratio(h_test)
                            if s_test > best_score:
                                best_score = s_test
                                h_opt[i] = old_val * factor
                                improved = True
                                break
                best_h = h_opt.copy()

    return list(best_h)
