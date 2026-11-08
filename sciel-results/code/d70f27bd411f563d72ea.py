import time
import numpy as np

N = 50

def _ratio(h):
    """Return the ratio of squared L2 norm to (L1 norm * supremum norm of autoconvolution).
    The constant step-width factor cancels, so this is equivalent to the true V."""
    h = np.asarray(h, dtype=np.float64)
    s1 = float(np.sum(h))
    if s1 <= 0:
        return 0.0
    s2 = float(np.dot(h, h))
    # linear autoconvolution
    c = np.convolve(h, h)
    m = float(np.max(c))
    if m <= 0:
        return 0.0
    return s2 / s1 / m

def construct():
    t0 = time.time()
    deadline = t0 + 140.0   # safety margin: evaluator kills at 150s

    # Seed: uniform
    best = np.ones(N, dtype=np.float64)
    best_val = _ratio(best)

    # ---- Phase 1: random initializations + coordinate ascent ----
    rng = np.random.default_rng(12345)

    def coordinate_ascent(h, max_iters=500):
        h = h.copy()
        for it in range(max_iters):
            improved = False
            for j in range(N):
                old = h[j]
                # Try a small set of candidate values
                # Optimal h[j] given others: we want to maximize s2/s1/m
                # This is nontrivial; use heuristic: try geometric steps
                candidates = [0.0, old]
                if old > 0:
                    for f in [0.2, 0.5, 0.8, 1.0, 1.25, 2.0, 5.0]:
                        candidates.append(old * f)
                else:
                    for v in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                        candidates.append(v)
                best_j = old
                best_val_j = _ratio(h)
                for v in candidates:
                    h[j] = v
                    val = _ratio(h)
                    if val > best_val_j + 1e-12:
                        best_val_j = val
                        best_j = v
                h[j] = best_j
                if best_j != old:
                    improved = True
            if not improved:
                break
        return h

    # Try multiple random starts
    n_starts = 8
    for k in range(n_starts):
        if time.time() > deadline:
            break
        # Random nonnegative heights
        if k == 0:
            h = np.ones(N)
        elif k == 1:
            # Spike
            h = np.zeros(N)
            h[N // 2] = 1.0
        elif k == 2:
            # Ramp
            h = np.linspace(0.1, 1.0, N)
        elif k == 3:
            # Symmetric bump
            h = np.exp(-0.5 * ((np.arange(N) - N / 2) / (N / 4)) ** 2)
        else:
            h = np.abs(rng.standard_normal(N)) ** 2
            h = h / h.max()

        h = coordinate_ascent(h, max_iters=300)
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best = h.copy()

    # ---- Phase 2: local refinement via small random perturbations ----
    n_perturb = 200
    for k in range(n_perturb):
        if time.time() > deadline:
            break
        h = best.copy()
        # Random perturbation on a subset of coordinates
        idx = rng.choice(N, size=rng.integers(1, 10), replace=False)
        for j in idx:
            h[j] *= np.exp(rng.standard_normal() * 0.3)
            if h[j] < 0:
                h[j] = 0.0
        h = coordinate_ascent(h, max_iters=100)
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best = h.copy()

    # ---- Phase 3: try specific structured families ----
    # Symmetric triangular
    for width in [N // 4, N // 3, N // 2, 2 * N // 3, N - 1]:
        if time.time() > deadline:
            break
        h = np.zeros(N)
        center = N // 2
        for d in range(width + 1):
            idx1 = center - d
            idx2 = center + d
            if 0 <= idx1 < N:
                h[idx1] = 1.0 - d / (width + 1)
            if 0 <= idx2 < N and idx2 != idx1:
                h[idx2] = 1.0 - d / (width + 1)
        if h.max() > 0:
            h = h / h.max()
            h = coordinate_ascent(h, max_iters=200)
            val = _ratio(h)
            if val > best_val:
                best_val = val
                best = h.copy()

    # Two-spike family
    for gap in [1, 5, 10, 15, 20, 25]:
        if time.time() > deadline:
            break
        h = np.zeros(N)
        h[0] = 1.0
        h[gap] = 1.0
        h = coordinate_ascent(h, max_iters=300)
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best = h.copy()

    # Return as list of floats
    return [float(x) for x in best]
