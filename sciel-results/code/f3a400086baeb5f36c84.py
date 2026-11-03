import numpy as np
from scipy.optimize import differential_evolution

N = 50


def _ratio(heights):
    """Compute the ratio ||x||_2^2 / (||x||_1 * ||x*x||_inf) for discrete heights."""
    x = np.array(heights, dtype=np.float64)
    if np.any(x < 0) or x.sum() == 0:
        return 0.0
    l2_sq = np.sum(x ** 2)
    l1 = np.sum(x)
    # Discrete autoconvolution
    conv = np.correlate(x, x, mode='full')
    m = np.max(conv)
    if m == 0:
        return 0.0
    return l2_sq / (l1 * m)


def _negative_ratio(params):
    """Negative ratio for minimization."""
    return -_ratio(params)


def construct():
    """
    Returns a sequence of 50 nonnegative step heights that maximize the ratio
    of squared L2 norm to L1 norm times supremum norm of the autoconvolution.
    """
    import time

    deadline = time.time() + 140  # Safety margin before 150s limit

    best_val = 0.0
    best_heights = np.zeros(N)

    # Strategy 1: Try a single spike
    h_spikes = np.zeros(N)
    h_spikes[0] = 1.0
    val_spikes = _ratio(h_spikes)
    if val_spikes > best_val:
        best_val = val_spikes
        best_heights = h_spikes.copy()

    # Strategy 2: Try a two-point configuration
    # x_i = a, x_j = b, rest 0
    # Ratio = (a^2 + b^2) / ((a+b) * max(a^2+b^2, 2ab))
    # If a = b, ratio = (2a^2) / (2a * 2a^2) = 1/2? No.
    # Let's just optimize over a few simple parametric families

    # Family 1: Two points at distance d
    # x[d//2] = 0.5, x[-d//2] = 0.5 (centered)
    for d in range(1, N):
        if time.time() > deadline - 10:
            break
        h = np.zeros(N)
        # Place two non-zero entries
        idx1 = d // 2
        idx2 = (N - 1) - idx1
        if idx1 == idx2:
            continue
        h[idx1] = 0.5
        h[idx2] = 0.5
        val = _ratio(h)
        if val > best_val:
            best_val = val
            best_heights = h.copy()

    # Family 2: Optimized two-point configuration using local search
    # Let's try to optimize a vector with a few non-zero entries using differential evolution
    # over the support and amplitudes.

    # Since N=50 is small, we can try differential evolution on the full 50-dim space
    # but it might be slow. Let's try a restricted search first.

    # Try differential evolution on 50 dimensions with bounds [0, 1]
    # This is computationally expensive, so we limit the iterations.

    bounds = [(0.0, 1.0)] * N
    # Use a small population size to keep it fast
    try:
        result = differential_evolution(
            _negative_ratio,
            bounds,
            maxiter=100,
            popsize=5,
            tol=0.01,
            seed=42,
            workers=1,
            init='latinhypercube',
            polish=True
        )
        if result.success or result.fun < 0:
            h_opt = result.x
            val_opt = -result.fun
            if val_opt > best_val:
                best_val = val_opt
                best_heights = h_opt.copy()
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 5:
        return best_heights.tolist()

    # Try a few more random restarts with local refinement
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        # Random initialization
        h_rand = np.random.rand(N)
        # Normalize to L1=1
        h_rand = h_rand / h_rand.sum()
        val_rand = _ratio(h_rand)
        if val_rand > best_val:
            best_val = val_rand
            best_heights = h_rand.copy()

    return best_heights.tolist()
