"""Rewritten program for the second autocorrelation inequality task."""
import numpy as np
import time

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)


def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Generate diverse parameterized starting shapes
    x = np.linspace(-1, 1, N)
    starts = []

    # Gaussian bumps with varying center and width
    for c in np.linspace(-0.7, 0.7, 9):
        for s in np.linspace(0.15, 0.8, 9):
            h = np.exp(-((x - c) ** 2) / (2 * s ** 2))
            starts.append(h)

    # Cosine bumps
    for c in np.linspace(-0.7, 0.7, 9):
        for s in np.linspace(0.2, 0.8, 9):
            mask = np.abs(x - c) < s
            h = np.zeros(N)
            h[mask] = 0.5 * (1 + np.cos(np.pi * (x[mask] - c) / s))
            starts.append(h)

    # Triangular bumps
    for c in np.linspace(-0.7, 0.7, 9):
        for s in np.linspace(0.2, 0.8, 9):
            h = np.maximum(0, 1 - np.abs(x - c) / s)
            starts.append(h)

    # Box bumps
    for c in np.linspace(-0.7, 0.7, 9):
        for s in np.linspace(0.1, 0.7, 9):
            mask = np.abs(x - c) < s
            h = np.zeros(N)
            h[mask] = 1.0
            starts.append(h)

    # Evaluate all starts
    for h in starts:
        if time.time() > deadline:
            break
        val = compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()

    # Sort starts by value and take top candidates for local refinement
    start_vals = []
    for h in starts:
        if time.time() > deadline:
            break
        val = compute_value(h)
        start_vals.append((val, h))
    start_vals.sort(key=lambda x: x[0], reverse=True)

    # Local optimization using coordinate descent + random perturbations
    top_starts = [h for v, h in start_vals[:8]]

    for h0 in top_starts:
        if time.time() > deadline - 15:
            break
        h_local = h0.copy()
        v_local = compute_value(h_local)
        if v_local > best_val:
            best_val = v_local
            best_h = h_local.copy()

        # Coordinate descent with adaptive step
        step_size = 0.1
        for it in range(300):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                old = h_local[i]
                for delta in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                    h_local[i] = max(0.0, old + delta)
                    v = compute_value(h_local)
                    if v > v_local + 1e-12:
                        v_local = v
                        improved = True
                        break
                else:
                    h_local[i] = old
            if not improved:
                step_size *= 0.7
                if step_size < 1e-4:
                    break
            else:
                step_size *= 1.05

        if v_local > best_val:
            best_val = v_local
            best_h = h_local.copy()

    # Global random perturbation search
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            h_rand = best_h + 0.1 * rng.standard_normal(N)
            h_rand = np.maximum(0, h_rand)
            v = compute_value(h_rand)
            if v > best_val:
                best_val = v
                best_h = h_rand

    return best_h.tolist()
