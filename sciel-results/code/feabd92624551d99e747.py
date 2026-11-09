import numpy as np

def _compute_P(s, w):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(s)
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    P = 0.0
    for l in range(n):
        for k in range(n):
            threshold = 2.0 * s[l] - s[k]
            inner_sum = np.sum(W[S < threshold])
            P += w[l] * w[k] * inner_sum
    return P

def _eval(s, w):
    """Validate and return P, or -1 if invalid."""
    s = np.asarray(s, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()
    if len(s) != len(w) or len(s) < 1:
        return -1.0
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return -1.0
    if np.any(np.isnan(s)) or np.any(np.isnan(w)) or np.any(np.isinf(s)) or np.any(np.isinf(w)):
        return -1.0
    if w.sum() <= 1e-9:
        return -1.0
    w = w / w.sum()
    idx = np.argsort(s)
    s = s[idx]
    w = w[idx]
    return _compute_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1.0
    best_s = None
    best_w = None

    def try_dist(s, w):
        nonlocal best_P, best_s, best_w
        P = _eval(s, w)
        if P > best_P:
            best_P = P
            best_s = np.array(s, dtype=np.float64).flatten()
            best_w = np.array(w, dtype=np.float64).flatten()

    # Family A: 2-point {0, M}
    # Analytic optimum: p = (15-sqrt(33))/16
    p_star = (15.0 - np.sqrt(33.0)) / 16.0
    for M in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        try_dist([0.0, M], [p_star, 1.0 - p_star])
    # Sweep p for various M
    for M in [1.0, 2.0, 5.0, 10.0, 50.0]:
        for p in np.linspace(0.2, 0.9, 100):
            try_dist([0.0, M], [p, 1.0 - p])

    # Family B: 3-point {0, a, M}
    for M in [2.0, 5.0, 10.0, 50.0]:
        for a in np.linspace(0.05, 0.9 * M, 30):
            for p in np.linspace(0.05, 0.8, 40):
                for q in np.linspace(0.01, 0.6, 40):
                    r = 1.0 - p - q
                    if r < 0.01 or r > 0.99:
                        continue
                    try_dist([0.0, a, M], [p, q, r])

    # Family C: 3-point {0, 1, M}
    for M in [2.0, 5.0, 10.0, 50.0]:
        for p in np.linspace(0.05, 0.8, 40):
            for q in np.linspace(0.01, 0.6, 40):
                r = 1.0 - p - q
                if r < 0.01 or r > 0.99:
                    continue
                try_dist([0.0, 1.0, M], [p, q, r])

    # Family D: 4-point {0, a, b, M} coarse
    for M in [5.0, 20.0]:
        for a in np.linspace(0.1, 0.3 * M, 8):
            for b in np.linspace(a + 0.1, 0.7 * M, 8):
                for p in np.linspace(0.1, 0.5, 8):
                    for q in np.linspace(0.05, 0.4, 8):
                        for r in np.linspace(0.05, 0.4, 8):
                            s_prob = 1.0 - p - q - r
                            if s_prob < 0.01 or s_prob > 0.99:
                                continue
                            try_dist([0.0, a, b, M], [p, q, r, s_prob])

    # Family E: 2-point {a, M} with a > 0
    for a in np.linspace(0.1, 1.0, 20):
        for M in np.linspace(a + 0.1, 20.0, 20):
            for p in np.linspace(0.2, 0.8, 50):
                try_dist([a, M], [p, 1.0 - p])

    # Family F: 3-point {a, b, M} all positive
    for M in [5.0, 20.0]:
        for a in np.linspace(0.1, 0.3 * M, 10):
            for b in np.linspace(a + 0.1, 0.7 * M, 10):
                for p in np.linspace(0.1, 0.5, 15):
                    for q in np.linspace(0.05, 0.5, 15):
                        r = 1.0 - p - q
                        if r < 0.01 or r > 0.99:
                            continue
                        try_dist([a, b, M], [p, q, r])

    if best_s is None:
        best_s = np.array([0.0, 1.0])
        best_w = np.array([p_star, 1.0 - p_star])

    support = np.asarray(best_s, dtype=np.float64).flatten()
    weights = np.asarray(best_w, dtype=np.float64).flatten()

    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])

    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])

    if weights.sum() <= 1e-9:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])

    return support, weights
