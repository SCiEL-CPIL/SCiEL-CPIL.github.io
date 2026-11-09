# EVOLVE-BLOCK-START
import numpy as np
import time


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize H(x-y) / max(H(x), H(x+y), H(x+2y)) over 10x10 distributions.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin within 630s limit

    n = 10
    xs = np.arange(n)
    ys = np.arange(n)

    # Precompute pooled value indices
    val_x = xs  # 0..9
    val_xmy = xs - ys  # -9..9
    val_xpy = xs + ys  # 0..18
    val_x2y = xs + 2 * ys  # 0..27

    def entropy_from_pool(counts):
        """Shannon entropy in bits from pooled counts (positive ones)."""
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def evaluate(P):
        """Compute D/M ratio for a 10x10 distribution P."""
        P = np.clip(P, 0, None)
        s = P.sum()
        if s <= 1e-12:
            return 0.0
        P = P / s

        # Pool for x - y (D)
        counts_xmy = np.zeros(19)
        for y in range(n):
            for x in range(n):
                idx = x - y + 9
                counts_xmy[idx] += P[y, x]
        D = entropy_from_pool(counts_xmy)

        # Pool for x (slope 0)
        counts_x = np.zeros(10)
        for y in range(n):
            for x in range(n):
                counts_x[x] += P[y, x]
        H0 = entropy_from_pool(counts_x)

        # Pool for x+y (slope 1)
        counts_xpy = np.zeros(19)
        for y in range(n):
            for x in range(n):
                idx = x + y
                counts_xpy[idx] += P[y, x]
        H1 = entropy_from_pool(counts_xpy)

        # Pool for x+2y (slope 2)
        counts_x2y = np.zeros(28)
        for y in range(n):
            for x in range(n):
                idx = x + 2 * y
                counts_x2y[idx] += P[y, x]
        H2 = entropy_from_pool(counts_x2y)

        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 0.0
        return D / M

    # Start with uniform
    P_best = np.ones((n, n)) / (n * n)
    best_val = evaluate(P_best)

    # Local optimization using coordinate ascent / hill climbing
    # Try small perturbations
    rng = np.random.RandomState(42)

    step_size = 0.05
    n_iterations = 5000

    for it in range(n_iterations):
        if time.time() > deadline:
            break

        # Random direction
        direction = rng.randn(n, n)
        direction -= direction.mean()  # keep sum approximately constant
        direction /= (np.abs(direction).max() + 1e-12)

        P_new = P_best + step_size * direction
        P_new = np.clip(P_new, 0, None)
        s = P_new.sum()
        if s <= 1e-12:
            continue
        P_new = P_new / s

        val_new = evaluate(P_new)
        if val_new > best_val:
            P_best = P_new.copy()
            best_val = val_new

    # Additional refinement: try structured distributions
    # Concentrate on diagonals x-y = const to maximize D
    for target_diag in range(-9, 10):
        if time.time() > deadline:
            break
        P_struct = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x - y == target_diag:
                    P_struct[y, x] = 1.0
        s = P_struct.sum()
        if s > 0:
            P_struct /= s
            val_struct = evaluate(P_struct)
            if val_struct > best_val:
                P_best = P_struct.copy()
                best_val = val_struct

    # Try anti-diagonal concentration (x+y = const)
    for target_sum in range(0, 19):
        if time.time() > deadline:
            break
        P_struct = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + y == target_sum:
                    P_struct[y, x] = 1.0
        s = P_struct.sum()
        if s > 0:
            P_struct /= s
            val_struct = evaluate(P_struct)
            if val_struct > best_val:
                P_best = P_struct.copy()
                best_val = val_struct

    # Try x+2y concentration
    for target_val in range(0, 28):
        if time.time() > deadline:
            break
        P_struct = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + 2 * y == target_val:
                    P_struct[y, x] = 1.0
        s = P_struct.sum()
        if s > 0:
            P_struct /= s
            val_struct = evaluate(P_struct)
            if val_struct > best_val:
                P_best = P_struct.copy()
                best_val = val_struct

    # Final cleanup: ensure validity
    P_best = np.clip(P_best, 0, None)
    s = P_best.sum()
    if s <= 1e-12:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best = P_best / s

    return P_best


# EVOLVE-BLOCK-END
