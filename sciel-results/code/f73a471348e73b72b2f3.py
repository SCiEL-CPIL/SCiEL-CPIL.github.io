"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 120
    best = np.ones(N)
    best_val = 2.0

    # Generate diverse starting patterns
    rng = np.random.RandomState(42)
    patterns = [
        np.ones(N),
        rng.standard_normal(N),
        rng.standard_normal(N),
        # Sparse patterns
        np.zeros(N),
        # Block patterns
        np.ones(N),
    ]
    
    # Create sparse patterns
    for k in [1, 2, 3, 5, 10, 20, 50]:
        idx = rng.choice(N, size=k, replace=False)
        p = np.zeros(N)
        p[idx] = 1.0
        patterns.append(p)
    
    # Create block patterns
    for block_len in [1, 2, 5, 10, 50]:
        p = np.zeros(N)
        p[:block_len] = 1.0
        patterns.append(p)
    
    # Create alternating patterns
    alt = np.ones(N)
    alt[::2] = -1.0
    patterns.append(alt)
    
    # Try each pattern
    for p in patterns:
        if time.time() > deadline - 2:
            break
        try:
            val = abs(2 * N * np.max(np.convolve(p, p)) / np.sum(p) ** 2)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best = p.copy()
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue

    # Simple local search from best pattern
    if best_val < 2.0:
        current = best.copy()
        current_val = best_val
        step_size = 0.01
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old_val = current[i]
                for delta in [-step_size, step_size]:
                    new_val = old_val + delta
                    if new_val == 0 and abs(old_val) < 1e-10:
                        continue
                    current[i] = new_val
                    s = np.sum(current)
                    if abs(s) < 1e-10:
                        current[i] = old_val
                        continue
                    try:
                        val = abs(2 * N * np.max(np.convolve(current, current)) / s ** 2)
                        if np.isfinite(val) and val < current_val:
                            current_val = val
                            improved = True
                            step_size = min(step_size * 1.5, 1.0)
                        else:
                            current[i] = old_val
                    except (ZeroDivisionError, FloatingPointError, ValueError):
                        current[i] = old_val
                if time.time() > deadline - 5:
                    break
            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    break

        if current_val < best_val:
            best_val = current_val
            best = current.copy()

    return best
# EVOLVE-BLOCK-END
