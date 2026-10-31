"""Cap set greedy priority: static heuristic favoring balanced, low-exposure vectors."""
import time

_DEADLINE = 330.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def priority(vector, n):
    # Deadline guard: stop doing heavy work near the wall-clock limit.
    if _elapsed() > _DEADLINE - 30.0:
        # Return a simple fast value; sorting will still produce a valid order.
        return float(sum(vector))

    # Static heuristic: prefer vectors with balanced coordinate distribution.
    # Count occurrences of each symbol; balanced (close to n/3 each) is better.
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    # Penalty for imbalance (deviation from uniform n/3).
    target = n / 3.0
    imbalance = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    # Higher priority = lower imbalance.
    base = -imbalance

    # Tiebreak: a deterministic pseudo-random spread to diversify order.
    # Use a lightweight hash of the vector.
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 7)) & 0xFFFFFFFF
    # Small perturbation so ties are broken by diversity, not lexicographic.
    return base + (h / float(0xFFFFFFFF)) * 1e-4
