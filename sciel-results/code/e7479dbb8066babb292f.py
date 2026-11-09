"""Refined cap set greedy ordering using balanced-digit heuristics."""
import time

_DEADLINE = time.monotonic() + 280.0
_START = time.monotonic()

def priority(vector, n):
    """Return a float priority; higher vectors are inserted first.
    
    Uses a multi-level heuristic:
    1. Quadratic form band (Q=1 is best).
    2. Balance of 1s and 2s (lower |n1-n2| is better).
    3. Hamming weight (lower is better).
    """
    if time.time() > _DEADLINE:
        return 0.0

    q = 0
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
            q += 1
        elif x == 2:
            n2 += 1
            q += 1
    q %= 3

    # Band: Q=1 is the cap set class
    if q == 1:
        band = 1000.0
    elif q == 0:
        band = 500.0
    else:
        band = 0.0

    # Balance: prefer vectors with similar counts of 1 and 2
    balance = abs(n1 - n2)
    
    # Weight: prefer lower Hamming weight
    weight = n1 + n2

    # Combine: Band is dominant. Balance is secondary. Weight is tertiary.
    # Scale to ensure separation.
    return band - 1.0 * balance - 0.1 * weight
