import numpy as np
from math import gcd
from functools import reduce
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = 30

    def _evaluate(coeffs):
        """Compute sum c_i / (i+1) for i in range(n), return (numerator, denominator, abs_value)."""
        num = 0
        den = 1
        for i in range(n):
            if coeffs[i] != 0:
                num = num * (i + 1) + coeffs[i] * den
                den = den * (i + 1)
        # Reduce
        g = gcd(abs(num), den)
        num //= g
        den //= g
        return num, den, abs(num) / den

    def _local_search(coeffs, deadline):
        """Iteratively flip coefficients to minimize |sum c_i/i|, avoiding zero."""
        best_num, best_den, best_val = _evaluate(coeffs)
        if best_val == 0:
            # Need nonzero; find best nonzero
            best_val = float('inf')
            best_num, best_den = 0, 1

        while time.time() < deadline:
            improved = False
            for i in range(n):
                old = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    coeffs[i] = new_val
                    num, den, val = _evaluate(coeffs)
                    if val > 0 and val < best_val:
                        best_val = val
                        best_num = num
                        best_den = den
                        improved = True
                    else:
                        coeffs[i] = old
            if not improved:
                break

        return best_num, best_den, best_val

    # Multiple random restarts
    best_overall_val = float('inf')
    best_overall_coeffs = None

    # Try different random seeds
    rng = random.Random(42)
    num_restarts = 200

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initialization
        coeffs = [rng.choice([-1, 0, 1]) for _ in range(n)]

        # Run local search
        num, den, val = _local_search(coeffs, deadline)

        if val > 0 and val < best_overall_val:
            best_overall_val = val
            best_overall_coeffs = coeffs[:]

        # Also try a structured initialization: pairs that cancel
        if restart % 2 == 1:
            coeffs = [0] * n
            # Create some random pairs
            for i in range(0, n - 1, 2):
                if rng.random() < 0.7:
                    coeffs[i] = 1
                    coeffs[i + 1] = -1
            num, den, val = _local_search(coeffs, deadline)
            if val > 0 and val < best_overall_val:
                best_overall_val = val
                best_overall_coeffs = coeffs[:]

    # Final polish: try to improve the best solution further
    if best_overall_coeffs is not None and time.time() < deadline - 5:
        # More aggressive local search
        coeffs = best_overall_coeffs[:]
        best_num, best_den, best_val = _local_search(coeffs, deadline)
        if best_val > 0 and best_val < best_overall_val:
            best_overall_val = best_val
            best_overall_coeffs = coeffs[:]

    if best_overall_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]

    return best_overall_coeffs
