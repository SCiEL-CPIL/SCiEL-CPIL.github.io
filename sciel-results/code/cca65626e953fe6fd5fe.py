"""Program for small nonzero signed harmonic subsums."""
import time
import numpy as np
from fractions import Fraction
from math import gcd
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    Uses dynamic programming + meet-in-the-middle + local search.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    n = 30

    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]

    # Meet-in-the-middle with dynamic programming
    half = n // 2
    left_weights = weights[:half]
    right_weights = weights[half:]

    # Enumerate all possible sums for left half using DP
    # We use a set to avoid duplicates
    left_sums = {0}
    for w in left_weights:
        new_sums = set()
        for s in left_sums:
            new_sums.add(s)
            new_sums.add(s + w)
            new_sums.add(s - w)
        left_sums = new_sums

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Enumerate all possible sums for right half
    right_sums = {0}
    for w in right_weights:
        new_sums = set()
        for s in right_sums:
            new_sums.add(s)
            new_sums.add(s + w)
            new_sums.add(s - w)
        right_sums = new_sums

    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Find the minimum nonzero |l + r|
    left_list = sorted(left_sums)
    right_list = sorted(right_sums)

    best_int_sum = None
    best_left_sum = None
    best_right_sum = None

    # Two-pointer search
    i = 0
    j = len(right_list) - 1

    while i < len(left_list) and j >= 0:
        if time.time() > deadline:
            break

        l_val = left_list[i]
        r_val = right_list[j]
        current_sum = l_val + r_val

        if current_sum != 0:
            abs_sum = abs(current_sum)
            if best_int_sum is None or abs_sum < best_int_sum:
                best_int_sum = abs_sum
                best_left_sum = l_val
                best_right_sum = r_val

                if best_int_sum == 1:
                    break

        if current_sum > 0:
            j -= 1
        else:
            i += 1

    if best_int_sum is None:
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients using backtracking
    def reconstruct(weights, target):
        """Backtracking to find coefficients in {-1,0,1} summing to target."""
        coeffs = [0] * len(weights)

        # Precompute suffix sums for pruning
        suffix_sums = [0] * (len(weights) + 1)
        for i in range(len(weights) - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + abs(weights[i])

        def bt(idx, cur_sum):
            if idx == len(weights):
                return cur_sum == target

            # Pruning
            if abs(target - cur_sum) > suffix_sums[idx]:
                return False

            w = weights[idx]
            for val in (0, 1, -1):
                coeffs[idx] = val
                if bt(idx + 1, cur_sum + val * w):
                    return True
            coeffs[idx] = 0
            return False

        if bt(0, 0):
            return coeffs
        return None

    left_coeffs = reconstruct(left_weights, int(best_left_sum))
    right_coeffs = reconstruct(right_weights, int(best_right_sum))

    if left_coeffs is None or right_coeffs is None:
        return [0] * 28 + [1, -1]

    result = left_coeffs + right_coeffs

    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(result) if c != 0)
    if total_frac == 0:
        return [0] * 28 + [1, -1]

    # Local search to try to improve
    best_result = result
    best_abs_value = abs(total_frac)

    # Try random perturbations
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline:
            break

        # Create a perturbed version
        perturbed = best_result[:]
        # Flip 1-3 random coefficients
        num_flips = random.randint(1, 3)
        for _ in range(num_flips):
            idx = random.randint(0, n - 1)
            perturbed[idx] = -perturbed[idx] if perturbed[idx] != 0 else random.choice([-1, 1])

        # Calculate new value
        new_frac = sum(Fraction(c, i + 1) for i, c in enumerate(perturbed) if c != 0)
        if new_frac == 0:
            continue

        new_abs = abs(new_frac)
        if new_abs < best_abs_value:
            best_abs_value = new_abs
            best_result = perturbed[:]

    return best_result

# EVOLVE-BLOCK-END
