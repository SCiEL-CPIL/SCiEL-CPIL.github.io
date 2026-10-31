"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Generate minimal Sidon sets for various sizes n
    - Sidon sets maximize |U-U| for a given n
    - For small n, the ratio |U-U|/|U+U| is high, and M is small
    - Select the set with the highest score
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def is_sidon(U):
        """Check if U is a Sidon set (B2 set)."""
        U_set = set(U)
        sums = set()
        diffs = set()
        for a in U_set:
            for b in U_set:
                if a < b:
                    if (a, b) in sums or (b, a) in sums:
                        return False
                    sums.add((a, b))
                    sums.add((b, a))
                elif a == b:
                    if (a, a) in sums:
                        return False
                    sums.add((a, a))
                d = a - b
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def generate_minimal_sidon(n):
        """
        Greedily construct a Sidon set of size n with minimal max element.
        Starts with 0, then adds the smallest possible integer that preserves the Sidon property.
        """
        U = [0]
        for _ in range(1, n):
            found = False
            for candidate in range(1, 1000):
                if candidate in U:
                    continue
                new_U = U + [candidate]
                if is_sidon(new_U):
                    U = new_U
                    found = True
                    break
            if not found:
                return None
        return U

    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try small n values
    for n in range(2, 16):
        U = generate_minimal_sidon(n)
        if U is None:
            continue
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U

    # Also try a few random perturbations of the best small sets to see if non-Sidon is better
    # (Though Sidon is theoretically optimal for the ratio, check just in case)
    for _ in range(50):
        n = random.randint(3, 8)
        max_val = random.randint(10, 50)
        U = [0]
        U_set = {0}
        for _ in range(n - 1):
            candidates = [x for x in range(1, max_val + 1) if x not in U_set]
            if not candidates:
                break
            # Pick a random candidate
            c = random.choice(candidates)
            U.append(c)
            U_set.add(c)
        U = sorted(U)
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U

    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
