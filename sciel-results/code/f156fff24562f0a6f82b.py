"""Find large reciprocal-subsum-free subsets of [1,24]."""
from fractions import Fraction
from itertools import combinations
import time

def construct():
    """
    Return a nonempty subset of [1,24] maximizing cardinality such that
    no reciprocal of a selected element equals the sum of reciprocals of a
    subset of the OTHER selected elements.
    """
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best_set = list(range(13, 25))  # fallback: seed answer (size 12)
    best_size = len(best_set)

    n = 24
    elements = list(range(1, n + 1))
    recip = {i: Fraction(1, i) for i in elements}

    # Precompute all subset sums of reciprocals for all subsets of [1..24]
    # This is 2^24 = 16,777,216 subsets -- too many to precompute all.
    # Instead, use meet-in-the-middle per candidate set.

    def is_valid(subset):
        """Check if subset is reciprocal-subsum-free."""
        s = set(subset)
        for elem in s:
            target = recip[elem]
            # Check all subsets of s \ {elem}
            others = [x for x in s if x != elem]
            m = len(others)
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]

            # Generate all subset sums for left
            left_sums = set()
            for mask in range(1 << len(left)):
                s_val = Fraction(0, 1)
                for i in range(len(left)):
                    if mask & (1 << i):
                        s_val += recip[left[i]]
                left_sums.add(s_val)

            # For each subset of right, check if target - right_sum is in left_sums
            for mask in range(1 << len(right)):
                r_val = Fraction(0, 1)
                for i in range(len(right)):
                    if mask & (1 << i):
                        r_val += recip[right[i]]
                if (target - r_val) in left_sums:
                    return False
        return True

    # Search from largest size downward
    for size in range(24, best_size, -1):
        if time.time() > deadline - 5.0:
            break
        found = False
        for combo in combinations(elements, size):
            if time.time() > deadline - 5.0:
                break
            if is_valid(combo):
                best_set = list(combo)
                best_size = size
                found = True
                break
        if found:
            break

    return best_set
