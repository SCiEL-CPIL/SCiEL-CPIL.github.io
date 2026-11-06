"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    # Deadline guard: evaluator kills at 90s; we stop searching at 80s.
    deadline = time.time() + 80.0

    # Denominators are 2..48, indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)

    # Greedy initial solution: add fractions in increasing denominator order
    # as long as sum stays strictly below 1.
    best_set = set()
    best_sum = Fraction(0, 1)
    for i, d in enumerate(denoms):
        if best_sum + Fraction(1, d) < 1:
            best_set.add(i)
            best_sum += Fraction(1, d)
    best_deficit = 1 - best_sum

    # Precompute 1/d as Fractions
    recip = [Fraction(1, d) for d in denoms]

    # Local search: try to improve by swapping or adding/removing
    # We want to minimize deficit = 1 - sum, i.e., maximize sum while sum < 1.

    # Strategy: iterative improvement with multiple passes
    # 1) Try adding any unselected index if it keeps sum < 1
    # 2) Try removing one and adding two (swap) to get closer to 1
    # 3) Try removing one and adding one

    def try_add(current_set, current_sum):
        """Try adding one more element."""
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum + recip[i]
                if new_sum < 1:
                    return i, new_sum
        return None, None

    def try_swap_remove_one_add_one(current_set, current_sum):
        """Try removing one element and adding another to get closer to 1."""
        best_gain = Fraction(0, 1)
        best_move = None
        for i in list(current_set):
            without_i = current_sum - recip[i]
            for j in range(n):
                if j not in current_set:
                    new_sum = without_i + recip[j]
                    if new_sum < 1:
                        gain = new_sum - current_sum
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, j, new_sum)
        return best_move

    def try_swap_remove_one_add_two(current_set, current_sum):
        """Try removing one element and adding two others."""
        best_gain = Fraction(0, 1)
        best_move = None
        for i in list(current_set):
            without_i = current_sum - recip[i]
            # Try all pairs (j, k) not in current_set, j < k
            for j in range(n):
                if j in current_set:
                    continue
                for k in range(j + 1, n):
                    if k in current_set:
                        continue
                    new_sum = without_i + recip[j] + recip[k]
                    if new_sum < 1:
                        gain = new_sum - current_sum
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, j, k, new_sum)
        return best_move

    def try_remove_one_add_three(current_set, current_sum):
        """Try removing one element and adding three others."""
        best_gain = Fraction(0, 1)
        best_move = None
        for i in list(current_set):
            without_i = current_sum - recip[i]
            # To limit complexity, only try if current set is small enough
            for j in range(n):
                if j in current_set:
                    continue
                for k in range(j + 1, n):
                    if k in current_set:
                        continue
                    for m in range(k + 1, n):
                        if m in current_set:
                            continue
                        new_sum = without_i + recip[j] + recip[k] + recip[m]
                        if new_sum < 1:
                            gain = new_sum - current_sum
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (i, j, k, m, new_sum)
        return best_move

    # Iterative improvement loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        current_sum = best_sum

        # Phase 1: try adding one
        idx, new_sum = try_add(current_set, current_sum)
        if idx is not None:
            current_set.add(idx)
            current_sum = new_sum
            best_set = set(current_set)
            best_sum = current_sum
            best_deficit = 1 - best_sum
            improved = True
            continue

        # Phase 2: try swap remove-one-add-one
        move = try_swap_remove_one_add_one(current_set, current_sum)
        if move is not None:
            i, j, new_sum = move
            current_set.discard(i)
            current_set.add(j)
            current_sum = new_sum
            best_set = set(current_set)
            best_sum = current_sum
            best_deficit = 1 - best_sum
            improved = True
            continue

        # Phase 3: try swap remove-one-add-two
        move = try_swap_remove_one_add_two(current_set, current_sum)
        if move is not None:
            i, j, k, new_sum = move
            current_set.discard(i)
            current_set.add(j)
            current_set.add(k)
            current_sum = new_sum
            best_set = set(current_set)
            best_sum = current_sum
            best_deficit = 1 - best_sum
            improved = True
            continue

        # Phase 4: try swap remove-one-add-three (more expensive)
        if len(current_set) <= 20:  # limit for performance
            move = try_remove_one_add_three(current_set, current_sum)
            if move is not None:
                i, j, k, m, new_sum = move
                current_set.discard(i)
                current_set.add(j)
                current_set.add(k)
                current_set.add(m)
                current_sum = new_sum
                best_set = set(current_set)
                best_sum = current_sum
                best_deficit = 1 - best_sum
                improved = True
                continue

    # Final: ensure we return a valid nonempty list
    if not best_set:
        # Fallback: just take index 0 (denom 2)
        return [0]

    return sorted(best_set)
# EVOLVE-BLOCK-END
