"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill

    # Precompute incompatibility: incompatible[i][j] = True if lcm(i+2, j+2) <= 256
    # We work with indices 0..254 representing values 2..256
    N = 255
    values = list(range(2, 257))

    # Build incompatibility matrix (symmetric)
    # incompatible[i] = set of j (j != i) such that lcm(values[i], values[j]) <= 256
    incompatible = [set() for _ in range(N)]
    for i in range(N):
        vi = values[i]
        for j in range(i + 1, N):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                incompatible[i].add(j)
                incompatible[j].add(i)

    # Objective: sum of 1/values[i] for i in selected
    # Use Fraction for exactness

    # Greedy initialization: sort by 1/value descending (i.e., value ascending),
    # but that's just the seed. Better: sort by value ascending and greedily add
    # if compatible with all current. Actually for max reciprocal sum, we want small numbers.
    # But small numbers are more likely to be incompatible with each other.
    # Let's try multiple greedy orders and pick the best.

    def greedy(order):
        selected = []
        sel_set = set()
        for idx in order:
            if all(j not in sel_set or idx not in incompatible[j] for j in sel_set):
                # Check: idx is compatible with all in selected
                compatible = True
                for j in sel_set:
                    if idx in incompatible[j]:
                        compatible = False
                        break
                if compatible:
                    selected.append(idx)
                    sel_set.add(idx)
        return selected

    # Try several greedy orders
    orders = [
        list(range(N)),  # ascending value (seed-like)
        list(range(N, -1, -1)),  # descending value
        sorted(range(N), key=lambda i: values[i]),  # same as ascending
    ]

    best_sel = []
    best_score = Fraction(0, 1)
    for order in orders:
        sel = greedy(order)
        score = sum(Fraction(1, values[i]) for i in sel)
        if score > best_score:
            best_score = score
            best_sel = sel

    # Local search: try swaps (remove one, add one) and single additions
    def score_of(sel):
        return sum(Fraction(1, values[i]) for i in sel)

    def is_compatible(idx, sel_set):
        for j in sel_set:
            if idx in incompatible[j]:
                return False
        return True

    sel_set = set(best_sel)
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        best_local_score = best_score
        best_local_sel = best_sel[:]

        # Try single additions
        for idx in range(N):
            if idx in sel_set:
                continue
            if is_compatible(idx, sel_set):
                new_sel = best_sel + [idx]
                new_score = best_score + Fraction(1, values[idx])
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_sel = new_sel
                    improved = True

        # Try swaps: remove one, add one
        for i in range(len(best_sel)):
            rem = best_sel[i]
            rem_score = best_score - Fraction(1, values[rem])
            # Try adding each idx not in best_sel
            for idx in range(N):
                if idx in sel_set:
                    continue
                # Check compatibility of idx with best_sel minus rem
                compatible = True
                for j in best_sel:
                    if j == rem:
                        continue
                    if idx in incompatible[j]:
                        compatible = False
                        break
                if compatible:
                    new_score = rem_score + Fraction(1, values[idx])
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_sel = [x for x in best_sel if x != rem] + [idx]
                        improved = True

        if improved:
            best_sel = best_local_sel
            best_score = best_local_score
            sel_set = set(best_sel)

    # Final polish: try more aggressive swaps if time permits
    if time.time() < DEADLINE - 5:
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            best_local_score = best_score
            best_local_sel = best_sel[:]
            sel_set = set(best_sel)

            # Try 2-for-1 swaps: remove two, add one
            for i in range(len(best_sel)):
                for j in range(i + 1, len(best_sel)):
                    rem1, rem2 = best_sel[i], best_sel[j]
                    rem_score = best_score - Fraction(1, values[rem1]) - Fraction(1, values[rem2])
                    for idx in range(N):
                        if idx in sel_set:
                            continue
                        compatible = True
                        for k in best_sel:
                            if k == rem1 or k == rem2:
                                continue
                            if idx in incompatible[k]:
                                compatible = False
                                break
                        if compatible:
                            new_score = rem_score + Fraction(1, values[idx])
                            if new_score > best_local_score:
                                best_local_score = new_score
                                best_local_sel = [x for x in best_sel if x != rem1 and x != rem2] + [idx]
                                improved = True

            if improved:
                best_sel = best_local_sel
                best_score = best_local_score

    return [values[i] for i in best_sel]
# EVOLVE-BLOCK-END
