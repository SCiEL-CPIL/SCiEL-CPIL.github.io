"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    N = 256
    nodes = list(range(2, N + 1))
    node_index = {x: i for i, x in enumerate(nodes)}
    n = len(nodes)

    # Precompute lcm matrix
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcm_table[i][j] = lcm_table[j][i] = lcm(nodes[i], nodes[j])

    # Precompute reciprocal as Fraction for exact arithmetic
    recip = [Fraction(1, nodes[i]) for i in range(n)]

    def valid_set(indices):
        s = set(indices)
        for i in s:
            for j in s:
                if i != j and lcm_table[i][j] <= 256:
                    return False
        return True

    def score(indices):
        total = Fraction(0, 1)
        for i in indices:
            total += recip[i]
        return total

    # Greedy initial solution: sort by 1/x descending, add if compatible
    order = sorted(range(n), key=lambda i: recip[i], reverse=True)
    selected = []
    for i in order:
        if all(lcm_table[i][j] > 256 for j in selected):
            selected.append(i)

    best_set = set(selected)
    best_score = score(best_set)

    # Local search with 1-for-1, 1-for-2, 2-for-2 swaps
    random.seed(42)

    def try_swaps(current_set, current_score):
        nonlocal best_set, best_score
        improved = True
        while improved:
            improved = False
            # 1-for-1 swap: remove a, add b
            for a in list(current_set):
                for b in range(n):
                    if b in current_set:
                        continue
                    # Check b compatible with all in current_set \ {a}
                    ok = True
                    for c in current_set:
                        if c != a and lcm_table[b][c] <= 256:
                            ok = False
                            break
                    if ok:
                        new_score = current_score - recip[a] + recip[b]
                        if new_score > current_score:
                            current_set.discard(a)
                            current_set.add(b)
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
            # 1-for-2 swap: remove a, add b1, b2
            for a in list(current_set):
                for b1 in range(n):
                    if b1 in current_set or b1 == a:
                        continue
                    for b2 in range(b1 + 1, n):
                        if b2 in current_set or b2 == a:
                            continue
                        # Check b1 and b2 compatible with all in current_set \ {a} and with each other
                        ok = lcm_table[b1][b2] > 256
                        if not ok:
                            continue
                        for c in current_set:
                            if c == a:
                                continue
                            if lcm_table[b1][c] <= 256 or lcm_table[b2][c] <= 256:
                                ok = False
                                break
                        if ok:
                            new_score = current_score - recip[a] + recip[b1] + recip[b2]
                            if new_score > current_score:
                                current_set.discard(a)
                                current_set.add(b1)
                                current_set.add(b2)
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_set = set(current_set)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # 2-for-2 swap: remove a1, a2, add b1, b2
            if not improved and time.time() < deadline - 5:
                for a1 in list(current_set):
                    for a2 in list(current_set):
                        if a1 >= a2:
                            continue
                        for b1 in range(n):
                            if b1 in current_set:
                                continue
                            for b2 in range(b1 + 1, n):
                                if b2 in current_set:
                                    continue
                                # Check b1, b2 compatible with all in current_set \ {a1, a2}
                                ok = lcm_table[b1][b2] > 256
                                if not ok:
                                    continue
                                for c in current_set:
                                    if c == a1 or c == a2:
                                        continue
                                    if lcm_table[b1][c] <= 256 or lcm_table[b2][c] <= 256:
                                        ok = False
                                        break
                                if ok:
                                    new_score = current_score - recip[a1] - recip[a2] + recip[b1] + recip[b2]
                                    if new_score > current_score:
                                        current_set.discard(a1)
                                        current_set.discard(a2)
                                        current_set.add(b1)
                                        current_set.add(b2)
                                        current_score = new_score
                                        improved = True
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_set = set(current_set)
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return current_set, current_score

    current_set, current_score = try_swaps(set(best_set), best_score)

    # Try a few random restarts
    for restart in range(5):
        if time.time() >= deadline - 10:
            break
        random.shuffle(order)
        selected = []
        for i in order:
            if all(lcm_table[i][j] > 256 for j in selected):
                selected.append(i)
        current_set, current_score = try_swaps(set(selected), score(set(selected)))

    # Return the best set as list of values
    result = [nodes[i] for i in best_set]
    if not result:
        result = [nodes[0]]
    return result
# EVOLVE-BLOCK-END
