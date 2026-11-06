import math
import time
import random

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _validate(result):
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_disjoint(result[i][0], result[i][1], result[j][0], result[j][1]):
                return False
    return True

def _greedy_order(moduli, order, deadline):
    result = []
    for m in order:
        if time.time() > deadline:
            break
        for r in range(m):
            ok = True
            for (r2, m2) in result:
                if not _is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                result.append([r, m])
                break
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80
    best_result = []

    for p in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        moduli = [m for m in range(2, 129) if m % p == 0]

        # Natural order
        result = _greedy_order(moduli, moduli, deadline)
        if _validate(result) and len(result) > len(best_result):
            best_result = result

        # Random orderings
        for trial in range(100):
            if time.time() > deadline:
                break
            random_order = moduli[:]
            random.shuffle(random_order)
            result = _greedy_order(moduli, random_order, deadline)
            if _validate(result) and len(result) > len(best_result):
                best_result = result

    if not best_result:
        return [[1, 2]]
    return best_result
