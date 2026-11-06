import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # W[i] = L // (i+1), so sum(c[i]/(i+1)) = (1/L) * sum(c[i]*W[i])
    # We want to minimize |sum(c[i]*W[i])| subject to it being nonzero
    W = [L // (i + 1) for i in range(n)]

    best_N = None
    best_coeffs = None
    best_abs_N = float('inf')

    def calc_N(coeffs):
        s = 0
        for i in range(n):
            c = coeffs[i]
            if c != 0:
                s += c * W[i]
        return s

    # Greedy initialization: start with all zeros, add terms one by one
    # choosing the coefficient that minimizes |N|
    def greedy_init():
        coeffs = [0] * n
        # Process in order of largest weight first for better greedy choices
        order = sorted(range(n), key=lambda i: -W[i])
        for i in order:
            best_c = 0
            best_abs = abs(calc_N(coeffs))
            for c in [-1, 0, 1]:
                if c == 0:
                    continue
                coeffs[i] = c
                val = abs(calc_N(coeffs))
                if val != 0 and val < best_abs:
                    best_abs = val
                    best_c = c
                coeffs[i] = 0
            if best_c != 0:
                coeffs[i] = best_c
        return coeffs

    # Local search: try all single-coordinate changes
    def local_search(coeffs, max_rounds=100):
        current = coeffs[:]
        current_N = calc_N(current)
        if current_N == 0:
            return None, None
        current_abs = abs(current_N)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            for i in range(n):
                if time.time() > deadline:
                    return current, calc_N(current)
                old = current[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    current[i] = new_val
                    new_N = calc_N(current)
                    if new_N != 0:
                        new_abs = abs(new_N)
                        if new_abs < current_abs:
                            current_abs = new_abs
                            improved = True
                            break
                    current[i] = old
                current[i] = old
        return current, calc_N(current)

    # Multi-coordinate local search: try pairs
    def pair_local_search(coeffs, max_rounds=50):
        current = coeffs[:]
        current_N = calc_N(current)
        if current_N == 0:
            return None, None
        current_abs = abs(current_N)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    old_i = current[i]
                    old_j = current[j]
                    for ci in [-1, 0, 1]:
                        if ci == old_i:
                            continue
                        for cj in [-1, 0, 1]:
                            if cj == old_j:
                                continue
                            current[i] = ci
                            current[j] = cj
                            new_N = calc_N(current)
                            if new_N != 0:
                                new_abs = abs(new_N)
                                if new_abs < current_abs:
                                    current_abs = new_abs
                                    improved = True
                                    current[i] = old_i
                                    current[j] = old_j
                                    break
                        else:
                            continue
                        break
                    current[i] = old_i
                    current[j] = old_j
        return current, calc_N(current)

    # Try greedy initialization
    if time.time() < deadline:
        g_coeffs = greedy_init()
        g_N = calc_N(g_coeffs)
        if g_N != 0:
            best_N = g_N
            best_coeffs = g_coeffs[:]
            best_abs_N = abs(g_N)

    # Random restarts with local search
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1

        # Random initialization
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        val = calc_N(coeffs)
        if val == 0:
            continue

        # Single-coordinate local search
        result_coeffs, result_N = local_search(coeffs, max_rounds=30)
        if result_N is not None and result_N != 0:
            if abs(result_N) < best_abs_N:
                best_abs_N = abs(result_N)
                best_N = result_N
                best_coeffs = result_coeffs[:]

        # Pair local search on the best found
        if time.time() < deadline:
            pc, pN = pair_local_search(best_coeffs, max_rounds=20)
            if pN is not None and pN != 0:
                if abs(pN) < best_abs_N:
                    best_abs_N = abs(pN)
                    best_N = pN
                    best_coeffs = pc[:]

        # Occasionally try greedy
        if num_restarts % 10 == 0 and time.time() < deadline:
            gc = greedy_init()
            gN = calc_N(gc)
            if gN != 0:
                rc, rN = local_search(gc, max_rounds=50)
                if rN is not None and rN != 0:
                    if abs(rN) < best_abs_N:
                        best_abs_N = abs(rN)
                        best_N = rN
                        best_coeffs = rc[:]

        if best_abs_N == 1:
            break  # Can't do better than 1

    if best_coeffs is None:
        # Fallback
        fallback = [0] * 28 + [1, -1]
        return fallback

    return best_coeffs
