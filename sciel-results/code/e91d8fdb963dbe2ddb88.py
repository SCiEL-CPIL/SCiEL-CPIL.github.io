"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}
_DEADLINE = 78

def _is_valid_set(S_list):
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if adding x to set S preserves validity. x not in S."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
            if (x * b) % a == 0:
                return False
            if (x * a) % b == 0:
                return False
    return True

def _sieve_primes(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

def construct():
    t0 = time.time()
    n = P['n']
    primes = _sieve_primes(n)
    S = set(primes)
    all_nums = list(range(2, n + 1))

    # Add prime powers greedily
    for p in primes:
        pk = p * p
        while pk <= n:
            if pk not in S and _can_add(S, pk):
                S.add(pk)
            pk *= p

    deadline_1f1 = t0 + _DEADLINE - 20
    deadline_2f2 = t0 + _DEADLINE - 3

    # 1-for-1 local search
    while time.time() < deadline_1f1:
        improved = False
        for rem in list(S):
            if time.time() >= deadline_1f1:
                break
            S_wo = S - {rem}
            for add in all_nums:
                if add in S_wo:
                    continue
                if _can_add(S_wo, add):
                    S = S_wo | {add}
                    improved = True
                    break
        if not improved:
            break

    # 2-for-2 local search (limited iterations)
    while time.time() < deadline_2f2:
        improved = False
        S_list = list(S)
        for i in range(len(S_list)):
            if time.time() >= deadline_2f2:
                break
            for j in range(i + 1, len(S_list)):
                if time.time() >= deadline_2f2:
                    break
                S_wo = S - {S_list[i], S_list[j]}
                for add1 in all_nums:
                    if time.time() >= deadline_2f2:
                        break
                    if add1 in S_wo:
                        continue
                    if not _can_add(S_wo, add1):
                        continue
                    S_temp = S_wo | {add1}
                    for add2 in all_nums:
                        if add2 in S_temp:
                            continue
                        if _can_add(S_temp, add2):
                            S = S_temp | {add2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    result = sorted(int(x) for x in S)

    if not _is_valid_set(result):
        result = primes

    return result
# EVOLVE-BLOCK-END
