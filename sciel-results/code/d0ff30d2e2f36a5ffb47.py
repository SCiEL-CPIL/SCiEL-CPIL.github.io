import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140

    L = 10**11

    # Generate powerful numbers up to L
    powerfuls = set()
    max_b = int(round(L ** (1.0/3.0))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)

    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}

    powerfuls = sorted(powerfuls)
    P = len(powerfuls)

    def count_pairs(N):
        i = 0
        j = P - 1
        c = 0
        while i <= j:
            s = powerfuls[i] + powerfuls[j]
            if s == N:
                c += 1
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        return c

    # Candidate N from sums of the first K powerful numbers, but cap the number of candidates.
    K = min(P, 600)
    cand = set()
    for i in range(K):
        ai = powerfuls[i]
        for j in range(i, K):
            cand.add(ai + powerfuls[j])
        if len(cand) > 200000:
            break
        if time.time() > deadline - 30:
            break

    best_N = 2
    best_count = 0
    for N in cand:
        if N < 2 or N > 2 * L:
            continue
        c = count_pairs(N)
        if c > best_count:
            best_count = c
            best_N = N
        if time.time() > deadline - 15:
            break

    pairs = []
    i = 0
    j = P - 1
    while i <= j:
        s = powerfuls[i] + powerfuls[j]
        if s == best_N:
            a = powerfuls[i]
            b = powerfuls[j]
            if a <= b:
                pairs.append([a, b])
            i += 1
            j -= 1
        elif s < best_N:
            i += 1
        else:
            j -= 1

    pairs.sort()
    return {'N': best_N, 'pairs': pairs}
