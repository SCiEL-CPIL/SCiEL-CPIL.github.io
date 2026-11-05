import time
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    L_nums = list(range(2, 129))
    H_nums = list(range(129, 257))
    nL = len(L_nums)
    nH = len(H_nums)

    # Precompute conflicts
    conflictL = [0] * nL
    for i in range(nL):
        a = L_nums[i]
        bits = 0
        for j in range(nL):
            if i == j:
                continue
            b = L_nums[j]
            if a * b // gcd(a, b) <= 256:
                bits |= (1 << j)
        conflictL[i] = bits

    C = [0] * nL
    for i in range(nL):
        a = L_nums[i]
        bits = 0
        for j in range(nH):
            h = H_nums[j]
            if a * h // gcd(a, h) <= 256:
                bits |= (1 << j)
        C[i] = bits

    wL = [1.0 / x for x in L_nums]
    wH = [1.0 / x for x in H_nums]
    allH = (1 << nH) - 1

    def compute_H(L_bits):
        blocked = 0
        x = L_bits
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            blocked |= C[i]
            x ^= lsb
        return allH ^ blocked

    def objective(L_bits):
        s = 0.0
        x = L_bits
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            s += wL[i]
            x ^= lsb
        H_bits = compute_H(L_bits)
        x = H_bits
        while x:
            lsb = x & (-x)
            j = lsb.bit_length() - 1
            s += wH[j]
            x ^= lsb
        return s

    def local_search(L_bits, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-15
            best_new_L = None
            H_bits = compute_H(L_bits)

            # 1-add
            for i in range(nL):
                if L_bits & (1 << i):
                    continue
                if L_bits & conflictL[i]:
                    continue
                lost = 0.0
                x = C[i] & H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    lost += wH[j]
                    x ^= lsb
                gain = wL[i] - lost
                if gain > best_gain:
                    best_gain = gain
                    best_new_L = L_bits | (1 << i)

            # 1-remove
            for i in range(nL):
                if not (L_bits & (1 << i)):
                    continue
                L_new = L_bits ^ (1 << i)
                H_new = compute_H(L_new)
                gained = 0.0
                x = H_new & ~H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    gained += wH[j]
                    x ^= lsb
                gain = gained - wL[i]
                if gain > best_gain:
                    best_gain = gain
                    best_new_L = L_new

            # 2-swap
            for i in range(nL):
                if not (L_bits & (1 << i)):
                    continue
                L_minus_i = L_bits ^ (1 << i)
                for j in range(nL):
                    if L_bits & (1 << j):
                        continue
                    if L_minus_i & conflictL[j]:
                        continue
                    L_new = L_minus_i | (1 << j)
                    H_new = compute_H(L_new)
                    gained = 0.0
                    x = H_new & ~H_bits
                    while x:
                        lsb = x & (-x)
                        k = lsb.bit_length() - 1
                        gained += wH[k]
                        x ^= lsb
                    lost = 0.0
                    x = H_bits & ~H_new
                    while x:
                        lsb = x & (-x)
                        k = lsb.bit_length() - 1
                        lost += wH[k]
                        x ^= lsb
                    gain = (wL[j] - wL[i]) + gained - lost
                    if gain > best_gain:
                        best_gain = gain
                        best_new_L = L_new

            if best_new_L is not None:
                L_bits = best_new_L
                w += best_gain
                improved = True

        return L_bits, w

    # Multiple initializations
    best_L = 0
    best_w = objective(best_L)

    # Greedy by weight
    order = sorted(range(nL), key=lambda i: -wL[i])
    L_greedy = 0
    for i in order:
        if not (L_greedy & (1 << i)) and not (L_greedy & conflictL[i]):
            L_greedy |= (1 << i)
    w_greedy = objective(L_greedy)
    L_greedy, w_greedy = local_search(L_greedy, w_greedy, DEADLINE)
    if w_greedy > best_w:
        best_w = w_greedy
        best_L = L_greedy

    # Random restarts
    import random
    rng = random.Random(12345)
    for _ in range(30):
        if time.time() > DEADLINE - 3:
            break
        L_rand = 0
        for i in range(nL):
            if rng.random() < 0.25:
                if not (L_rand & conflictL[i]):
                    L_rand |= (1 << i)
        w_rand = objective(L_rand)
        L_rand, w_rand = local_search(L_rand, w_rand, DEADLINE)
        if w_rand > best_w:
            best_w = w_rand
            best_L = L_rand

    # Final polish: try adding/removing single elements one more time
    H_final = compute_H(best_L)
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        H_bits = compute_H(best_L)
        for i in range(nL):
            if not (best_L & (1 << i)) and not (best_L & conflictL[i]):
                lost = 0.0
                x = C[i] & H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    lost += wH[j]
                    x ^= lsb
                if wL[i] - lost > 1e-15:
                    best_L |= (1 << i)
                    changed = True
                    break
            elif best_L & (1 << i):
                L_new = best_L ^ (1 << i)
                H_new = compute_H(L_new)
                gained = 0.0
                x = H_new & ~H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    gained += wH[j]
                    x ^= lsb
                if gained - wL[i] > 1e-15:
                    best_L = L_new
                    changed = True
                    break

    # Build result
    H_final = compute_H(best_L)
    result = []
    x = best_L
    while x:
        lsb = x & (-x)
        i = lsb.bit_length() - 1
        result.append(L_nums[i])
        x ^= lsb
    x = H_final
    while x:
        lsb = x & (-x)
        j = lsb.bit_length() - 1
        result.append(H_nums[j])
        x ^= lsb

    return result
