import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Try size 1: x = n * m^2
    for m in range(1, 200):
        x = n * m * m
        if x > max_val:
            break
        if x > n:
            prod = n * x
            if math.isqrt(prod) ** 2 == prod:
                return [int(x)]

    # Try size 2: n * x * y = square => x * y = n * k^2
    best_pair = None
    for k in range(1, 300):
        target = n * k * k
        if target > max_val * max_val:
            break
        for x in range(n + 1, min(target // (n + 1) + 1, max_val + 1)):
            if target % x != 0:
                continue
            y = target // x
            if y <= n or y > max_val or y == x:
                continue
            if n < x <= max_val and n < y <= max_val:
                mx = max(x, y)
                if best_pair is None or mx < best_pair[0]:
                    best_pair = (mx, [x, y])
        if best_pair and best_pair[0] - n <= 28:
            break

    if best_pair:
        return [int(x) for x in best_pair[1]]

    # Linear algebra approach
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    def get_mask(x):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << prime_index[p])
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    target_mask = get_mask(n)
    basis = {}

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = get_mask(M)
        h = 1 << (M - 211)

        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                basis[p] = (temp_v, temp_h)
                break

        t_v = target_mask
        t_h = 0
        while t_v:
            p = t_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                t_v ^= b_vec
                t_h ^= b_hist
            else:
                break

        if t_v == 0:
            sol = []
            hh = t_h
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol.append(211 + idx)
                hh ^= lsb

            if len(sol) <= 100:
                prod = n
                for x in sol:
                    prod *= x
                if math.isqrt(prod) ** 2 == prod:
                    return [int(x) for x in sol]

    # Fallback: size 1 solution
    return [840]
