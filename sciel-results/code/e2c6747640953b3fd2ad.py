import math
import time

def construct():
    limit = 10210
    start_time = time.time()
    deadline = start_time + 55.0

    # Step 1: Smallest prime factor sieve
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Step 2: Precompute parity bitmasks
    bitmasks = {}
    for n in range(211, limit + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                x //= p
                count += 1
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        bitmasks[n] = mask

    # Step 3: Target bitmask for 210
    target_mask = 0
    x = 210
    while x > 1:
        p = spf[x]
        count = 0
        while x % p == 0:
            x //= p
            count += 1
        if count % 2 == 1:
            target_mask |= (1 << prime_to_idx[p])

    # Step 4: Gaussian Elimination
    basis = {}  # pivot -> (vec, comb)
    pivots = [] # sorted list of pivot indices (descending)
    
    t_u = target_mask
    t_c = 0

    for M in range(211, limit + 1):
        if time.time() > deadline:
            break

        v = bitmasks[M]
        c = 1 << (M - 211)

        # Reduce v against existing basis
        # Iterate through pivots in descending order
        for p in pivots:
            if p >= v.bit_length():
                continue
            if (v >> p) & 1:
                b_vec, b_comb = basis[p]
                v ^= b_vec
                c ^= b_comb

        if v != 0:
            pivot = v.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (v, c)
                # Insert pivot into sorted list
                # pivots is sorted descending
                # Find insertion point
                # Since pivots are unique, we can use bisect on negated values or manual
                # Manual insertion is O(N) but N is small (~1000)
                inserted = False
                for i in range(len(pivots)):
                    if pivots[i] < pivot:
                        pivots.insert(i, pivot)
                        inserted = True
                        break
                if not inserted:
                    pivots.append(pivot)

                # Update persistent reduced target
                if (t_u >> pivot) & 1:
                    t_u ^= v
                    t_c ^= c

        if t_u == 0:
            # Extract selected elements
            selected = []
            # t_c is a bitmask where bit i corresponds to element 211 + i
            # We only care about bits up to M-211
            for i in range(M - 211 + 1):
                if (t_c >> i) & 1:
                    selected.append(211 + i)
            
            if 1 <= len(selected) <= 100:
                # Verify
                prod = 210
                for x in selected:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(x) for x in selected]

    return [840]
