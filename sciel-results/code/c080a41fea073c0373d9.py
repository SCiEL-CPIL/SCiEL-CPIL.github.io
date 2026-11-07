import math
import time

def construct():
    t0 = time.time()
    deadline = t0 + 50.0
    LO = 210
    MAXV = 10210
    MAX_K = 100

    # Sieve primes up to MAXV
    limit = MAXV
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Compute squarefree parity vector for 210
    def parity_vec(n):
        v = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt ^= 1
                if cnt:
                    v |= (1 << i)
        if temp > 1:
            idx = prime_to_idx.get(temp)
            if idx is not None:
                v |= (1 << idx)
        return v

    target = parity_vec(210)

    # Precompute parity vectors for all x in (210, MAXV]
    vecs = [0] * (MAXV + 1)
    for x in range(211, MAXV + 1):
        vecs[x] = parity_vec(x)

    # Incremental GF(2) basis with solution tracking
    # basis: list of (pivot_bit, vector, solution_set)
    basis = []
    pivot_map = {}  # pivot_bit -> index in basis

    for Y in range(211, MAXV + 1):
        if time.time() > deadline:
            break

        v = vecs[Y]
        sol = {Y}

        # Reduce v by existing basis
        while v:
            hsb = v.bit_length() - 1
            if hsb in pivot_map:
                idx = pivot_map[hsb]
                b_vec, b_sol = basis[idx]
                v ^= b_vec
                # XOR solution sets
                new_sol = set(sol)
                for s in b_sol:
                    if s in new_sol:
                        new_sol.discard(s)
                    else:
                        new_sol.add(s)
                sol = new_sol
            else:
                break

        if v == 0:
            # Y is in span; check if target is in span
            v_t = target
            sol_t = set()
            while v_t:
                hsb = v_t.bit_length() - 1
                if hsb in pivot_map:
                    idx = pivot_map[hsb]
                    b_vec, b_sol = basis[idx]
                    v_t ^= b_vec
                    new_sol = set(sol_t)
                    for s in b_sol:
                        if s in new_sol:
                            new_sol.discard(s)
                        else:
                            new_sol.add(s)
                    sol_t = new_sol
                else:
                    break
            if v_t == 0 and len(sol_t) <= MAX_K and len(sol_t) >= 1:
                # Verify
                prod = 210
                for x in sol_t:
                    prod *= x
                r = math.isqrt(prod)
                if r * r == prod:
                    return [int(x) for x in sol_t]
        else:
            # Add to basis
            hsb = v.bit_length() - 1
            pivot_map[hsb] = len(basis)
            basis.append((v, sol))

    # Fallback: try greedy approach for safety
    # Simple fallback
    return [840]
