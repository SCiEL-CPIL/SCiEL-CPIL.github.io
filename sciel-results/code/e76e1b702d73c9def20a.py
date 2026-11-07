import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    MIN_VAL = 211
    
    # Sieve for primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def compute_vec(n):
        vec = 0
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt % 2 == 1:
                    vec |= (1 << prime_to_idx[p])
        if x > 1:
            vec |= (1 << prime_to_idx[x])
        return vec
    
    vec_210 = compute_vec(210)
    
    # basis_map: pivot_idx -> (vector, subset_mask)
    basis_map = {}
    rank = 0
    
    def add_to_basis(vec, num_idx):
        nonlocal rank
        u = vec
        S = 1 << num_idx
        while u != 0:
            h = u.bit_length() - 1
            if h in basis_map:
                b, S_b = basis_map[h]
                u ^= b
                S ^= S_b
            else:
                basis_map[h] = (u, S)
                rank += 1
                return True
        return False
        
    def check_span(target_vec):
        t = target_vec
        S_t = 0
        while t != 0:
            h = t.bit_length() - 1
            if h in basis_map:
                b, S_b = basis_map[h]
                t ^= b
                S_t ^= S_b
            else:
                return False, None
        return True, S_t
        
    for M in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            break
        num_idx = M - MIN_VAL
        vec_M = compute_vec(M)
        added = add_to_basis(vec_M, num_idx)
        if added:
            found, subset_mask = check_span(vec_210)
            if found:
                subset = []
                for i in range(num_idx + 1):
                    if subset_mask & (1 << i):
                        subset.append(MIN_VAL + i)
                if not subset:
                    continue
                prod = 210
                for x in subset:
                    prod *= x
                k = math.isqrt(prod)
                if k * k == prod:
                    return [int(x) for x in subset]
                    
    return [840]
