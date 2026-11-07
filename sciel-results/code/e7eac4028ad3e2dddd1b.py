import math
import time

def construct():
    N = 256
    deadline = time.time() + 85
    
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    prime_factors = [set()] * (N + 1)
    for i in range(1, N + 1):
        prime_factors[i] = get_prime_factors(i)
    
    all_primes = [p for p in range(2, N + 1) if prime_factors[p] == {p}]
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    prime_mask = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for p in prime_factors[vi]:
            if p in prime_to_idx:
                mask |= (1 << prime_to_idx[p])
        prime_mask[i] = mask
    
    weights = [i + 1 for i in range(N)]
    order = sorted(range(N), key=lambda i: weights[i], reverse=True)
    new_weights = [weights[order[k]] for k in range(N)]
    new_prime_mask = [prime_mask[order[k]] for k in range(N)]
    
    suffix_sum = [0] * (N + 1)
    for k in range(N - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + new_weights[k]
    
    best_sum = 0
    best_set = set()
    
    def bnb(k, current_sum, used_mask, selected):
        nonlocal best_sum, best_set
        if time.time() > deadline:
            return
        if k == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            return
        ub = current_sum + suffix_sum[k]
        if ub <= best_sum:
            return
        bnb(k + 1, current_sum, used_mask, selected)
        if (new_prime_mask[k] & used_mask) == 0:
            selected.append(k)
            bnb(k + 1, current_sum + new_weights[k], used_mask | new_prime_mask[k], selected)
            selected.pop()
    
    bnb(0, 0, 0, [])
    
    result = [order[k] + 1 for k in best_set]
    if 1 not in result:
        result.append(1)
    result.sort()
    
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return [1] + [a for a in range(2, N + 1) if all(a % d for d in range(2, math.isqrt(a) + 1))]
    
    return [int(x) for x in result]

