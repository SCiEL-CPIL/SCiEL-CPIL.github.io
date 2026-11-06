import time
import math
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 148.0
    
    # Generate primes up to 10^6
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0:
            is_prime[0] = 0
        if n >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(10**6)
    
    # Generate powerful numbers up to limit
    def generate_powerful(limit):
        pw = set([1])
        pw_list = [1]
        for p in primes:
            p2 = p * p
            if p2 > limit:
                break
            new_entries = []
            for x in pw_list:
                if x * p2 > limit:
                    break
                y = x * p2
                while y <= limit:
                    if y not in pw:
                        pw.add(y)
                        new_entries.append(y)
                    y *= p
            pw_list.extend(new_entries)
            pw_list.sort()
        return sorted(pw)
    
    # Use limit 10^8 for manageable pair counting
    limit_pw = 10**8
    pw_sorted = generate_powerful(limit_pw)
    pw_set = set(pw_sorted)
    n_pw = len(pw_sorted)
    
    # Count pairs (a, b) with a <= b, a + b = N
    pair_counts = defaultdict(int)
    for i in range(n_pw):
        a = pw_sorted[i]
        for j in range(i, n_pw):
            b = pw_sorted[j]
            N = a + b
            if N > 10**18:
                break
            pair_counts[N] += 1
    
    if not pair_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(pair_counts, key=pair_counts.get)
    
    # Find all pairs for best_N
    pairs_list = []
    for a in pw_sorted:
        b = best_N - a
        if b < a:
            break
        if b in pw_set:
            pairs_list.append([a, b])
    
    if time.time() > deadline:
        return {'N': best_N, 'pairs': pairs_list[:200]}
    
    return {'N': best_N, 'pairs': pairs_list}
