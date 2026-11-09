"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

DEADLINE = 80.0

def _can_add(a, S):
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % a == 0:
                return False
            if (a * c) % b == 0:
                return False
            if (a * b) % c == 0:
                return False
    return True

def _greedy_construct(order, base_set):
    S = list(base_set)
    S_set = set(S)
    for a in order:
        if a in S_set:
            continue
        if _can_add(a, S):
            S.append(a)
            S_set.add(a)
    return S

def _local_search(S, deadline):
    S = list(S)
    S_set = set(S)
    all_nums = list(range(2, 257))
    
    while time.time() < deadline:
        improved = False
        
        for a in all_nums:
            if time.time() >= deadline:
                break
            if a in S_set:
                continue
            if _can_add(a, S):
                S.append(a)
                S_set.add(a)
                improved = True
        
        if time.time() >= deadline:
            break
        
        for i in range(len(S)):
            if time.time() >= deadline:
                break
            a = S[i]
            S_without = S[:i] + S[i+1:]
            S_without_set = S_set - {a}
            for b in all_nums:
                if b in S_without_set:
                    continue
                if _can_add(b, S_without):
                    S = S_without + [b]
                    S_set = S_without_set | {b}
                    improved = True
                    break
        
        if not improved:
            break
    
    return S

def construct():
    primes = []
    for n in range(2, 257):
        is_p = True
        for d in range(2, isqrt(n) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
    
    base = list(primes)
    best_S = list(base)
    best_len = len(base)
    
    non_primes = [x for x in range(2, 257) if x not in set(primes)]
    
    orderings = [
        non_primes,
        sorted(non_primes, reverse=True),
        sorted(non_primes, key=lambda x: x.bit_count()),
        sorted(non_primes, key=lambda x: -x.bit_count()),
    ]
    
    for order in orderings:
        if time.time() >= DEADLINE:
            break
        candidate = _greedy_construct(order, base)
        if len(candidate) > best_len:
            best_S = candidate
            best_len = len(candidate)
    
    if time.time() < DEADLINE - 5:
        best_S = _local_search(best_S, DEADLINE - 5)
    
    result = list(set(int(x) for x in best_S))
    return result
# EVOLVE-BLOCK-END
