import time
import math
import bisect
from collections import defaultdict

MAX_N = 10_000_000
DEADLINE_FRACTION = 0.90

def _build_spf(n):
    spf = list(range(n + 1))
    if n >= 1:
        spf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def construct():
    t_start = time.time()
    deadline = t_start + 90.0 * DEADLINE_FRACTION
    
    # Build SPF
    spf = _build_spf(MAX_N)
    
    # Compute Sigmas
    sigmas = [0] * (MAX_N + 1)
    sigmas[1] = 1
    for n in range(2, MAX_N + 1):
        x = n
        s = 1
        while x > 1:
            p = spf[x]
            term = 1
            power = 1
            while x % p == 0:
                x //= p
                power *= p
                term += power
            s *= term
        sigmas[n] = s
        
    # Group by Sigma
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigmas[n]].append(n)
        
    # Search
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for b in range(1, MAX_N + 1):
        if time.time() > deadline:
            break
            
        s = sigmas[b]
        lst = groups[s]
        if len(lst) < 2:
            continue
            
        target = 1.5 * b
        # Binary search for target in lst
        idx = bisect.bisect_left(lst, target)
        
        # Check idx and idx-1
        if idx < len(lst):
            a = lst[idx]
            if 2 * a != 3 * b: # Avoid exact 3/2
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
        if idx > 0:
            a = lst[idx - 1]
            if 2 * a != 3 * b:
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
