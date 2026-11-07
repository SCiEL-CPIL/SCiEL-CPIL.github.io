import time
import random
from math import isqrt

N_MAX = 256

def _get_primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _is_valid(S):
    """Check if set S is valid."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            for k in range(j + 1, n):
                c = s_list[k]
                # Check a | b*c
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to S."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            # Check x | a*b
            if (a * b) % x == 0:
                return False
            # Check a | x*b
            if (x * b) % a == 0:
                return False
            # Check b | x*a
            if (x * a) % b == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = N_MAX
    primes = _get_primes(n)
    S = set(primes)
    
    # Greedy Add
    for x in range(2, n + 1):
        if time.time() > deadline:
            break
        if x not in S:
            if _can_add(S, x):
                S.add(x)
                
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        added = False
        for x in range(2, n + 1):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    added = True
        if added:
            continue
        
        # Try 1-swap
        s_list = list(S)
        c_list = [x for x in range(2, n + 1) if x not in S]
        
        if not s_list or not c_list:
            break
            
        random.shuffle(s_list)
        random.shuffle(c_list)
        
        swap_found = False
        for y in s_list:
            if swap_found or time.time() > deadline:
                break
            for x in c_list:
                if swap_found or time.time() > deadline:
                    break
                S_trial = S - {y}
                if _can_add(S_trial, x):
                    S = S_trial | {x}
                    improved = True
                    swap_found = True
                    break
        
        if improved:
            continue
            
        # Try 2-swap (limited)
        # Only try a few random pairs to save time
        if len(s_list) >= 2 and len(c_list) >= 2:
            # Pick random pairs
            for _ in range(20):
                if time.time() > deadline:
                    break
                y1_idx = random.randint(0, len(s_list) - 1)
                y2_idx = random.randint(0, len(s_list) - 1)
                if y1_idx == y2_idx:
                    continue
                y1 = s_list[y1_idx]
                y2 = s_list[y2_idx]
                
                x1_idx = random.randint(0, len(c_list) - 1)
                x2_idx = random.randint(0, len(c_list) - 1)
                if x1_idx == x2_idx:
                    continue
                x1 = c_list[x1_idx]
                x2 = c_list[x2_idx]
                
                S_trial = S - {y1, y2}
                # Check if x1 can be added
                if _can_add(S_trial, x1):
                    S_trial.add(x1)
                    # Check if x2 can be added
                    if _can_add(S_trial, x2):
                        S = S_trial | {x2}
                        improved = True
                        break

    return [int(x) for x in S]
