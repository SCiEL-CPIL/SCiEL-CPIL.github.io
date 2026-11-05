"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _is_compatible_with_set(x, S):
    """Check if x can be added to set S (all pairs have lcm <= N)."""
    for y in S:
        if _lcm(x, y) > N:
            return False
    return True

def _greedy_add(S, candidates):
    """Try to add each candidate to S in order; return updated S."""
    S = set(S)
    for x in candidates:
        if x not in S and _is_compatible_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, start_time, deadline):
    """Local search: try to swap out elements for better ones to grow the set."""
    S = set(S)
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        current_size = len(S)
        S_list = sorted(S)
        
        # Try removing one element and adding a better one
        for i, elem in enumerate(S_list):
            if (time.time() - start_time) > deadline:
                break
            S_without = S - {elem}
            # Try to find an element not in S that fits with S_without
            for x in range(1, N + 1):
                if x in S:
                    continue
                if _is_compatible_with_set(x, S_without):
                    new_S = S_without | {x}
                    if len(new_S) > current_size:
                        S = new_S
                        improved = True
                        break
            if improved:
                break
    return S

def _try_divisors_of(m, start_time, deadline):
    """Get divisors of m, then greedily add compatible elements."""
    divs = []
    for i in range(1, int(m**0.5) + 1):
        if m % i == 0:
            divs.append(i)
            if i != m // i:
                divs.append(m // i)
    divs = [d for d in divs if d <= N]
    S = set(divs)
    # Greedy add
    for x in range(1, N + 1):
        if (time.time() - start_time) > deadline:
            break
        if x not in S and _is_compatible_with_set(x, S):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Divisors of highly composite numbers
    # Try various highly composite numbers up to 512
    highly_composite = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 480]
    # Also try numbers with many divisors
    for m in [360, 240, 180, 120, 480, 336, 420, 504]:
        if (time.time() - start_time) > deadline:
            break
        S = _try_divisors_of(m, start_time, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Strategy 2: Start with divisors of 360 (known good seed), then local search
    if (time.time() - start_time) < deadline:
        S360 = _try_divisors_of(360, start_time, deadline)
        if len(S360) > best_size:
            best_size = len(S360)
            best_S = S360
        # Local search from this
        S_local = _local_search(S360, start_time, deadline)
        if len(S_local) > best_size:
            best_size = len(S_local)
            best_S = S_local
    
    # Strategy 3: Greedy from scratch with different orderings
    if (time.time() - start_time) < deadline:
        # Greedy by increasing value
        S_greedy = set()
        for x in range(1, N + 1):
            if (time.time() - start_time) > deadline:
                break
            if x not in S_greedy and _is_compatible_with_set(x, S_greedy):
                S_greedy.add(x)
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_S = S_greedy
        # Local search
        S_local2 = _local_search(S_greedy, start_time, deadline)
        if len(S_local2) > best_size:
            best_size = len(S_local2)
            best_S = S_local2
    
    # Strategy 4: Randomized greedy with multiple restarts
    if (time.time() - start_time) < deadline:
        import random
        random.seed(42)
        for trial in range(50):
            if (time.time() - start_time) > deadline:
                break
            S_rand = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if (time.time() - start_time) > deadline:
                    break
                if x not in S_rand and _is_compatible_with_set(x, S_rand):
                    S_rand.add(x)
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_S = S_rand
            # Local search from random result
            if (time.time() - start_time) < deadline - 5:
                S_rand_local = _local_search(S_rand, start_time, deadline)
                if len(S_rand_local) > best_size:
                    best_size = len(S_rand_local)
                    best_S = S_rand_local
    
    # Final local search from best
    if (time.time() - start_time) < deadline:
        S_final = _local_search(best_S, start_time, deadline)
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final
    
    if not best_S:
        best_S = {1}
    
    return list(best_S)
# EVOLVE-BLOCK-END
