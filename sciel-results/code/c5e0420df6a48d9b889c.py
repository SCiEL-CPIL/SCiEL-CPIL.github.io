"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL = list(range(1, N + 1))

def _build_conflict_sets():
    """
    For each element x, compute conflict[x] = set of y such that there exists
    some z (z != x, z != y) with {x,y,z} a bad triple.
    Actually we need: for adding x to set S, x is invalid iff there exist a,b in S
    with gcd(x,a)==gcd(x,b)==gcd(a,b).
    
    We precompute for each pair (a,b) the set of x that would form a bad triple:
    bad_with_pair[a][b] = set of x such that gcd(x,a)==gcd(x,b)==gcd(a,b)
    """
    # conflict[x] = set of y where x and y are "conflicting" (there exists z making bad triple)
    # But more useful: for a set S, x is addable iff for all pairs (a,b) in S, x not in bad_with_pair[a][b]
    
    # Precompute gcd table
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    
    # For each pair (a,b), compute which x make a bad triple
    # bad_with_pair[a][b] = {x : x != a, x != b, gcd(x,a) == gcd(x,b) == gcd(a,b)}
    bad_with_pair = [[set() for _ in range(N+1)] for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = g[a][b]
            for x in range(1, N+1):
                if x == a or x == b:
                    continue
                if g[x][a] == d and g[x][b] == d:
                    bad_with_pair[a][b].add(x)
                    bad_with_pair[b][a].add(x)
    
    return g, bad_with_pair

def _can_add_fast(x, S_list, g, bad_with_pair):
    """Check if x can be added to S_list (list of selected elements)."""
    k = len(S_list)
    for i in range(k):
        a = S_list[i]
        ga = g[x][a]
        for j in range(i+1, k):
            b = S_list[j]
            if g[x][b] == ga and g[a][b] == ga:
                return False
    return True

def _greedy(order, g, bad_with_pair, deadline):
    S = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(x, S, g, bad_with_pair):
            S.append(x)
    return S

def _local_search(S, g, bad_with_pair, deadline, max_rounds=100):
    """1-for-0 (add), 1-for-1 (swap), 2-for-1 (remove one, add two) moves."""
    S_set = set(S)
    best = list(S)
    best_size = len(best)
    missing = set(ALL) - S_set
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding
        for x in list(missing):
            if _can_add_fast(x, best, g, bad_with_pair):
                best.append(x)
                best_size += 1
                missing.discard(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swap
        swapped = False
        for out in list(best):
            for x in missing:
                if time.time() > deadline:
                    break
                trial = [v for v in best if v != out]
                if _can_add_fast(x, trial, g, bad_with_pair):
                    best = trial + [x]
                    best_size = len(best)
                    missing.discard(x)
                    missing.add(out)
                    swapped = True
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try 2-for-1: remove one, try to add two
        found = False
        for out in list(best):
            if time.time() > deadline:
                break
            trial = [v for v in best if v != out]
            added = []
            for x in missing:
                if x == out:
                    continue
                if _can_add_fast(x, trial, g, bad_with_pair):
                    trial.append(x)
                    added.append(x)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                best = trial
                best_size = len(best)
                missing.discard(added[0])
                missing.discard(added[1])
                missing.add(out)
                found = True
                improved = True
                break
        if improved:
            continue
        
        break
    
    return best

def _simulated_annealing(S, g, bad_with_pair, deadline, max_iters=500):
    """SA with 1-for-1 and 2-for-1 moves, accepting worse solutions with probability."""
    best = list(S)
    best_size = len(best)
    S_set = set(best)
    missing = set(ALL) - S_set
    current_size = best_size
    
    T = 1.0
    cool = 0.995
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random move
        r = random.random()
        if r < 0.4:
            # 1-for-1 swap
            if not best or not missing:
                continue
            out = random.choice(best)
            x = random.choice(list(missing))
            if x == out:
                continue
            trial = [v for v in best if v != out]
            if _can_add_fast(x, trial, g, bad_with_pair):
                # Accept (same size)
                best = trial + [x]
                S_set = set(best)
                missing = set(ALL) - S_set
                current_size = len(best)
        elif r < 0.7:
            # 2-for-1: remove one, add two
            if len(best) < 3 or len(missing) < 2:
                continue
            out = random.choice(best)
            trial = [v for v in best if v != out]
            added = []
            candidates = [x for x in missing if x != out]
            random.shuffle(candidates)
            for x in candidates:
                if _can_add_fast(x, trial, g, bad_with_pair):
                    trial.append(x)
                    added.append(x)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                best = trial
                S_set = set(best)
                missing = set(ALL) - S_set
                current_size = len(best)
                if current_size > best_size:
                    best_size = current_size
        else:
            # Try to add (1-for-0)
            if missing:
                x = random.choice(list(missing))
                if _can_add_fast(x, best, g, bad_with_pair):
                    best.append(x)
                    S_set = set(best)
                    missing = set(ALL) - S_set
                    current_size = len(best)
                    if current_size > best_size:
                        best_size = current_size
        
        T *= cool
    
    return best

def _is_valid(S, g):
    S = list(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if g[a][b] == g[b][c] == g[a][c]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Precompute
    g, bad_with_pair = _build_conflict_sets()
    
    if time.time() > deadline:
        return [1]
    
    best_set = []
    best_size = 0
    
    # Multiple greedy orderings
    orderings = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        sorted(range(1, N+1), key=lambda x: (x.bit_count(), x)),
        sorted(range(1, N+1), key=lambda x: (-x.bit_count(), x)),
        # Primes first
        sorted(range(1, N+1), key=lambda x: (0 if _is_prime(x) else 1, x)),
        # Random orderings
    ]
    
    for _ in range(10):
        rnd = list(range(1, N+1))
        random.shuffle(rnd)
        orderings.append(rnd)
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy(order, g, bad_with_pair, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search on best
    if time.time() < deadline:
        improved = _local_search(best_set, g, bad_with_pair, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Simulated annealing from best
    if time.time() < deadline:
        sa_result = _simulated_annealing(best_set, g, bad_with_pair, deadline, max_iters=800)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result
    
    # More local search
    if time.time() < deadline:
        improved = _local_search(best_set, g, bad_with_pair, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Additional SA restarts from different starting points
    for _ in range(3):
        if time.time() > deadline:
            break
        # Start from a random greedy
        rnd = list(range(1, N+1))
        random.shuffle(rnd)
        S = _greedy(rnd, g, bad_with_pair, deadline)
        if len(S) < best_size:
            # Pad with best elements? No, just use as-is for SA
            pass
        sa_result = _simulated_annealing(S, g, bad_with_pair, deadline, max_iters=400)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result
    
    # Final local search
    if time.time() < deadline:
        improved = _local_search(best_set, g, bad_with_pair, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Verify
    if not _is_valid(best_set, g):
        # Fallback
        S = []
        for x in range(1, N+1):
            if _can_add_fast(x, S, g, bad_with_pair):
                S.append(x)
        best_set = S
    
    return best_set

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
# EVOLVE-BLOCK-END
