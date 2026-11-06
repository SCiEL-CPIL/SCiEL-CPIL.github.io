"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

_N = 128
_DEADLINE = 55.0  # safety margin under 60s kill

# Precompute LCM table for O(1) lookup
_LCM_TABLE = [[0] * (_N + 1) for _ in range(_N + 1)]
for i in range(1, _N + 1):
    for j in range(i, _N + 1):
        l = i // math.gcd(i, j) * j
        _LCM_TABLE[i][j] = l
        _LCM_TABLE[j][i] = l

def _bad_triple_exists(selected, x, lcm_table):
    """Check if adding x creates a bad triple with two existing elements."""
    s = selected
    n = len(s)
    # Optimization: Check pairs in s. If lcm(a,b) == lcm(a,x) == lcm(b,x), it's bad.
    # We can speed this up by checking lcm(a,x) first.
    for i in range(n):
        a = s[i]
        la_x = lcm_table[a][x]
        for j in range(i + 1, n):
            b = s[j]
            # If lcm(a,b) != la_x, it can't be equal to lcm(b,x) which must also be la_x
            if lcm_table[a][b] == la_x:
                if lcm_table[b][x] == la_x:
                    return True
    return False

def _greedy(order, lcm_table):
    selected = []
    for x in order:
        if not _bad_triple_exists(selected, x, lcm_table):
            selected.append(x)
    return selected

def construct():
    t0 = time.time()
    base = list(range(1, _N + 1))
    
    # Deterministic orders
    orders = [
        base,
        list(reversed(base)),
        sorted(base, key=lambda v: -v.bit_count()),
        sorted(base, key=lambda v: v.bit_count()),
        sorted(base, key=lambda v: -math.gcd(v, 128)),
        sorted(base, key=lambda v: v.bit_length()),
    ]
    
    best = []
    best_len = 0
    
    # Run greedy on deterministic orders
    for o in orders:
        if time.time() - t0 > _DEADLINE - 2:
            break
        cand = _greedy(o, _LCM_TABLE)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
    # Randomized Greedy with Restarts (Kick-and-Refill strategy)
    # This is much faster than complex local search and explores wider neighborhood
    rng = random.Random(42)
    # We can run many more random restarts now
    max_restarts = 5000 
    for _ in range(max_restarts):
        if time.time() - t0 > _DEADLINE - 2:
            break
        
        # Random order
        o = base[:]
        rng.shuffle(o)
        
        # Optional: Bias towards numbers with fewer divisors or specific properties?
        # For now, pure random shuffle is robust.
        
        cand = _greedy(o, _LCM_TABLE)
        
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
            # If we found a very good solution, we can stop early or continue to polish?
            # Let's continue to see if we can beat it.
            
            # If we reach 128, we are done (unlikely but possible)
            if best_len == _N:
                break

    # Final Polish: Try to add any remaining elements to the best set
    # This is cheap because we only check once per missing element
    if time.time() - t0 < _DEADLINE - 1:
        best_set = set(best)
        # Sort candidates by some heuristic? Or just iterate 1..N
        # To be safe and fast, just iterate
        for x in range(1, _N + 1):
            if time.time() - t0 > _DEADLINE - 0.5:
                break
            if x not in best_set:
                if not _bad_triple_exists(best, x, _LCM_TABLE):
                    best.append(x)
                    best_set.add(x)
                    best_len += 1

    if not best:
        best = [1]
        
    return best
# EVOLVE-BLOCK-END
