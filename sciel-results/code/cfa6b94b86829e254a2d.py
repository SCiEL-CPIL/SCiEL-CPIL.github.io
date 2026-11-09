"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, lcm
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_conflicts():
    """
    Precompute for each pair (a,b) the set of c values such that
    lcm(a,b) == lcm(a,c) == lcm(b,c).
    Store as: conflicts[a][b] = frozenset of c's
    Also store: for each element x, the set of pairs (a,b) already in the solution
    that would form a bad triple with x.
    """
    N = 128
    # conflicts[x] = set of (a,b) pairs such that {a,b,x} is a bad triple
    # Actually, let's think differently.
    # For a given x, which pairs (a,b) with a,b < x (or a,b in some set) form a bad triple with x?
    # lcm(a,b) == lcm(a,x) == lcm(b,x)
    
    # Let's precompute: for each pair (a,b), what's lcm(a,b)?
    # And for each x, which (a,b) pairs satisfy lcm(a,b)==lcm(a,x)==lcm(b,x)?
    
    # More efficient: for each x, find all a,b such that the triple is bad.
    # Precompute a 2D table: bad[a][b] = set of x's that complete a bad triple with a,b
    # Or: for each x, precompute which pairs (a,b) are incompatible with x.
    
    # Let's use: conflict_mask[x] = bitmask of elements y such that there exists z in current set
    # where {x,y,z} is a bad triple. But this depends on current set.
    
    # Better approach: Precompute for each pair (a,b) the set of c such that {a,b,c} is bad.
    # Then when building a set, for each new element x, check if any pair (a,b) in current set
    # has x in bad_pairs[a][b].
    
    N = 128
    # bad_pairs[a][b] = frozenset of c values (c != a, c != b) such that {a,b,c} is a bad triple
    bad_pairs = [[frozenset() for _ in range(N+1)] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = _lcm(a, b)
            for c in range(1, N+1):
                if c == a or c == b:
                    continue
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bad_pairs[a][b].add(c)
                    bad_pairs[b][a].add(c)
    
    return bad_pairs

def _greedy_with_order(order, bad_pairs):
    """Greedy: add elements in given order if no conflict."""
    selected = []
    selected_set = set()
    for x in order:
        # Check: is there any pair (a,b) in selected such that x in bad_pairs[a][b]?
        conflict = False
        for a in selected:
            for b in selected:
                if a >= b:
                    continue
                if x in bad_pairs[a][b]:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, bad_pairs, max_iters=5000, time_limit=None, start_time=None):
    """
    Local search: try to add elements not in selected, or swap.
    Try adding each excluded element if it doesn't create a conflict.
    """
    if time_limit is None:
        time_limit = 50
    if start_time is None:
        start_time = time.time()
    
    selected = list(selected)
    selected_set = set(selected)
    excluded = [x for x in range(1, 129) if x not in selected_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iters += 1
        # Try adding each excluded element
        for i, x in enumerate(excluded):
            if time.time() - start_time > time_limit:
                break
            # Check if x can be added
            conflict = False
            for a in selected:
                for b in selected:
                    if a >= b:
                        continue
                    if x in bad_pairs[a][b]:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                selected.append(x)
                selected_set.add(x)
                excluded.pop(i)
                improved = True
                break  # restart scan
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    bad_pairs = _precompute_conflicts()
    
    if time.time() > deadline:
        # Fallback: just return powers of 2 and odd numbers
        return [x for x in range(1, 129) if x & (x-1) == 0 or x % 2 == 1]
    
    best = []
    
    # Strategy 1: Natural order greedy
    order1 = list(range(1, 129))
    result1 = _greedy_with_order(order1, bad_pairs)
    result1 = _local_search(result1, bad_pairs, max_iters=2000, time_limit=10, start_time=time.time())
    best = result1
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Reverse order
    order2 = list(range(128, 0, -1))
    result2 = _greedy_with_order(order2, bad_pairs)
    result2 = _local_search(result2, bad_pairs, max_iters=2000, time_limit=10, start_time=time.time())
    if len(result2) > len(best):
        best = result2
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Prime-priority order (primes first, then composites)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(1, 129) if not is_prime(x) and x > 1]
    ones = [1]
    order3 = primes + composites + ones
    result3 = _greedy_with_order(order3, bad_pairs)
    result3 = _local_search(result3, bad_pairs, max_iters=2000, time_limit=10, start_time=time.time())
    if len(result3) > len(best):
        best = result3
    
    if time.time() > deadline:
        return best
    
    # Strategy 4: Random restarts with local search
    rng = random.Random(42)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        order = list(range(1, 129))
        rng.shuffle(order)
        result = _greedy_with_order(order, bad_pairs)
        result = _local_search(result, bad_pairs, max_iters=1000, time_limit=5, start_time=time.time())
        if len(result) > len(best):
            best = result
    
    if time.time() > deadline:
        return best
    
    # Final: try to add more elements to best
    best = _local_search(best, bad_pairs, max_iters=5000, time_limit=15, start_time=time.time())
    
    return best

# EVOLVE-BLOCK-END
