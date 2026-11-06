import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # 10s safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    
    def can_add(triple, current_set):
        a, b, c = triple
        # Check if adding (a,b,c) completes a K(2,2,2)
        # Need a', b', c' all distinct from each other and from a,b,c
        remaining = [x for x in range(n) if x not in (a, b, c)]
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check all 7 other transversals
                    if ((a, b, c_prime) in current_set or (a, c_prime, b) in current_set or (b, a, c_prime) in current_set or (b, c_prime, a) in current_set or (c_prime, a, b) in current_set or (c_prime, b, a) in current_set):
                        pass
                    else:
                        continue
                    # Actually, let me use a helper
                    pass
        
        # Let me redo this more carefully
        def has_triple(p, q, r, s_set):
            # Check if {p,q,r} is in s_set (as a frozenset or sorted tuple)
            t = tuple(sorted((p, q, r)))
            return t in s_set
        
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check 7 other transversals
                    if (has_triple(a, b, c_prime, current_set) and
                        has_triple(a, b_prime, c, current_set) and
                        has_triple(a, b_prime, c_prime, current_set) and
                        has_triple(a_prime, b, c, current_set) and
                        has_triple(a_prime, b, c_prime, current_set) and
                        has_triple(a_prime, b_prime, c, current_set) and
                        has_triple(a_prime, b_prime, c_prime, current_set)):
                        return False
        return True
    
    def greedy_pass(order):
        current = set()
        for t in order:
            if time.time() > DEADLINE:
                break
            if can_add(t, current):
                current.add(t)
        return current
    
    def fill_pass(current):
        """Try to add any missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t in all_triples:
                if time.time() > DEADLINE:
                    return current
                if t not in current and can_add(t, current):
                    current.add(t)
                    improved = True
        return current
    
    best = set()
    
    # Multiple randomized greedy passes
    rng = random.Random(42)
    num_passes = 20
    
    for i in range(num_passes):
        if time.time() > DEADLINE:
            break
        order = list(all_triples)
        rng.shuffle(order)
        result = greedy_pass(order)
        result = fill_pass(result)
        if len(result) > len(best):
            best = result
    
    # Also try a few more structured orderings
    # Sort by some heuristic
    for i in range(5):
        if time.time() > DEADLINE:
            break
        order = sorted(all_triples, key=lambda t: (t[0], t[1], t[2]), reverse=True)
        rng.shuffle(order)
        result = greedy_pass(order)
        result = fill_pass(result)
        if len(result) > len(best):
            best = result
    
    return [list(t) for t in best]
