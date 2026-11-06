"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add(frozenset([a, b, c]))
    
    # Precompute: for each x, the list of bad triples containing x
    bad_triples_by_elem = {x: [] for x in range(1, N + 1)}
    for bt in bad_triples:
        for x in bt:
            bad_triples_by_elem[x].append(bt)
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for bt in bad_triples_by_elem[x]:
            other = [y for y in bt if y != x]
            if other[0] in S and other[1] in S:
                return False
        return True
    
    best = set()
    best_size = 0
    
    # Multiple randomized greedy passes with local search
    num_starts = 500
    for start in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Randomize the order
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Local search: try to improve
        if time.time() > deadline - 5:
            break
        
        # Try add-one moves until no more can be added
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
        
        # Try 1-out 1-in swap moves
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            if not S:
                break
            # Try removing a random element and adding a different one
            candidates_to_remove = list(S)
            random.shuffle(candidates_to_remove)
            for to_remove in candidates_to_remove:
                S.remove(to_remove)
                # Try to find a better element to add
                best_add = None
                for x in elements:
                    if x != to_remove and x not in S and can_add(x, S):
                        # Prefer adding elements that allow more future additions
                        S.add(x)
                        # Check if this leads to a larger set (try to fill)
                        S_copy = set(S)
                        fill_improved = True
                        while fill_improved:
                            fill_improved = False
                            for y in elements:
                                if y not in S_copy and can_add(y, S_copy):
                                    S_copy.add(y)
                                    fill_improved = True
                        if len(S_copy) > len(S):
                            best_add = x
                            S = S_copy
                            break
                        else:
                            S.remove(x)
                if best_add is None:
                    S.add(to_remove)  # restore
                # If we found an improvement, continue; otherwise break
                if best_add is not None:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 2:
        for x in elements:
            if x not in best and can_add(x, best):
                best.add(x)
    
    if not best:
        best = {1}
    
    return list(best)
# EVOLVE-BLOCK-END
