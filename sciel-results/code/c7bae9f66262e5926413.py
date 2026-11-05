import math
import time
import itertools
from typing import Dict, List, Tuple

def construct() -> Dict:
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Property: Every set of 4 vertices must have a common dominator.
    # In a tournament, a common dominator for {u, v, w, x} is a vertex y
    # such that y->u, y->v, y->w, and y->x.
    # Note: The problem statement says "Every set of 4 vertices must have a vertex with arcs to all four".
    # This implies the dominator is *outside* the set of 4? 
    # "Every set of 4 vertices must have a vertex with arcs to all four."
    # Usually, this phrasing in tournament theory (Schütte's problem) refers to the property that 
    # for any 4 vertices, there is *another* vertex dominating them.
    # Let's verify the interpretation.
    # If the dominator could be *inside* the set, it would say "Every set of 4 vertices contains a vertex..."
    # "Have a vertex with arcs to all four" implies the vertex is distinct from the four? 
    # Actually, if y is in {u,v,w,x}, it cannot have an arc to itself. 
    # So the dominator MUST be outside the set of 4.
    # So for every subset S of V with |S|=4, there exists y in V\S such that y->s for all s in S.

    # Paley tournaments exist for primes p = 4k+3.
    # We want to minimize n. So we should check small primes first.
    # Smallest prime p=4k+3 is 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 83.
    
    # Erdős lower bound is 31. So n must be at least 31.
    # Let's check primes p >= 31 that are 4k+3.
    # 31 (7*4+3)
    # 43 (10*4+3)
    # 47 (11*4+3)
    # 59 (14*4+3)
    # 67 (16*4+3)
    # 71 (17*4+3)
    # 83 (20*4+3)
    
    # The seed program uses 83. It might be that smaller primes don't satisfy the property.
    # We will iterate through candidate primes in increasing order.
    # For each prime, we construct the Paley tournament and check the property.
    # If it satisfies, we return it.
    # If none satisfy within time, we return the smallest one we found or a fallback.
    
    # However, constructing and checking the property for every 4-subset is O(n^4).
    # For n=31, 31^4 ~ 923k. For n=83, 83^4 ~ 47M. This might be slow in Python if not optimized.
    # We can optimize the check.
    
    # Let's precompute the "out-neighborhood" bitmasks for each vertex.
    # For a set of 4 vertices {u,v,w,x}, we need a vertex y such that 
    # u in out(y) AND v in out(y) AND w in out(y) AND x in out(y).
    # This is equivalent to: out(y) & (1<<u | 1<<v | 1<<w | 1<<x) == (1<<u | 1<<v | 1<<w | 1<<x).
    # Alternatively, let mask_4 = 1<<u | 1<<v | 1<<w | 1<<x.
    # We need to check if there exists y such that (out(y) & mask_4) == mask_4.
    # This is faster than iterating 4 subsets explicitly if we use bit operations.
    
    # Even better: For each vertex y, let S_y be the set of vertices it dominates.
    # The condition "y dominates S" means S is a subset of S_y.
    # So for every 4-subset S, there must be some y such that S \subseteq S_y.
    # This is equivalent to saying that the family of sets {S_y | y in V} covers all 4-subsets.
    
    # Algorithm for checking a tournament:
    # 1. Build out_mask[i] for each vertex i.
    # 2. Iterate over all 4-subsets (u,v,w,x).
    # 3. Compute mask = (1<<u) | (1<<v) | (1<<w) | (1<<x).
    # 4. Check if any out_mask[y] is a superset of mask.
    #    i.e., (out_mask[y] & mask) == mask.
    #    We can precompute a lookup or just iterate y.
    #    Iterating y for each 4-subset is O(n * C(n,4)).
    #    For n=31: 31 * 31495 ~ 1M ops. Very fast.
    #    For n=83: 83 * 1,581,582 ~ 131M ops. Might take a few seconds in Python.
    #    We can optimize by breaking early if found.
    
    def is_valid_paley(p):
        # Construct Paley tournament on p vertices
        # i -> j iff (j-i) % p is a quadratic residue mod p
        # p is prime 4k+3
        squares = set()
        for x in range(1, p):
            squares.add((x*x) % p)
        
        # Build in-neighborhood masks: in_mask[v] has bit u set if u -> v
        in_mask = [0] * p
        for v in range(p):
            mask = 0
            for u in range(p):
                if u != v and (v - u) % p in squares:
                    mask |= (1 << u)
            in_mask[v] = mask
        
        # Check: for every 4-subset {u,v,w,x}, intersection of in_mask[u], in_mask[v], in_mask[w], in_mask[x] is non-empty
        for u in range(p):
            for v in range(u+1, p):
                inter1 = in_mask[u] & in_mask[v]
                if inter1 == 0:
                    continue
                for w in range(v+1, p):
                    inter2 = inter1 & in_mask[w]
                    if inter2 == 0:
                        continue
                    for x in range(w+1, p):
                        if (inter2 & in_mask[x]) == 0:
                            return False
        return True

    # List of candidate primes p = 4k+3
    # We start from the smallest possible n.
    # Lower bound is 31.
    # Primes 4k+3: 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 83
    # We skip p < 5 because n>=5.
    # We skip p < 31 because of lower bound? 
    # The task says "Erdős's lower bound gives f(4) >= 31". 
    # So we can safely start checking from p=31.
    
    candidates = [31, 43, 47, 59, 67, 71, 83]
    
    best_result = None
    
    for p in candidates:
        if time.time() > deadline - 10.0:
            break
            
        if is_valid_paley(p):
            # Construct orientation for return
            squares = set()
            for x in range(1, p):
                squares.add((x*x) % p)
            
            orientation = []
            for i in range(p):
                for j in range(i+1, p):
                    if (j - i) % p in squares:
                        orientation.append(0) # i -> j
                    else:
                        orientation.append(1) # j -> i
                        
            return {'n': p, 'orientation': orientation}
            
    # Fallback: Return the seed program's result (83) if no smaller one found or valid
    # Just in case the check failed due to time or logic error, return a known valid one.
    # The seed program implies 83 is valid.
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
