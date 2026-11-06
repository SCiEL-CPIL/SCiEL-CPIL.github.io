"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 192
    MAX_SIZE = 48
    
    # Precompute all triples (a, b, c) of distinct integers in [1, N]
    # and for each, the set of divisors of (a+b+c) that are also in [1, N]
    # We'll represent conflicts as: for each element x, a set of triples (a,b,c)
    # such that x divides a+b+c and x not in {a,b,c}
    
    # Actually, let's think about this differently for efficiency.
    # For a given set S, x in S violates if there exist a,b,c in S\{x} with x | (a+b+c)
    
    # Precompute: for each x in [1,N], store a set of (a,b,c) triples where 
    # a<b<c, a,b,c != x, and x divides (a+b+c)
    # But that's too many. Instead, let's use a bitset approach.
    
    # Alternative: precompute for each pair (a,b) the set of x that divide (a+b+c) 
    # for various c. That's complex.
    
    # Let's use a different approach: 
    # For each x, precompute which pairs (a,b) with a<b and a,b != x 
    # have the property that there exists c > b, c != x, such that x | (a+b+c)
    # This is still complex.
    
    # Simplest efficient approach: 
    # For a candidate set S, check validity by:
    # For each x in S, for each triple (a,b,c) in S\{x}, check x | (a+b+c)
    # With |S| <= 48, this is at most 48 * C(47,3) = 48 * 16215 = ~778k checks per validity test
    # That's too slow for many iterations.
    
    # Better: precompute for each x, a lookup structure.
    # For each x in [1,N], precompute a 2D boolean array or use a different encoding.
    
    # Let's try: for each x, store a set of sums s such that x | s.
    # Then for a triple (a,b,c), compute s = a+b+c and check if s % x == 0.
    # The check is just one modulo operation. The issue is the number of triples.
    
    # For local search with swaps, we can be smarter.
    
    # Let's do a greedy construction first, then local search.
    
    def is_valid_set(S):
        """Check if set S is valid. S is a list of distinct integers."""
        S_set = set(S)
        for x in S:
            others = [v for v in S if v != x]
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True
    
    def can_add(S, x):
        """Check if adding x to S keeps it valid. S is a list, x not in S."""
        S_set = set(S)
        if x in S_set:
            return False
        # Check x: does any triple in S sum to multiple of x?
        for a, b, c in combinations(S, 3):
            if (a + b + c) % x == 0:
                return False
        # Check each y in S: does any triple in S\{y} + {x} sum to multiple of y?
        # The new triples involving x are: (x, a, b) for a, b in S\{y}, a != b
        for y in S:
            for a, b in combinations(S, 2):
                if a == y or b == y:
                    continue
                # Triple (x, a, b)
                if (x + a + b) % y == 0:
                    return False
        return True
    
    # Greedy: try to build a large set
    # Start with large numbers (they're less likely to divide sums of others)
    S = []
    for x in range(N, 0, -1):
        if len(S) >= MAX_SIZE:
            break
        if can_add(S, x):
            S.append(x)
    
    best = list(S)
    best_size = len(S)
    
    # Local search: try swaps
    # Try to replace an element with another to improve size
    # Actually, we want to maximize cardinality, so we try to add elements.
    # If we can't add, try swapping to free up space.
    
    # Let's try a different local search: 
    # Try all pairs (remove x, add y) where y not in S, x in S
    # If the result is valid and same size, it might allow future additions.
    
    # Actually, let's try to add elements first.
    for x in range(1, N+1):
        if x in S:
            continue
        if can_add(S, x):
            S.append(x)
            best = list(S)
            best_size = len(S)
            if len(S) >= MAX_SIZE:
                break
        if time.time() > deadline - 5:
            break
    
    # Now try local search with swaps to potentially allow more additions
    if time.time() < deadline - 10:
        # Try to find a valid swap that allows adding another element
        S_set = set(S)
        for x in range(1, N+1):
            if x in S_set:
                continue
            # Try removing each y in S and adding x
            for y in list(S):
                if time.time() > deadline - 5:
                    break
                S.remove(y)
                if can_add(S, x):
                    S.append(x)
                    S_set.discard(y)
                    S_set.add(x)
                    # Now try to add more
                    for z in range(1, N+1):
                        if z in S_set:
                            continue
                        if can_add(S, z):
                            S.append(z)
                            S_set.add(z)
                    if len(S) > best_size:
                        best = list(S)
                        best_size = len(S)
                    if len(S) >= MAX_SIZE:
                        break
                else:
                    S.append(y)
                    S_set.add(y)
                if time.time() > deadline - 5:
                    break
    
    # Return the best set found
    return best[:MAX_SIZE]
# EVOLVE-BLOCK-END
