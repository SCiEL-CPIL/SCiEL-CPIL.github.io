import time
import numpy as np
import itertools

DEADLINE_MARGIN = 20  # seconds before the 150s limit to stop

def construct():
    t0 = time.time()
    best_tournament = None
    best_n = None

    # We'll try a few different prime-sized Paley tournaments and then local search
    # But first, let's just return the known best Paley tournament quickly
    # Actually, let's try to find a smaller one via local search from random starts
    
    # Strategy: 
    # 1. Try small n from 31 upward (Erdos lower bound is 31)
    # 2. For each n, try to construct a valid tournament via local search
    # 3. If we find one, return it
    # 4. If time runs out, return the best found (or the Paley 83)
    
    # For efficiency, we'll use a vectorized approach
    # A tournament is valid if every 4-set has a vertex with out-degree 3 within that set
    # Equivalently, no 4-set is a "bad" 4-set (where no vertex dominates all others)
    
    # A 4-set is bad if it contains no vertex that beats all 3 others.
    # In a 4-vertex tournament, the score sequences are:
    # (3,1,0,0) - transitive, has a source
    # (2,2,1,1) - has two sources? No, let's think...
    # Actually, in any tournament on 4 vertices, there is always a vertex with out-degree >= 2.
    # But we need out-degree 3 (dominates all 3 others).
    # A 4-set has a dominator iff the induced tournament on those 4 vertices is transitive (has a source with out-degree 3).
    # Wait, no. A vertex with out-degree 3 in a 4-vertex tournament means it beats all 3 others.
    # This happens iff the tournament is transitive with that vertex as the source.
    # Actually no, a tournament can have a vertex with out-degree 3 without being fully transitive.
    # E.g., vertex 0 beats 1,2,3; vertex 1 beats 2,3; vertex 2 beats 3. That's transitive.
    # E.g., vertex 0 beats 1,2,3; vertex 1 beats 2; vertex 3 beats 2. 
    # Scores: 0:3, 1:1, 3:1, 2:0. Vertex 0 has out-degree 3. Not fully transitive (1 beats 2, 3 beats 2, but 1 and 3 tie? No, one beats the other).
    # Anyway, the condition is: for every 4-set, at least one vertex has out-degree 3 within that set.
    
    # Equivalent: no 4-set is a "directed 4-cycle" or "transitive with no source of out-degree 3" 
    # Actually, the bad 4-sets are those where the maximum out-degree within the 4-set is < 3.
    # So max out-degree is 0,1,2.
    # Score sequences with max score <= 2: 
    # (2,2,1,1) - two vertices with out-degree 2
    # (2,1,1,0) - one vertex with out-degree 2
    # (1,1,1,1) - not possible in tournament (sum must be 6)
    # (2,2,2,0) - sum=6, possible? Three vertices each with out-degree 2, one with 0.
    #   The one with out-degree 0 loses to all. The other three form a 3-cycle. So yes.
    # (1,1,1,3) - has a vertex with out-degree 3, so it's good.
    # (2,1,1,2) same as (2,2,1,1).
    # So bad 4-sets have score sequences (2,2,1,1) or (2,2,2,0) or (2,1,1,0) or (1,1,1,1)[impossible] etc.
    
    # For local search, we can count the number of bad 4-sets.
    # Flipping an edge (i,j) changes which 4-sets are bad.
    # A 4-set containing both i and j is affected. There are C(n-2,2) such 4-sets.
    # For each such 4-set {i,j,k,l}, we check if it was bad before and is good now, or vice versa.
    
    # Let's implement a local search that minimizes the number of bad 4-sets.
    
    def count_bad(A):
        """Count the number of 4-sets with no dominator. A is n x n adjacency matrix, A[i,j]=1 iff i->j."""
        n = len(A)
        bad = 0
        for quad in itertools.combinations(range(n), 4):
            i, j, k, l = quad
            # Check if any vertex dominates all others in the quad
            # i dominates if A[i,j]=A[i,k]=A[i,l]=1
            if A[i,j] and A[i,k] and A[i,l]:
                continue
            if A[j,i] and A[j,k] and A[j,l]:
                continue
            if A[k,i] and A[k,j] and A[k,l]:
                continue
            if A[l,i] and A[l,j] and A[l,k]:
                continue
            bad += 1
        return bad
    
    def try_local_search(n, max_time=None, seed=None):
        """Try to find a valid tournament on n vertices via local search."""
        if seed is not None:
            np.random.seed(seed)
        
        # Start with random tournament
        A = np.random.randint(0, 2, (n, n))
        np.fill_diagonal(A, 0)
        # Ensure tournament property: A[i,j] + A[j,i] = 1 for i != j
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == A[j,i]:
                    # Flip one
                    if A[i,j] == 1:
                        A[i,j] = 0
                        A[j,i] = 1
                    else:
                        A[i,j] = 1
                        A[j,i] = 0
        
        bad_count = count_bad(A)
        if bad_count == 0:
            return A, bad_count
        
        # Local search: flip edges that reduce bad_count
        no_improve = 0
        max_no_improve = 500
        t_start = time.time()
        
        while True:
            if max_time and time.time() - t_start > max_time:
                break
            if no_improve > max_no_improve:
                break
            
            # Pick a random edge
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            # Compute delta in bad_count if we flip edge (i,j)
            # A 4-set is affected if it contains both i and j.
            # For each pair (k,l) with k,l != i,j, k<l, the 4-set {i,j,k,l} is affected.
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k+1, n):
                    if l == i or l == j:
                        continue
                    # Quad: {i,j,k,l}
                    # Current: check if bad
                    # We need to check all 4 vertices for domination in the current and flipped tournament.
                    
                    # Current state
                    def is_bad_quad(a, b, c, d, A):
                        # Check if any of a,b,c,d dominates all others
                        if A[a,b] and A[a,c] and A[a,d]: return False
                        if A[b,a] and A[b,c] and A[b,d]: return False
                        if A[c,a] and A[c,b] and A[c,d]: return False
                        if A[d,a] and A[d,b] and A[d,c]: return False
                        return True
                    
                    cur_bad = is_bad_quad(i, j, k, l, A)
                    
                    # Flipped: A[i,j] and A[j,i] are swapped
                    # We need to create a modified view or just check manually
                    # Let's just check manually for efficiency
                    # After flip: A[i,j] becomes 1-A[i,j], A[j,i] becomes 1-A[j,i]
                    
                    # Check if i dominates in flipped
                    i_dominates_flip = (1 - A[i,j]) and A[i,k] and A[i,l]
                    j_dominates_flip = (1 - A[j,i]) and A[j,k] and A[j,l]
                    k_dominates_flip = A[k,i] and A[k,j] and A[k,l]
                    l_dominates_flip = A[l,i] and A[l,j] and A[l,k]
                    
                    flip_bad = not (i_dominates_flip or j_dominates_flip or k_dominates_flip or l_dominates_flip)
                    
                    if cur_bad and not flip_bad:
                        delta -= 1
                    elif not cur_bad and flip_bad:
                        delta += 1
            
            if delta < 0:
                # Accept flip
                A[i,j] = 1 - A[i,j]
                A[j,i] = 1 - A[j,i]
                bad_count += delta
                no_improve = 0
            else:
                no_improve += 1
                # Occasionally accept neutral or worsening moves (simulated annealing lite)
                if np.random.rand() < 0.05:
                    A[i,j] = 1 - A[i,j]
                    A[j,i] = 1 - A[j,i]
                    bad_count += delta
        
        return A, bad_count
    
    # Try small n values
    for n in range(31, 50):
        # Time budget per n
        remaining = 150 - (time.time() - t0)
        if remaining < DEADLINE_MARGIN:
            break
        per_n_time = min(5, remaining * 0.5)
        
        A, bad = try_local_search(n, max_time=per_n_time, seed=42)
        if bad == 0:
            # Convert to orientation list
            orientation = []
            for i in range(n):
                for j in range(i+1, n):
                    orientation.append(int(A[i,j]))
            return {'n': n, 'orientation': orientation}
    
    # If we didn't find a small one, return the Paley tournament on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
