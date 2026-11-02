"""
Strategy: Branch-and-Bound for Maximum Independent Set.
The problem is equivalent to finding the largest subset of integer vectors 
in R^11 such that the minimum pairwise distance is at least the maximum norm.
This is an Independent Set problem on a graph where vertices are candidate vectors 
and edges connect incompatible pairs.
We use bitsets (Python integers) for fast neighbor lookups and a recursive 
branch-and-bound algorithm with a strict time limit.
"""
import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Generate Candidate Vectors
    # We include weight 1, 2, 3, and 4 vectors to maximize density.
    # Weight 1: 11 * 2 = 22 vectors
    # Weight 2: C(11,2)*4 = 55*4 = 220 vectors
    # Weight 3: C(11,3)*8 = 165*8 = 1320 vectors
    # Weight 4: C(11,4)*16 = 330*16 = 5280 vectors
    # Total candidates ~ 6842. This size is manageable for bitset B&B.
    
    candidates = []
    for w in [1, 2, 3, 4]:
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(np.array(v, dtype=np.int64))
    
    n_cand = len(candidates)
    
    # Fallback in case generation fails or is empty (should not happen)
    if n_cand == 0:
         return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    # 2. Precompute Conflict Matrix
    # Two vectors u, v are compatible if ||u-v||^2 >= max(||u||^2, ||v||^2)
    # Let N_u = ||u||^2, N_v = ||v||^2.
    # ||u-v||^2 = N_u + N_v - 2<u,v>
    # Condition: N_u + N_v - 2<u,v> >= max(N_u, N_v)
    # If N_u >= N_v: N_u + N_v - 2<u,v> >= N_u  =>  N_v >= 2<u,v>
    # If N_v > N_u:  N_u + N_v - 2<u,v> >= N_v  =>  N_u >= 2<u,v>
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    dots = candidates @ candidates.T
    
    # Vectorized conflict calculation
    # min_norm = min(N_u, N_v), max_norm = max(N_u, N_v)
    # Condition: min_norm >= 2 * dot
    # Note: If dot is negative, it's always true since norms are positive.
    # If dot is positive, we check.
    
    min_norms = np.minimum(norms_sq[:, np.newaxis], norms_sq[np.newaxis, :])
    
    # Conflict if min_norms < 2 * dots
    conflict_bool = min_norms < (2 * dots)
    
    # Diagonal is not a conflict (vector is compatible with itself in the set context, 
    # but we pick distinct indices. In graph terms, no self-loops).
    np.fill_diagonal(conflict_bool, False)
    
    # 3. Convert to Bitsets
    # adj[i] is a bitset of neighbors of i (j where conflict_bool[i,j] is True)
    # We use Python integers for bitsets.
    # N ~ 6842 bits. Python handles arbitrary precision ints efficiently.
    
    adj = [0] * n_cand
    for i in range(n_cand):
        # Get indices where conflict is True
        neighbors = np.nonzero(conflict_bool[i])[0]
        mask = 0
        for j in neighbors:
            mask |= (1 << j)
        adj[i] = mask
        
    # 4. Branch and Bound
    best_count = 0
    best_set = []
    
    # Initial lower bound: Greedy
    # Pick random permutation, greedy select
    # But for B&B, we need a good initial bound to prune.
    # Let's do a quick greedy to get a baseline.
    current_greedy_set = []
    available = (1 << n_cand) - 1
    # Simple greedy: pick node with fewest neighbors? Or just first available?
    # Let's just pick the first available in index order for a quick baseline
    # Actually, random greedy is better.
    import random
    random.seed(42)
    perm = list(range(n_cand))
    random.shuffle(perm)
    temp_available = (1 << n_cand) - 1
    temp_set = []
    for i in perm:
        if (temp_available >> i) & 1:
            temp_set.append(i)
            # Remove i and its neighbors from available
            temp_available &= ~(1 << i)
            temp_available &= ~adj[i]
            if time.time() > deadline - 10: break
    if len(temp_set) > best_count:
        best_count = len(temp_set)
        best_set = temp_set.copy()

    # Recursive B&B
    # State: P (bitset of candidate vertices), count (current set size)
    # We add vertices to the set.
    # Pruning: if count + popcount(P) <= best_count, return
    
    def popcount(x):
        return x.bit_count()
        
    def branch_and_bound(P, current_set, count):
        nonlocal best_count, best_set
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if count > best_count:
                best_count = count
                best_set = current_set.copy()
            return
            
        # Upper bound: count + number of candidates left
        # If even taking all remaining doesn't beat best, prune
        if count + popcount(P) <= best_count:
            return
            
        # Heuristic: Order candidates in P by degree (fewest neighbors first)
        # To speed up, we can just pick the first set bit, or the one with min degree in P.
        # Finding min degree in P requires iterating P bits.
        # Let's pick a vertex v from P. 
        # A common heuristic is to pick v with minimum degree in the induced subgraph P.
        # For simplicity and speed, let's pick the lowest index in P, 
        # or better, the one with fewest neighbors in P.
        
        # Extract bits of P
        # P is an int.
        # Let's find a vertex v in P.
        # We can iterate through bits of P.
        
        # Optimization: If P is sparse, this is fast.
        # Let's pick the vertex v in P that has the minimum number of neighbors in P.
        # deg_P(v) = popcount(adj[v] & P)
        
        # To avoid iterating all bits of P for every node (expensive if P is dense),
        # we can use a simpler heuristic: pick the first bit.
        # But min-degree is much better for pruning.
        # Let's try min-degree.
        
        # Iterate bits of P
        p_temp = P
        best_v = -1
        min_deg = n_cand + 1
        
        # Optimization: Only check a subset? No, let's check all if P is small.
        # If P is large, this loop is slow.
        # Alternative: Just pick the lowest set bit.
        # Let's stick to lowest set bit for speed, but maybe sort P initially?
        # Actually, let's use a hybrid: if popcount(P) < 100, find min deg. Else, pick lowest.
        
        if popcount(P) < 200:
            while p_temp:
                lsb = p_temp & (-p_temp)
                v = lsb.bit_length() - 1
                p_temp ^= lsb
                deg = popcount(adj[v] & P)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
        else:
            # Pick lowest set bit
            lsb = P & (-P)
            best_v = lsb.bit_length() - 1

        if best_v == -1:
            return

        # Branch 1: Include best_v
        # New P = P \ ( {best_v} U N(best_v) )
        new_P_include = P & ~( (1 << best_v) | adj[best_v] )
        current_set.append(best_v)
        branch_and_bound(new_P_include, current_set, count + 1)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_P_exclude = P & ~(1 << best_v)
        branch_and_bound(new_P_exclude, current_set, count)

    # Start B&B
    # Initial P is all candidates
    P_init = (1 << n_cand) - 1
    branch_and_bound(P_init, [], 0)
    
    if time.time() > deadline:
         # Return best found so far
         if best_set:
             return candidates[best_set]
         else:
             return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    if best_set:
        return candidates[best_set]
    else:
        # Should not happen, but fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

# EVOLVE-BLOCK-END
