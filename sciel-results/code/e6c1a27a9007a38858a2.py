import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate all possible supports of size 4 in R^11
    # There are C(11, 4) = 330 supports.
    all_supports = list(combinations(range(D), 4))
    n_supports = len(all_supports)
    
    # Map support tuple to index for fast lookup
    support_to_idx = {s: i for i, s in enumerate(all_supports)}
    
    # 2. Precompute support intersections to determine compatibility
    # Two vectors with supports S1 and S2 are compatible if:
    # Let k = |S1 ∩ S2|.
    # If k=0, dot product is 0. Distance squared = 4+4-0=8 >= 4. OK.
    # If k=1, dot product is ±1. Distance squared = 4+4-2(±1) = 6 or 10. Both >= 4. OK.
    # If k=2, dot product is -2, 0, or 2.
    #   If dot=2, dist^2 = 4. OK (boundary).
    #   If dot=0, dist^2 = 8. OK.
    #   If dot=-2, dist^2 = 12. OK.
    # If k=3, dot product is -3, -1, 1, 3.
    #   If dot=3, dist^2 = 4+4-6=2 < 4. FAIL.
    #   If dot=1, dist^2 = 4+4-2=6 >= 4. OK.
    #   If dot=-1, dist^2 = 8. OK.
    #   If dot=-3, dist^2 = 10. OK.
    # If k=4, supports are identical.
    #   Vectors are u, v from {±1}^4.
    #   dot can be 4, 2, 0, -2, -4.
    #   If dot=4, dist^2=0. FAIL (same vector).
    #   If dot=2, dist^2=4. OK.
    #   If dot=0, dist^2=8. OK.
    #   If dot=-2, dist^2=12. OK.
    #   If dot=-4, dist^2=16. OK.
    
    # So, the critical constraints are:
    # 1. Avoid identical vectors (same support, same signs).
    # 2. If intersection size k=3, avoid dot product = 3 (all signs match on intersection).
    # 3. If intersection size k=4 (same support), avoid dot product = 4 (identical).
    
    # Actually, for k=3, if dot=3, it means the 3 common indices have the same sign in both vectors.
    # The 4th index in S1 and 4th in S2 are different.
    # Let S1 = {a,b,c,d}, S2 = {a,b,c,e}.
    # u = (s1, s2, s3, s4, 0...), v = (t1, t2, t3, 0, t5...).
    # dot = s1t1 + s2t2 + s3t3.
    # We need s1t1 + s2t2 + s3t3 != 3.
    # Since s_i, t_i are ±1, sum is 3 only if all three are 1.
    # So we just need to avoid the case where the signs on the common support are identical.
    
    # Strategy:
    # We will select a set of supports first, then assign signs.
    # However, the sign assignment depends on the specific pair.
    # A simpler, robust approach for high density:
    # Use the known structure of the Kissing Number in D=11.
    # The kissing number k(11) is 504? No, k(11) is not known exactly?
    # Wait, k(8)=240, k(12)=2560. k(11) is likely between 240 and 2560?
    # Actually, the lower bound for k(11) is often constructed via the D_11 lattice or similar.
    # The D_n lattice kissing number is 2n(n-1). For n=11, 2*11*10 = 220.
    # The current program gets ~220+ vectors.
    # Can we do better?
    # The A_n lattice?
    
    # Let's try to maximize the number of vectors from the m=4 shell.
    # Total vectors in m=4 shell: 330 * 16 = 5280.
    # We want a subset where pairwise constraints hold.
    
    # Let's use a Backtracking CSP on the supports.
    # If we pick two supports S1, S2 with |S1 ∩ S2| = 3, we must ensure there exist signs such that dot != 3.
    # This is always possible unless the signs are forced.
    # Actually, if we pick supports, we can always choose signs to satisfy the constraint for any pair, 
    # EXCEPT if we have a triangle of supports S1, S2, S3 where the sign constraints conflict.
    # But for k=3, the constraint is "not all signs match". This is a local constraint on the pair.
    # If we just pick a set of supports such that no two supports are identical (trivial) 
    # and we handle the k=3 case by sign choice, we can likely pack many.
    
    # However, if we pick ALL 330 supports, can we assign signs?
    # Let's check the density.
    # If we take all 330 supports, we have 330 * 16 = 5280 vectors.
    # But we can't take all vectors of a support if they conflict with others.
    
    # Better Strategy: Greedy + Local Search on the full vector set?
    # No, 5280 vectors is small enough for exact search if structured.
    
    # Let's try a different known construction:
    # The vectors of the form (±1, ±1, ±1, ±1, 0, ..., 0) with sum of squares 4.
    # Consider the graph where nodes are these 5280 vectors.
    # Edge if incompatible (dist < 2).
    # We want Max Independent Set.
    # The current program does this on the *support* level (330 nodes), which is an approximation.
    # The approximation assumes that if supports are compatible, signs can be fixed.
    # But signs interact.
    
    # Let's implement a smarter search:
    # 1. Generate all 5280 vectors.
    # 2. Compute incompatibility graph.
    # 3. Use a greedy heuristic with random restarts to find a large independent set.
    # 4. Improve with local search (swap nodes).
    
    # Given the 120s limit, exact MIS on 5280 nodes is likely too slow.
    # But 330 supports is fast.
    # The current program gets ~220.
    # Let's see if we can get more by using the m=2 shell as well?
    # Current program adds m=2 vectors if compatible.
    
    # Alternative: Use the known optimal packing for D=11?
    # The kissing number k(11) is at least 504? 
    # Actually, the lower bound for k(11) is 504 (from D_11 lattice + some extras? No, D_11 is 220).
    # Wait, the D_n lattice has 2n(n-1) vectors. For n=11, 220.
    # The A_n lattice?
    
    # Let's try to construct a set of size > 220.
    # If we take the 220 vectors of D_11, can we add more?
    # The D_11 vectors are those with even number of -1s in the standard basis? No.
    # D_n is generated by e_i - e_j and (1,1,...,1).
    # The minimal vectors of D_n are e_i - e_j (2n(n-1) of them).
    # For n=11, 220 vectors.
    # Are there more?
    # The kissing number of D_11 is 220.
    # But we are not restricted to D_11. We can use any integer vectors.
    # The m=4 shell vectors are NOT in D_11 (they have 4 non-zeros, sum of coords is even? 
    # If 4 entries are ±1, sum is even. So they ARE in the lattice of even sum? 
    # D_n is the lattice of integer vectors with even sum of coordinates.
    # Yes, m=4 vectors with ±1 have even sum (4 terms of ±1).
    # So the m=4 shell is a subset of D_11.
    # But D_11 minimal vectors are only e_i - e_j (weight 2).
    # The m=4 vectors have norm 2, while e_i-e_j have norm sqrt(2).
    # The task requires normalizing to norm 2.
    # So e_i - e_j (norm sqrt(2)) must be scaled? No, "integer vectors... normalizing each vector to norm 2".
    # So we can use any integer vector, scale it to norm 2.
    # If we use e_i - e_j, norm is sqrt(2). Scaling to norm 2 means multiplying by sqrt(2).
    # Result is sqrt(2)(e_i - e_j). This is NOT an integer vector.
    # The task says "Construct nonzero integer vectors... Normalizing each vector to norm 2 certifies...".
    # This implies the *original* vectors are integers, and we check the condition on the *normalized* ones?
    # "minimum pairwise distance is at least the maximum vector norm".
    # Let u, v be integer vectors.
    # Condition: ||u - v|| >= max(||u||, ||v||).
    # If we normalize to norm 2, let u' = 2u/||u||, v' = 2v/||v||.
    # Distance ||u' - v'|| = 2 ||u - v|| / (||u|| ||v||) * something?
    # Actually, if all vectors have the same norm R, then ||u' - v'|| = 2/R ||u-v||.
    # Condition ||u-v|| >= R becomes 2/R ||u-v|| >= 2 => ||u-v|| >= R.
    # So the condition is simply: ||u-v|| >= ||u|| (assuming all same norm).
    # If we mix norms, it's more complex.
    # The current program uses m=4 (norm 2) and m=2 (norm sqrt(2)).
    # If we mix norms, the "maximum vector norm" is 2.
    # So we need ||u - v|| >= 2 for all pairs.
    # If u has norm 2 and v has norm sqrt(2):
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 2 - 2 u.v = 6 - 2 u.v.
    # We need 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    # This matches the current program's logic (max_ip <= 1).
    
    # So the strategy is: Pack as many vectors as possible from the m=4 shell (norm 2) 
    # and m=2 shell (norm sqrt(2)) such that:
    # 1. Within m=4: ||u-v|| >= 2 => ||u-v||^2 >= 4 => 8 - 2 u.v >= 4 => u.v <= 2.
    #    Since u,v are ±1 on 4 supports, u.v is even.
    #    u.v can be 4, 2, 0, -2, -4.
    #    u.v <= 2 means we cannot have u.v = 4.
    #    u.v = 4 implies u=v. So we just need distinct vectors.
    #    Wait, if supports intersect in 3, u.v can be 3? No, u.v is integer.
    #    If supports intersect in 3, u.v = sum of 3 products of ±1. Sum is odd.
    #    So u.v can be 3, 1, -1, -3.
    #    We need u.v <= 2. So u.v=3 is forbidden.
    #    This confirms the earlier analysis.
    
    # 2. Within m=2: ||u-v|| >= 2 => ||u-v||^2 >= 4 => 4 - 2 u.v >= 4 => u.v <= 0.
    #    u,v are weight 2. u.v can be 2, 0, -2, 1, -1.
    #    u.v <= 0 means we must avoid u.v = 1 and u.v = 2.
    #    u.v = 2 implies u=v.
    #    u.v = 1 implies they share 1 index with same sign.
    
    # 3. Between m=4 and m=2: u.v <= 1.
    
    # The current program finds a MIS on the m=4 supports (330 nodes) where edges are k=3 conflicts.
    # This is a necessary condition but not sufficient?
    # If we pick supports with no k=3 pairs, then any pair of vectors from these supports has u.v <= 1?
    # If k=0,1,2: max u.v is 2 (if k=2 and signs match). 2 <= 2. OK.
    # So if we pick a set of supports with no intersection of size 3, ANY sign assignment is valid for the m=4 shell!
    # This is a huge simplification.
    # We just need to find the Maximum Independent Set in the graph of supports where edge = intersection size 3.
    # The current program does this.
    # How large is this MIS?
    # Total supports 330.
    # Each support has C(7,1)*C(4,3)? No.
    # Number of supports intersecting in 3:
    # Pick 3 common elements. Pick 1 from remaining 7 for S1. Pick 1 from remaining 6 for S2?
    # No, S1 and S2 are distinct.
    # If S1, S2 share 3, they differ by 1 element each.
    # Number of such S2 for a fixed S1: 4 (choices for removed element) * 7 (choices for added element) = 28.
    # So graph is 28-regular on 330 nodes.
    # Average degree 28.
    # Heuristic size of MIS ~ n / (1 + d/2) ? Or n / (d+1)?
    # 330 / 29 ~ 11. This is too low.
    # Actually, for random graphs, MIS is ~ 2n ln(d)/d.
    # But this graph is not random.
    # The current program gets ~220/16 = 13.75 supports?
    # No, current program gets 220 vectors total.
    # If it uses 14 supports * 16 signs = 224 vectors.
    # So it finds ~14 supports.
    # Can we find more supports?
    # If we relax the "no k=3" constraint, we can use more supports, but must carefully choose signs.
    # The current program's approximation (pick supports, then all signs) is very restrictive.
    # It forces u.v <= 1 for ALL pairs, but we only need u.v <= 2.
    # So we can allow k=3 pairs IF we choose signs such that u.v != 3.
    # This allows us to pack MORE supports.
    
    # New Strategy:
    # 1. Generate all 330 supports.
    # 2. Use Backtracking to select a large set of supports.
    # 3. For each selected support, we will assign signs later.
    # 4. The constraint is global on signs.
    # 5. This is a hard CSP.
    
    # Alternative:
    # Use a greedy algorithm on the 5280 vectors directly.
    # Sort vectors by "potential" (e.g., number of conflicts).
    # Pick one, remove all conflicts.
    # This is what the current program effectively does for m=2.
    # Let's apply it to m=4.
    
    # 1. Generate all 5280 m=4 vectors.
    # 2. Compute conflict graph (u.v > 2).
    # 3. Greedy MIS.
    
    # This is O(N^2) for graph construction, N=5280. 27M ops. Fast.
    # Greedy MIS is O(N^2). Fast.
    
    # Let's try this.
    
    def generate_shell(m):
        if m > D:
            return np.empty((0, D), dtype=np.int64)
        vecs = []
        for comb in combinations(range(D), m):
            for mask in range(1 << m):
                v = np.zeros(D, dtype=np.int64)
                for j, idx in enumerate(comb):
                    if mask & (1 << j):
                        v[idx] = 1
                    else:
                        v[idx] = -1
                vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    S4 = generate_shell(4)
    n4 = len(S4)
    
    # Compute inner products
    # S4 @ S4.T
    dots4 = S4 @ S4.T
    
    # Conflict if dot > 2
    # We want Independent Set in graph with edge if dot > 2.
    # Greedy:
    # 1. Calculate degree (number of conflicts) for each node.
    # 2. Pick node with min degree.
    # 3. Add to set, remove node and all neighbors.
    
    # This is a standard greedy heuristic.
    # To improve, we can use "Weighted" greedy or random restarts.
    # Given time limit, let's do a few random restarts with different orderings.
    
    best_set = []
    best_size = 0
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Shuffle indices
        indices = np.arange(n4)
        np.random.shuffle(indices)
        
        # Calculate degrees initially?
        # Or just pick randomly from alive?
        # Min-degree is better.
        
        # Precompute adjacency list for speed?
        # dots4 is 5280x5280. 22M floats. 176MB. Might be too big for memory?
        # 5280*5280*8 bytes = 223 MB. OK.
        
        # Conflict mask
        conflicts = dots4 > 2
        
        # Greedy
        alive = set(range(n4))
        selected = []
        
        # To speed up, we can use a list of alive nodes and filter.
        # But removing from set is O(1).
        # Finding min degree in alive is O(|alive| * deg).
        
        # Let's just pick randomly from alive for speed, or min degree if fast enough.
        # With 5280 nodes, 20 restarts, it should be fine.
        
        while alive:
            # Pick node with min conflicts in alive
            # This is O(N^2) per restart. 20 * 27M = 540M ops. Might be slow in Python.
            # Optimization: Use numpy to find min?
            # Or just random pick?
            
            # Let's try random pick first.
            v = indices[np.random.randint(len(indices))]
            # Ensure v is still alive? 
            # Better to maintain a list of alive indices.
            
            # Let's use a different approach:
            # 1. Compute degree of each node.
            # 2. Sort by degree.
            # 3. Iterate.
            
            pass # Placeholder, will implement below
    
    # Actually, let's use the support-based approach but with sign assignment.
    # If we pick a set of supports, we can solve the sign assignment problem.
    # The sign assignment problem is:
    # For each support S, choose sign vector s_S in {±1}^4.
    # Constraint: For any pair S1, S2, dot(s_S1, s_S2) <= 2.
    # This is a SAT problem.
    # With 330 supports, we can't pick all.
    # But we can pick a subset.
    
    # Let's stick to the Greedy on 5280 vectors but optimize it.
    
    # 1. Compute conflicts
    conflicts = dots4 > 2
    
    # 2. Greedy with random restarts
    best_vectors = []
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        alive = list(range(n4))
        np.random.shuffle(alive)
        
        selected = []
        
        # Use a boolean array for alive for fast lookup
        alive_mask = np.ones(n4, dtype=bool)
        
        while np.any(alive_mask):
            # Pick a random alive node
            # To bias towards low degree, we could weight, but random is okay for start.
            idx = np.random.choice(np.where(alive_mask)[0])
            
            selected.append(idx)
            
            # Remove idx and all its neighbors
            # Neighbors are where conflicts[idx] is True
            to_remove = np.where(conflicts[idx])[0]
            alive_mask[to_remove] = False
            alive_mask[idx] = False
            
        if len(selected) > len(best_vectors):
            best_vectors = selected
            if len(best_vectors) > 500: # If we get really good, stop?
                pass
    
    S4_best = S4[best_vectors]
    
    # Now add m=2 vectors
    S2 = generate_shell(2)
    if len(S4_best) > 0:
        dots_24 = S2 @ S4_best.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = S2[valid_2_mask]
        
        n2 = len(S2_candidates)
        if n2 > 0:
            dots_22 = S2_candidates @ S2_candidates.T
            # Conflict if dot > 0
            conflicts2 = dots_22 > 0
            
            # Greedy on S2
            best_s2 = []
            for restart in range(20):
                if time.time() > deadline - 5:
                    break
                alive_mask2 = np.ones(n2, dtype=bool)
                selected2 = []
                while np.any(alive_mask2):
                    idx = np.random.choice(np.where(alive_mask2)[0])
                    selected2.append(idx)
                    to_remove = np.where(conflicts2[idx])[0]
                    alive_mask2[to_remove] = False
                    alive_mask2[idx] = False
                if len(selected2) > len(best_s2):
                    best_s2 = selected2
            
            S2_best = S2_candidates[best_s2]
            result = np.vstack([S4_best, S2_best])
        else:
            result = S4_best
    else:
        result = S4_best
        
    return result

# EVOLVE-BLOCK-END
