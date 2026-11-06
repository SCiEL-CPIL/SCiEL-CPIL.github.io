import time
import random
import sys
from itertools import combinations

# Increase recursion limit for deep branch-and-bound
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES) # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

def _build_conflict_graph():
    """
    Build the conflict graph where an edge exists between two triples 
    if they are part of a K(2,2,2) configuration.
    """
    # adjacency[i] is a bitmask of triples j that conflict with i
    adjacency = [0] * N_TRIPLES
    
    # For each pair of triples, check if they conflict.
    # Two triples (a,b,c) and (d,e,f) conflict if there exist 
    # a', b', c' such that the 8 transversals exist.
    # Actually, the definition of K(2,2,2) free means we cannot have 
    # three disjoint pairs (x1,x2), (y1,y2), (z1,z2) such that all 8 
    # combinations (xi, yj, zk) are in the hypergraph.
    
    # A set of 8 triples forms a K(2,2,2).
    # Any pair of triples within this K(2,2,2) that share no vertices? 
    # No, the condition is that the *set* of triples contains the 8 transversals.
    # So, two triples T1 and T2 "conflict" in the independent set sense 
    # if adding T2 to a set containing T1 makes it impossible to extend 
    # to a K(2,2,2)-free set? 
    # Actually, for MIS formulation:
    # Nodes = Triples.
    # Edge (T1, T2) exists if {T1, T2} is a subset of some K(2,2,2).
    # Wait, if {T1, T2} is a subset of a K(2,2,2), it doesn't mean the 
    # hypergraph contains the whole K(2,2,2). It just means they *could* 
    # be part of one.
    # The constraint is: No 8 specific triples (forming a K222) are ALL present.
    # This is a 3-uniform hypergraph constraint, not a pairwise graph constraint 
    # in the standard sense. 
    # HOWEVER, K(2,2,2) is a specific structure. 
    # If we pick an independent set in a graph where edges represent 
    # "being in the same K222", is that sufficient?
    # No. A K222 consists of 8 triples. If we have 7 of them, it's fine. 
    # If we have all 8, it's bad.
    # So the "conflict" is a hyperedge of size 8.
    # Finding Max Independent Set in a hypergraph is harder.
    
    # BUT, let's look at the structure. 
    # A K(2,2,2) is defined by 3 pairs of vertices.
    # There are C(16,2)*C(14,2)*C(12,2)/... many such structures.
    
    # Alternative approach: 
    # The current seed program uses a direct check `_check_conflict_fast`.
    # It checks if adding a triple creates a K(2,2,2).
    # This implies the property is "For any 3 disjoint pairs, not all 8 transversals are present".
    
    # Let's stick to the efficient local search + greedy but optimize it heavily, 
    # or use a SAT solver if available? No, standard lib + numpy/scipy.
    
    # The Tomita algorithm works on graphs. 
    # Can we reduce this to a graph problem?
    # If we define a graph where T1 ~ T2 if they are in a common K222, 
    # then an independent set in this graph might still contain a K222 
    # (if it contains 8 nodes that are pairwise non-adjacent? No, in a K222, 
    # many triples share vertices or are in the same K222).
    
    # Actually, any two distinct triples in a K222 are "related".
    # If we build a graph where T1-T2 edge exists if they are part of 
    # the SAME K222, then a K222 is a clique of size 8? 
    # Let's check. In a K222 formed by pairs (a,a'), (b,b'), (c,c'), 
    # the triples are (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), etc.
    # Do (a,b,c) and (a',b',c') share a K222? Yes, the one defined by 
    # (a,a'), (b,b'), (c,c').
    # So all 8 triples in a K222 form a clique in this "conflict graph".
    # Therefore, any Independent Set in this graph cannot contain all 8 
    # triples of any K222.
    # Is it possible that an Independent Set contains a K222 that is NOT 
    # a clique in this graph? 
    # No, by definition, if they form a K222, they are in the same K222, 
    # so they are connected in the graph.
    # So, MIS in this graph gives a K222-free hypergraph.
    # Is it the MAXIMUM? 
    # Yes, because any K222-free hypergraph is an independent set in this graph.
    # And any independent set in this graph is K222-free.
    
    # So the problem reduces to MIS on a graph with 560 nodes.
    
    pass

# Build the conflict graph
# For each triple T, find all other triples T' such that T and T' are in a common K222.
# T=(a,b,c). 
# A K222 containing T must have T as one of the 8 transversals.
# The pairs of the K222 must include pairs that "cover" a,b,c.
# Specifically, the K222 is defined by 3 disjoint pairs of vertices.
# Let the pairs be P1, P2, P3.
# T = {x, y, z} where x in P1, y in P2, z in P3.
# So P1 must contain a (or b or c). 
# Actually, T picks one vertex from each pair.
# So if T={a,b,c}, then one of the pairs must contain a, one must contain b, one must contain c.
# Let P1 contain a. P1 = {a, a'}.
# Let P2 contain b. P2 = {b, b'}.
# Let P3 contain c. P3 = {c, c'}.
# The K222 is determined by a', b', c'.
# The 8 triples are all combinations of {a,a'} x {b,b'} x {c,c'}.
# For T' to be in the same K222, T' must be one of these 8 triples.
# So T' shares 2 vertices with T? 
# No. T={a,b,c}. T'={a,b,c'}. They share a,b.
# T'={a',b,c}. They share b,c.
# T'={a',b',c}. They share c.
# T'={a',b',c'}. They share 0 vertices.
# So T and T' are in a common K222 if and only if there exist a', b', c' 
# such that T' is in the set of 8 triples generated by (a,a'),(b,b'),(c,c').
# This implies that T and T' must have a "compatible" structure.
# Specifically, let S = T \cap T'.
# If |S|=2, say {a,b}, then T={a,b,c}, T'={a,b,c'}. 
# We need a K222 containing both. 
# Pairs: {a,a'}, {b,b'}, {c,c'}. 
# T uses a,b,c. T' uses a,b,c'.
# This works if we set the pairs as {a,a'}, {b,b'}, {c,c'}. 
# But wait, the pairs must be DISJOINT.
# {a,a'} and {b,b'} are disjoint if a!=b, a!=b', a'!=b, a'!=b'.
# Since a,b distinct, and a',b' chosen distinct from {a,b,c,c'}, this is fine.
# So if |T \cap T'| = 2, they are in a common K222?
# Let T={1,2,3}, T'={1,2,4}.
# Pairs: {1,x}, {2,y}, {3,4}. 
# K222 triples: (1,2,3), (1,2,4), (1,y,3), (1,y,4), (x,2,3), (x,2,4), (x,y,3), (x,y,4).
# Yes, T and T' are in this K222.
# So if |T \cap T'| = 2, they conflict.

# If |T \cap T'| = 1, say {a}. T={a,b,c}, T'={a,d,e}.
# Can they be in a common K222?
# The K222 must have a pair containing a. Say {a, a'}.
# The other two pairs must cover {b,c} and {d,e}.
# Since the pairs are disjoint, one pair must contain b (or c) and the other d (or e).
# But a K222 has exactly 3 pairs.
# T uses a, b, c. T' uses a, d, e.
# If they are in the same K222, the 3 pairs of the K222 must generate both.
# This implies b,c,d,e must be distributed among the other 2 pairs?
# But each pair has 2 vertices. Total 4 vertices.
# So {b,c,d,e} must be exactly the vertices in the other two pairs.
# So one pair is {b,d} or {b,e} or {c,d} etc?
# If pair 2 is {b,d}, pair 3 is {c,e}.
# Then K222 triples include (a,b,c) and (a,d,e).
# Yes.
# So if |T \cap T'| = 1, and the remaining 4 vertices can be partitioned into 2 pairs, 
# they are in a common K222.
# The remaining 4 vertices are always partitionable into 2 pairs (3 ways).
# So any two triples sharing exactly 1 vertex are in a common K222.

# If |T \cap T'| = 0. T={a,b,c}, T'={d,e,f}.
# Can they be in a common K222?
# The 3 pairs of the K222 must cover a,b,c and d,e,f.
# Since the pairs are disjoint, and there are 6 vertices, each vertex must belong to exactly one pair.
# So the 3 pairs must be a perfect matching of the 6 vertices.
# T picks one from each pair. T' picks one from each pair.
# This is possible if and only if there exists a perfect matching of {a,b,c,d,e,f} 
# such that T and T' are transversals.
# Actually, ANY two disjoint triples are in a common K222?
# Let's check.
# We need to partition {a,b,c,d,e,f} into 3 pairs P1,P2,P3 
# such that T has one from each, and T' has one from each.
# This is equivalent to saying that the bipartite graph between T and T' 
# (with edges representing "can be in same pair"?) No.
# It's equivalent to finding a permutation.
# Actually, it's always possible. 
# Example: T={1,2,3}, T'={4,5,6}.
# Pairs: {1,4}, {2,5}, {3,6}.
# T is (1,2,3). T' is (4,5,6).
# Both are transversals.
# So any two disjoint triples are in a common K222.

# Conclusion: 
# Any two distinct triples T1, T2 are in a common K222.
# Proof:
# - If |intersection|=2, we showed it.
# - If |intersection|=1, we showed it.
# - If |intersection|=0, we showed it.
# - If |intersection|=3, they are the same triple.
#
# So the conflict graph is a COMPLETE GRAPH (K_560)??
# If the conflict graph is complete, the MIS size is 1.
# That can't be right. The seed program finds ~2.4 * seed. 
# Wait, the seed program finds a VALID system. 
# The definition of "conflict" in MIS formulation must be:
# "T1 and T2 cannot BOTH be in the hypergraph".
# This happens if {T1, T2} is a subset of a K222 that is "forced"?
# No. The condition is: No K222 is FULLY present.
# So {T1, T2} is allowed as long as they don't complete a K222 with other existing triples.
# The pairwise conflict graph approach (MIS on graph where edge=shared K222) 
# is TOO STRONG. It forbids any two triples that share a K222, 
# even if the other 6 triples of that K222 are not present.
# So MIS on that graph is a LOWER bound, not the exact solution.
# And since the graph is complete (any two triples share a K222), 
# MIS size would be 1. 
# Wait, did I prove any two triples share a K222?
# Yes. 
# So the "Conflict Graph" approach is invalid for the exact solution.
# The constraint is a 3-uniform hypergraph constraint (forbidden subhypergraph).
# This is a "Maximum 3-uniform hypergraph without a K222".
# This is NOT a standard MIS problem on a simple graph.
#
# Therefore, we must stick to the heuristic search (Greedy + Local Search) 
# but optimize it significantly.
#
# Optimizations:
# 1. Fast conflict check using bitmasks (already in seed).
# 2. Better Local Search: 
#    - Try removing 1 triple and adding multiple.
#    - Try removing 2 triples and adding multiple.
#    - Use a priority queue for "most promising" additions?
# 3. Randomized Restart with different initial orderings.
# 4. Use a time budget to run many iterations.

def _check_conflict_fast(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    """
    # a' must be in partner_mask[b][c]
    mask_a_candidates = partner_mask[b*V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    # b' must be in partner_mask[a][c]
    mask_b_base = partner_mask[a*V + c]
    mask_b_base &= ~(1 << a)
    mask_b_base &= ~(1 << b)
    mask_b_base &= ~(1 << c)
    
    # c' must be in partner_mask[a][b]
    mask_c_base = partner_mask[a*V + b]
    mask_c_base &= ~(1 << a)
    mask_c_base &= ~(1 << b)
    mask_c_base &= ~(1 << c)
    
    while mask_a_candidates:
        lsb = mask_a_candidates & -mask_a_candidates
        a_p = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        mask_b_valid = mask_b_base & partner_mask[a_p*V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        mask_c_valid = mask_c_base & partner_mask[a_p*V + b]
        mask_c_valid &= ~(1 << a_p)
        
        if mask_c_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            mask_c_final = mask_c_valid
            mask_c_final &= partner_mask[a*V + b_p]
            mask_c_final &= partner_mask[a_p*V + b_p]
            mask_c_final &= ~(1 << b_p)
            
            if mask_c_final:
                return True
                
    return False

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a*V + b] |= (1 << c)
    partner_mask[b*V + a] |= (1 << c)
    partner_mask[a*V + c] |= (1 << b)
    partner_mask[c*V + a] |= (1 << b)
    partner_mask[b*V + c] |= (1 << a)
    partner_mask[c*V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a*V + b] &= ~(1 << c)
    partner_mask[b*V + a] &= ~(1 << c)
    partner_mask[a*V + c] &= ~(1 << b)
    partner_mask[c*V + a] &= ~(1 << b)
    partner_mask[b*V + c] &= ~(1 << a)
    partner_mask[c*V + b] &= ~(1 << a)

def _greedy_with_presence(order, deadline, partner_mask):
    presence = [False] * N_TRIPLES
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict_fast(a, b, c, partner_mask):
            continue
        
        count += 1
        _add_triple(a, b, c, partner_mask, presence, idx)
        
    return count, presence

def _local_search(count, presence, partner_mask, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing triple
        for idx in range(N_TRIPLES):
            if time.time() >= deadline:
                return count, presence, partner_mask
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict_fast(a, b, c, partner_mask):
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx)
                improved = True
        
        if improved:
            continue
            
        # 2. Swaps
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        if not present_indices:
            break
            
        swaps_tried = 0
        max_swaps = 100
        
        while swaps_tried < max_swaps and time.time() < deadline:
            swaps_tried += 1
            idx_out = random.choice(present_indices)
            a, b, c = ALL_TRIPLES[idx_out]
            
            old_partner_mask = partner_mask[:]
            old_presence = presence[:]
            old_count = count
            
            _remove_triple(a, b, c, partner_mask, presence, idx_out)
            count -= 1
            
            added_any = False
            for idx_in in range(N_TRIPLES):
                if time.time() >= deadline:
                    partner_mask = old_partner_mask
                    presence = old_presence
                    count = old_count
                    return count, presence, partner_mask
                if presence[idx_in]:
                    continue
                a_in, b_in, c_in = ALL_TRIPLES[idx_in]
                if not _check_conflict_fast(a_in, b_in, c_in, partner_mask):
                    count += 1
                    _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
                    added_any = True
            
            if count > old_count:
                improved = True
                break
            else:
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
                
    return count, presence, partner_mask

def construct():
    start = time.time()
    deadline = start + 115
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    random.seed(42)
    
    # Run multiple attempts with different random orders
    while time.time() < deadline:
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        partner_mask = [0] * (V * V)
        
        count, presence = _greedy_with_presence(order, deadline, partner_mask)
        count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
            
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result
