"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _get_supports(m):
    """Generate all supports (frozensets of indices) of size m."""
    return [frozenset(comb) for comb in combinations(range(D), m)]

def _generate_vectors_from_support(support):
    """Generate all 2^m sign variants for a given support."""
    indices = list(support)
    m = len(indices)
    vecs = []
    for mask in range(1 << m):
        v = np.zeros(D, dtype=np.int64)
        for j, idx in enumerate(indices):
            if mask & (1 << j):
                v[idx] = 1
            else:
                v[idx] = -1
        vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Strategy:
    # We model the problem using "supports" (the set of non-zero indices).
    # For a vector u with support S_u and vector v with support S_v, 
    # the inner product u.v depends on the intersection S_u ∩ S_v and the signs.
    # 
    # If we restrict ourselves to vectors with norm squared 4 (i.e., |S|=4, entries ±1),
    # then M^2 = 4.
    # Constraint: ||u - v||^2 >= 4  =>  4 + 4 - 2 u.v >= 4  =>  u.v <= 2.
    # 
    # For two vectors with supports S1, S2:
    # u.v = sum_{i in S1 ∩ S2} u_i v_i.
    # Since u_i, v_i are ±1, u.v is an integer between -|S1 ∩ S2| and |S1 ∩ S2|.
    # The maximum possible inner product for a given pair of supports is |S1 ∩ S2| 
    # (achieved when signs match on the intersection).
    # 
    # If we select a set of supports such that |S1 ∩ S2| <= 2 for all distinct pairs,
    # then for ANY sign assignments, u.v <= |S1 ∩ S2| <= 2.
    # Thus, the constraint u.v <= 2 is automatically satisfied for all sign variants.
    # 
    # This reduces the problem to finding a Maximum Independent Set (MIS) in a graph where:
    # - Nodes are all supports of size 4 (there are C(11,4) = 330).
    # - An edge exists between two supports if their intersection size is > 2.
    # 
    # Once we have the MIS of supports, we can include ALL sign variants of each support.
    # Total vectors k = sum_{S in MIS} 2^{|S|} = |MIS| * 16.
    
    # Step 1: Generate all supports of size 4
    supports = list(_get_supports(4))
    n_supports = len(supports)
    
    # Step 2: Build conflict graph
    # Conflict if |S1 ∩ S2| > 2
    # We can precompute the bit representation of each support for fast intersection
    support_bits = []
    for s in supports:
        bits = 0
        for idx in s:
            bits |= (1 << idx)
        support_bits.append(bits)
        
    # Build adjacency list or conflict matrix
    # n_supports = 330, so a 330x330 matrix is small.
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            inter = support_bits[i] & support_bits[j]
            # Count bits in intersection
            # Python int has no popcount in older versions, but bin(x).count('1') works
            # Or use numpy: np.bit_count is available in newer numpy, but let's be safe
            count = bin(inter).count('1')
            if count > 2:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Step 3: Find Maximum Independent Set (MIS)
    # Since n=330 is small enough for heuristic search but too large for brute force,
    # we use a greedy heuristic with random restarts.
    # Greedy: Pick a node, remove it and its neighbors, repeat.
    # To improve, we can try different orderings.
    
    best_mis_size = 0
    best_mis_nodes = []
    
    # Heuristic: Greedy with randomization
    # We can also try a simple local search or simulated annealing if time permits.
    # Given the budget, a well-tuned greedy with multiple restarts should be sufficient.
    # The graph is relatively sparse? 
    # Degree of a node: How many other supports intersect it in > 2 elements?
    # If S1 has 4 elements, S2 must share at least 3.
    # If they share 3, S2 is determined by 3 elements of S1 and 1 new element from remaining 7.
    # C(4,3)*7 = 4*7 = 28.
    # If they share 4, S2 = S1 (but we consider distinct supports, so no self-loop).
    # So degree is roughly 28. Total edges ~ 330*28/2 = 4620. Very sparse.
    
    # Greedy MIS on sparse graph
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        nodes = list(range(n_supports))
        np.random.shuffle(nodes)
        
        selected = []
        removed = np.zeros(n_supports, dtype=bool)
        
        for node in nodes:
            if removed[node]:
                continue
            
            # Select this node
            selected.append(node)
            
            # Remove node and all its neighbors
            removed[node] = True
            neighbors = np.where(conflict[node])[0]
            removed[neighbors] = True
            
            # Optimization: if no nodes left, break
            if not np.any(~removed):
                break
        
        if len(selected) > best_mis_size:
            best_mis_size = len(selected)
            best_mis_nodes = selected
            
        # Early stop if we found a very good solution?
        # Theoretical upper bound: 
        # Each support uses 4 indices. Total indices 11.
        # If pairwise intersection <= 2, it's a 2-(11,4,lambda) packing design?
        # Actually, it's a constant weight code with distance constraint.
        # Let's just run the restarts.
        
    # Step 4: Construct the vector array
    if best_mis_size == 0:
        # Fallback: just one support
        best_mis_nodes = [0]
        
    vectors = []
    for node_idx in best_mis_nodes:
        support = supports[node_idx]
        vecs = _generate_vectors_from_support(support)
        vectors.append(vecs)
        
    if not vectors:
        # Should not happen
        return np.zeros((0, D), dtype=np.int64)
        
    result = np.vstack(vectors)
    
    # Optional: Verify the result to be safe
    # k = len(result)
    # if k > 1:
    #     norms_sq = np.sum(result**2, axis=1)
    #     M_sq = np.max(norms_sq)
    #     # Check min pairwise distance
    #     # This is O(k^2 * D). k is at most 330*16 = 5280. 
    #     # 5280^2 * 11 ~ 300 million ops. Might be slow in Python loops, but vectorized numpy is fast.
    #     # Let's do a quick check if k is small enough, or trust the construction.
    #     # Given the logic is sound (intersection <= 2 implies dot <= 2), we trust it.
    #     pass
        
    return result

# EVOLVE-BLOCK-END
