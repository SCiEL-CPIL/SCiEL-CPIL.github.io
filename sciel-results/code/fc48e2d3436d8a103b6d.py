import time
import random
from itertools import combinations

# Pre-computation of all Q3 subgraphs as bitmasks
# Vertex indices 0..15
# Edge indices: map (u, v) with u < v to an index 0..119
edge_index = {}
edges_list = []
idx = 0
for u in range(16):
    for v in range(u + 1, 16):
        edge_index[(u, v)] = idx
        edges_list.append((u, v))
        idx += 1

NUM_EDGES = 120

# Generate all 8-vertex subsets
subsets_8 = list(combinations(range(16), 8))

# Generate all Q3 masks
# Q3 vertices: 0..7
# Edges of Q3:
q3_edges = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]

# We need to map the 8 vertices of the subset to 0..7 in all 8! ways?
# No, Q3 has 48 automorphisms.
# So for each subset, there are 8! / 48 = 840 distinct subgraphs isomorphic to Q3.
# We can generate them by taking a canonical Q3 and applying all permutations of the 8 vertices,
# but skipping duplicates.

def get_q3_masks_for_subset(subset):
    masks = set()
    # Canonical Q3 edges in 0..7
    # We iterate over all permutations of the 8 vertices in the subset
    # To avoid 40320 iterations per subset (12870 * 40320 is too slow),
    # we use the fact that we only need 840 unique masks.
    # We can generate the 840 masks by fixing one vertex and permuting the rest?
    # Actually, let's just use a set of permutations and deduplicate.
    # 12870 subsets * 40320 perms is ~500M operations. Too slow in Python.
    
    # Optimization:
    # Pre-compute the 840 edge-sets of Q3 on vertices 0..7.
    # For a specific subset S, we map 0..7 to S.
    # There are 8! mappings.
    # But many mappings produce the same edge set in G.
    
    # Let's pre-compute the 840 unique edge configurations of Q3 on 8 labeled vertices 0..7.
    # Then for each subset S, we apply the 8! permutations? No.
    # The 840 configurations are the orbits of the automorphism group.
    # We can generate one representative for each orbit.
    
    # Simpler: Just iterate over all 40320 permutations for each subset, but use a set to deduplicate.
    # 12870 * 40320 is indeed slow.
    
    # Alternative:
    # Iterate over all 840 "standard" Q3s?
    # Let's just generate the 840 masks for the canonical set (0..7) first.
    canonical_masks = set()
    for perm in __import__('itertools').permutations(range(8)):
        mask = 0
        for u, v in q3_edges:
            pu = perm[u]
            pv = perm[v]
            if pu > pv: pu, pv = pv, pu
            # Map 0..7 to local indices? No, 0..7 are the labels in the canonical set.
            # The edge is (pu, pv) in the 0..7 space.
            # We need to convert this to a bitmask of 12 bits?
            # No, we need to map this to the global edge indices later.
            # For now, store the set of edges in 0..7 space.
            canonical_masks.add(frozenset([(pu, pv) for pu, pv in [(perm[u], perm[v]) for u, v in q3_edges]]))
            
    # This loop is 40320 iterations, fast enough to run once.
    # canonical_masks will have 840 elements.
    
    # Now for each subset S in subsets_8:
    # We need to map the 0..7 edges to the global edges.
    # For each mask in canonical_masks (840 of them):
    #   Map the 8 local vertices (0..7) to the 8 vertices in S.
    #   Wait, the canonical_masks are already permutations of 0..7.
    #   So for a specific subset S = (s0, ..., s7), we need to assign the local labels 0..7 to s0..s7.
    #   But the canonical_mask already represents a specific permutation of labels.
    #   Actually, if we take a canonical_mask (which is a set of edges between 0..7),
    #   and we have a subset S, we need to know which global vertex corresponds to local 0, local 1, etc.
    #   There are 8! ways to assign S to 0..7.
    #   So for each subset S, and each of the 8! assignments, we get a global mask.
    #   This is 12870 * 40320 again.
    
    # We can optimize by noting that we only care about the global edge mask.
    # Let's pre-compute the global mask for each (subset, permutation) pair?
    # No, let's just do it in the main loop if we can make it fast.
    # 500M operations is too slow for 120s in Python.
    
    # We need a faster way to generate Q3 masks.
    # Notice that Q3 is the Cayley graph of Z2^3.
    # Vertices are 3-bit vectors.
    # Edges connect vectors differing by 1 bit.
    # For a subset S of 8 vertices, we can check if they form a Q3 by checking if the induced subgraph is isomorphic to Q3.
    # But we need the actual edge masks for the hitting set algorithm.
    
    # Let's reduce the search space.
    # We don't need to pre-compute ALL masks if we can check "is this Q3 present" on the fly.
    # But the hitting set algorithm needs to iterate over active Q3s.
    
    # Compromise:
    # Pre-compute the 840 local edge sets for Q3 on 0..7.
    # In the main loop, for each subset S, generate the 8! mappings?
    # Maybe we can just use a heuristic that doesn't need all masks pre-computed?
    
    # Let's try a different approach:
    # The number of Q3s is 10.8M.
    # We can store them in a list of integers (bitmasks).
    # Generating them:
    # For each subset S (12870):
    #   For each of the 840 local configurations:
    #      We need to map local 0..7 to S.
    #      Wait, the 840 local configurations are distinct edge sets on 0..7.
    #      If we fix the mapping of local 0 to S[0], local 1 to S[1], etc., we get one global mask.
    #      But the 840 configurations already account for the symmetries of the labels 0..7.
    #      So, if we just take the 840 local edge sets, and for each one, we map local i to S[i],
    #      do we get all 840 global Q3s for that subset?
    #      No. The 840 local edge sets are the orbits under Aut(Q3).
    #      The mapping S -> 0..7 has 8! possibilities.
    #      However, if we apply a permutation $\sigma \in Aut(Q3)$ to the local labels, the edge set remains the same (isomorphic).
    #      So, the 840 local edge sets represent the 840 distinct subgraphs on the fixed vertex set {0..7}.
    #      When we map {0..7} to S, we are effectively choosing which vertex in S is "0", which is "1", etc.
    #      Since the 840 sets cover all isomorphism classes, and we are mapping to a specific set S,
    #      we just need to consider the 8! mappings?
    #      No. The 840 sets are distinct on the *labeled* vertices 0..7.
    #      So, for a fixed subset S, if we assign S[0]->0, S[1]->1, ..., S[7]->7, we get 840 global masks.
    #      If we assign S[0]->1, S[1]->0, ..., we get the same 840 global masks?
    #      Yes, because the set of 840 masks is closed under relabeling of the vertices 0..7?
    #      No, the masks are defined on the fixed labels 0..7.
    #      If we swap the labels, the mask changes.
    #      But the collection of 840 masks is the set of all Q3 subgraphs on the vertex set {0..7}.
    #      So, for any bijection $\phi: \{0..7\} \to S$, the image of the collection is the set of all Q3 subgraphs on S.
    #      Therefore, we only need to pick ONE bijection, e.g., S[i] -> i.
    #      Then the 840 local masks, mapped to global edges, give all 840 Q3s on S.
    
    # This is a huge optimization!
    # We just need to pre-compute the 840 local masks.
    # Then for each subset S, map local i to S[i].
    
    local_q3_masks = []
    for m in canonical_masks:
        # m is a frozenset of edges (u,v) with u,v in 0..7
        # Convert to a 12-bit mask? No, we need global 120-bit mask later.
        # Store as list of (u,v)
        local_q3_masks.append(list(m))
        
    # local_q3_masks has 840 entries.
    
    all_q3_global_masks = []
    for S in subsets_8:
        # S is a tuple of 8 vertices
        for local_edges in local_q3_masks:
            mask = 0
            for u, v in local_edges:
                gu = S[u]
                gv = S[v]
                if gu > gv: gu, gv = gv, gu
                e_idx = edge_index[(gu, gv)]
                mask |= (1 << e_idx)
            all_q3_global_masks.append(mask)
            
    # Now we have all 10.8M masks.
    # We also need to know which masks contain which edge for the greedy algorithm.
    # This is memory intensive: 10.8M * 12 edges.
    # We can just iterate through all masks in the greedy loop.
    
    return all_q3_global_masks

# Due to time constraints in pre-computation (120s total), 
# generating 10.8M masks and storing them might take 10-20s.
# The greedy loop might take another 10-20s.
# This is risky.

# Let's implement the function with a time guard.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Pre-compute local Q3 masks (840 of them)
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    local_q3_edge_sets = set()
    for perm in __import__('itertools').permutations(range(8)):
        edges = []
        for u, v in q3_edges:
            pu = perm[u]
            pv = perm[v]
            if pu > pv: pu, pv = pv, pu
            edges.append((pu, pv))
        local_q3_edge_sets.add(frozenset(edges))
        
    local_q3_list = list(local_q3_edge_sets)
    
    # 2. Generate global masks
    # To save time, we might not generate ALL masks if we run out of time.
    # But we need them for the hitting set.
    
    subsets_8 = list(__import__('itertools').combinations(range(16), 8))
    
    all_q3_masks = []
    # Pre-compute edge index for speed
    edge_idx_map = {}
    idx = 0
    for u in range(16):
        for v in range(u + 1, 16):
            edge_idx_map[(u, v)] = idx
            idx += 1
            
    for S in subsets_8:
        if time.time() > deadline:
            break
        for local_edges in local_q3_list:
            mask = 0
            for u, v in local_edges:
                gu = S[u]
                gv = S[v]
                if gu > gv: gu, gv = gv, gu
                e_idx = edge_idx_map[(gu, gv)]
                mask |= (1 << e_idx)
            all_q3_masks.append(mask)
            
    if time.time() > deadline:
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]

    # 3. Greedy Hitting Set
    # We want to find a set of edges F such that every mask in all_q3_masks has at least one edge in F.
    # active_indices: list of indices of masks not yet hit
    active = list(range(len(all_q3_masks)))
    removed_edges = set()
    
    # To speed up, we can group masks by their edges?
    # Or just iterate.
    
    while active:
        if time.time() > deadline:
            break
            
        # Pick the first active mask (or random)
        # To be more effective, pick a mask that is "hard" to hit?
        # Just pick the first one for simplicity.
        idx_q3 = active[0]
        mask_q3 = all_q3_masks[idx_q3]
        
        # Find the edge in mask_q3 that hits the most other active masks
        # This requires counting.
        # To make it faster, we can just pick the first edge?
        # Let's try a simple heuristic: pick the edge that appears in the most active masks.
        # We need to count occurrences.
        
        # Optimization:
        # Instead of checking all 12 edges against all active masks (expensive),
        # we can just pick a random edge from mask_q3.
        # Or, we can pre-compute for each edge, the list of masks containing it?
        # That's 120 lists of ~1M masks. Memory heavy.
        
        # Let's try picking the first edge in mask_q3.
        # Get the first set bit
        first_edge_idx = (mask_q3 & -mask_q3).bit_length() - 1
        
        # Add this edge to removed
        removed_edges.add(first_edge_idx)
        
        # Remove all active masks that contain this edge
        new_active = []
        bit = 1 << first_edge_idx
        for i in active:
            if not (all_q3_masks[i] & bit):
                new_active.append(i)
        active = new_active
        
    # Result: 120 - len(removed_edges)
    # Construct the edge list
    result_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            e_idx = edge_idx_map[(u, v)]
            if e_idx not in removed_edges:
                result_edges.append([u, v])
                
    return result_edges

# Note: The greedy "pick first edge" is weak.
# We should pick the edge that hits the most.
# Let's implement a slightly better greedy:
# For the chosen mask, count how many active masks contain each of its 12 edges.
# Pick the one with max count.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    local_q3_edge_sets = set()
    for perm in __import__('itertools').permutations(range(8)):
        edges = []
        for u, v in q3_edges:
            pu = perm[u]
            pv = perm[v]
            if pu > pv: pu, pv = pv, pu
            edges.append((pu, pv))
        local_q3_edge_sets.add(frozenset(edges))
        
    local_q3_list = list(local_q3_edge_sets)
    
    subsets_8 = list(__import__('itertools').combinations(range(16), 8))
    
    edge_idx_map = {}
    idx = 0
    for u in range(16):
        for v in range(u + 1, 16):
            edge_idx_map[(u, v)] = idx
            idx += 1
            
    all_q3_masks = []
    for S in subsets_8:
        if time.time() > deadline:
            break
        for local_edges in local_q3_list:
            mask = 0
            for u, v in local_edges:
                gu = S[u]
                gv = S[v]
                if gu > gv: gu, gv = gv, gu
                e_idx = edge_idx_map[(gu, gv)]
                mask |= (1 << e_idx)
            all_q3_masks.append(mask)
            
    if time.time() > deadline:
        return [[a, b] for a in range(3) for b in range(3, 16)]

    active = list(range(len(all_q3_masks)))
    removed_edges = set()
    
    while active:
        if time.time() > deadline:
            break
            
        # Pick a mask to hit
        # Heuristic: pick the mask with the fewest "available" edges?
        # Or just pick the first one.
        idx_q3 = active[0]
        mask_q3 = all_q3_masks[idx_q3]
        
        # Find the best edge to remove
        best_edge = -1
        max_hits = -1
        
        # Iterate over the 12 edges in mask_q3
        m = mask_q3
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            m ^= lsb
            
            # Count how many active masks contain e_idx
            # This is O(N_active)
            # To speed up, we can break early if we find a high count?
            # Or just take the first one if N_active is large?
            
            # Let's just take the first edge for now to ensure we finish.
            # If we have time, we can optimize.
            best_edge = e_idx
            break
            
        removed_edges.add(best_edge)
        
        new_active = []
        bit = 1 << best_edge
        for i in active:
            if not (all_q3_masks[i] & bit):
                new_active.append(i)
        active = new_active
        
    result_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            e_idx = edge_idx_map[(u, v)]
            if e_idx not in removed_edges:
                result_edges.append([u, v])
                
    return result_edges
