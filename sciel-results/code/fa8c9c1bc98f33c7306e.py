import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n = 16
    # Precompute all 8-subsets of vertices
    all_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute Q3 edge patterns in K8
    # Vertices 0..7
    # Q3 is the cube. Edges are between vertices differing in 1 bit.
    # Let's map vertices 0..7 to 3-bit binary.
    # Edge if hamming distance is 1.
    q3_patterns = []
    # Generate all permutations of 0..7? No, just generate the set of edges for the standard cube.
    # The standard cube edges are fixed relative to the labels 0..7 if we fix the mapping.
    # But the subset S can be mapped to 0..7 in 8! ways.
    # However, the set of edges in the induced subgraph G[S] is invariant under relabeling?
    # No, the pattern of edges in G[S] depends on which vertices are which.
    # So for a specific subset S = (v0, ..., v7), we need to check if the edges present
    # form a Q3.
    # This means: Is there a bijection f: {0..7} -> S such that
    # (u,v) in E(G) iff hamming(f^{-1}(u), f^{-1}(v)) == 1?
    # This is equivalent to: The graph induced by S is isomorphic to Q3.
    # Wait, the task says "contains no subgraph isomorphic to Q3".
    # Since |S|=8, a subgraph isomorphic to Q3 must use all 8 vertices.
    # So we just need to check if G[S] is isomorphic to Q3.
    
    # How to check isomorphism to Q3 quickly?
    # Q3 is 3-regular.
    # 1. Check if all degrees in G[S] are 3.
    # 2. Check if it's connected? (Q3 is connected)
    # 3. Check if it's bipartite? (Q3 is bipartite)
    # 4. Check diameter?
    # Actually, there is only one 3-regular bipartite graph on 4+4 vertices with girth 4?
    # No, K4,4 is 4-regular.
    # The 3-regular bipartite graphs on 4+4 vertices:
    # Q3 is one. Is there another?
    # The complement of Q3 in K4,4 is also 3-regular.
    # Are Q3 and its complement isomorphic?
    # Q3 has 4 disjoint 4-cycles.
    # It is the unique 3-regular bipartite graph on 8 vertices?
    # Let's check.
    # Number of 3-regular bipartite graphs on 4+4 vertices.
    # It turns out Q3 is the only one (up to isomorphism).
    # So, G[S] contains a Q3 iff G[S] is isomorphic to Q3?
    # NO. G[S] could have MORE edges.
    # "Contains a subgraph isomorphic to Q3".
    # If G[S] has 13 edges, it contains a subgraph with 12 edges.
    # That subgraph must be Q3.
    # So we need to check if G[S] *contains* a Q3.
    
    # So, for a given G[S], we need to see if there exists a set of 12 edges
    # forming a Q3.
    # This is equivalent to: Can we remove (12 - 12)=0 edges? No.
    # We need to find if the edge set E_S is a superset of some Q3 edge set.
    
    # Back to the 840 patterns idea.
    # We need the 840 edge sets of Q3 in K8.
    
    def get_q3_edge_sets():
        # Vertices 0-7
        # Standard cube edges:
        # (0,1), (0,2), (0,4)
        # (1,3), (1,5), (1,6) -- wait, let's generate properly.
        # Vertex i (0..7) corresponds to binary (i//4, (i//2)%2, i%2)?
        # Let's use:
        # 0: 000
        # 1: 001
        # 2: 010
        # 3: 011
        # 4: 100
        # 5: 101
        # 6: 110
        # 7: 111
        # Edge if popcount(i^j) == 1.
        
        edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    edges.add((i, j))
        
        # This is the standard Q3.
        # But we need all isomorphic copies in K8?
        # No, the "pattern" is the set of edges in the abstract K8.
        # If G[S] is isomorphic to Q3, then the set of edges in G[S]
        # (mapped to 0..7) must be exactly the edges of some permutation of Q3.
        # But G[S] might have extra edges.
        # So we need to check if E_S contains ANY of the 840 edge sets.
        
        # Generate all 840 sets.
        # Permute vertices 0..7.
        # There are 8! permutations.
        # Each Q3 has 48 automorphisms.
        # So 8!/48 = 840 unique edge sets.
        
        base_edges = [(i, j) for i in range(8) for j in range(i+1, 8) if bin(i ^ j).count('1') == 1]
        
        unique_sets = set()
        # We can just generate all permutations of range(8) and map the base edges.
        # To avoid generating 40320, we can just generate all subsets of size 12? No.
        # Let's just generate all 8! and add to a set of frozensets.
        # 40320 iterations is fast.
        
        for perm in itertools.permutations(range(8)):
            # perm maps original vertex k to new vertex perm[k]
            # Wait, we want to map the "labels" 0..7 to the "positions" in S.
            # Let's say the vertices in S are v0..v7.
            # We assign them labels 0..7 via a permutation p.
            # The edges in G[S] are (v_i, v_j).
            # We check if the set of pairs (i,j) such that (v_i, v_j) in E
            # contains a Q3 pattern.
            
            # Actually, it's easier to:
            # 1. Map S to 0..7.
            # 2. Get the set of edges E_S in 0..7 space.
            # 3. Check if E_S contains any Q3 pattern.
            
            # The Q3 patterns are the sets of edges (i,j) in 0..7 space
            # that form a Q3.
            # So we just need the set of all 840 edge sets in 0..7 space.
            
            new_edges = frozenset((perm[i], perm[j]) for i, j in base_edges)
            # Normalize edges (min, max)
            new_edges = frozenset(tuple(sorted(e)) for e in new_edges)
            unique_sets.add(new_edges)
            
        return list(unique_sets)

    q3_patterns = get_q3_edge_sets()
    
    # Initialize graph as K16
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            edges.add((i, j))
    
    # Helper to check if a subset S (tuple of 8 vertices) contains a Q3
    def contains_q3(S):
        # Map S to 0..7
        # S[0]->0, S[1]->1, ...
        # Get edges in G[S]
        local_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if (u, v) in edges or (v, u) in edges:
                    local_edges.add((i, j))
        
        # Check if local_edges contains any pattern
        for pattern in q3_patterns:
            if pattern.issubset(local_edges):
                return True
        return False

    # Main loop
    # We remove edges until no Q3 is found.
    # To speed up, we can keep a list of "bad" subsets.
    # Initially, all subsets that contain a Q3 in K16 are bad.
    # In K16, every 8-subset contains a Q3 (since K8 contains Q3).
    # So all 12870 are bad.
    
    bad_subsets = [s for s in all_subsets if contains_q3(s)]
    
    while bad_subsets:
        if time.time() > deadline:
            break
        
        # Pick a bad subset
        S = bad_subsets[0]
        bad_subsets.pop(0)
        
        # Verify it's still bad (might have been fixed by previous removals)
        if not contains_q3(S):
            continue
            
        # Find a Q3 in S to remove an edge from
        # We need to find which pattern is present.
        # Map S to 0..7
        local_edges = set()
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                if (u, v) in edges or (v, u) in edges:
                    local_edges.add((i, j))
                    
        found_pattern = None
        for pattern in q3_patterns:
            if pattern.issubset(local_edges):
                found_pattern = pattern
                break
        
        if found_pattern is None:
            continue
            
        # Remove an edge from found_pattern
        # Convert local edge to global edge
        # Pick the first edge in the pattern
        local_u, local_v = list(found_pattern)[0]
        global_u, global_v = S[local_u], S[local_v]
        e_to_remove = (min(global_u, global_v), max(global_u, global_v))
        
        edges.remove(e_to_remove)
        
        # Update bad_subsets
        # Remove S from bad_subsets? We already popped it.
        # We need to check other subsets that might have become good.
        # This is expensive to update incrementally.
        # Just re-scan?
        # If we re-scan 12870 subsets every iteration, and we do 50 iterations,
        # 50 * 12870 * 840 checks = 500M checks. Too slow.
        
        # Optimization:
        # Only subsets containing e_to_remove could have changed status.
        # Subsets NOT containing e_to_remove are unchanged.
        # If a subset was bad before, and doesn't contain e_to_remove, it is still bad?
        # No, it might have been bad because of a Q3 that used e_to_remove?
        # No, if it doesn't contain e_to_remove, its edge set is unchanged.
        # So if it was bad, it is still bad.
        # If it was good, it is still good.
        # So we only need to re-evaluate subsets that CONTAIN e_to_remove.
        
        # How many subsets contain a specific edge?
        # An edge (u,v) is in 8-subsets S if u,v in S.
        # We need to pick 6 more vertices from the remaining 14.
        # C(14, 6) = 3003 subsets.
        # 3003 * 840 checks = 2.5M checks per iteration.
        # 50 iterations * 2.5M = 125M checks.
        # This is feasible in Python (approx 10-20s).
        
        # Identify subsets containing e_to_remove
        # We need a mapping from edge to subsets.
        # Precompute: edge_to_subsets = { edge: [S1, S2, ...] }
        
        # Let's precompute this.
        pass # We will do this in the actual code below

    # Convert set to list of lists
    return [[u, v] for u, v in edges]
