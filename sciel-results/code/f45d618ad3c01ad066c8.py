import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precomputation
    # 1. Edge indices
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # 2. Subsets of size 8
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # 3. Local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # 4. Generate Q3 masks for K8
    # A Q3 is a subgraph of K8. We represent a specific set of 12 edges that form a Q3.
    # We iterate over all ways to partition 8 vertices into two sets of 4 (A and B)
    # and all bijections between them? No, that's for isomorphism.
    # We need to find all edge sets E' subset of E(K8) such that (V8, E') is isomorphic to Q3.
    # Q3 has 12 edges.
    # There are C(28, 12) possible edge sets, which is huge.
    # Better: Iterate over all graph isomorphisms of Q3 to K8.
    # Number of automorphisms of Q3 is 48.
    # Number of embeddings of Q3 in K8 is 8! / 48 = 40320 / 48 = 840.
    # So there are 840 distinct Q3 subgraphs in K8.
    
    # Let's generate them.
    # Standard Q3 vertices: 0..7. Edges: (i, j) if i^j is power of 2.
    # Let's pick one canonical Q3 edge set.
    q3_canonical_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:
                q3_canonical_edges.append((i, j))
    
    # Generate all permutations of 8 vertices that map the canonical Q3 to a subgraph of K8
    # Since K8 is complete, any permutation gives a valid Q3 subgraph.
    # We just need the set of edges.
    # Two permutations that differ by an automorphism of Q3 will produce the same edge set.
    
    q3_masks_for_k8 = []
    seen_masks = set()
    
    # To speed up, we can fix the image of vertex 0 to be 0, 1, or 2?
    # No, just iterate all permutations? 8! = 40320. That's small.
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (u, v) in q3_canonical_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)
            
    # 5. Map global edges to subsets and local indices
    # edge_to_subsets[e] = list of subset indices
    # local_idx[sub_idx][e] = local bit position
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Helper to check if a 28-bit mask contains a Q3
    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    def can_add_edge(G, subset_masks, e_idx):
        """Check if edge e_idx can be added to graph G without creating Q3."""
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                return False
        return True

    def add_edge_to_state(G, subset_masks, e_idx):
        """Add edge e_idx to graph G and update subset_masks."""
        G |= (1 << e_idx)
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            subset_masks[s_idx] |= (1 << l_pos)
        return G

    def greedy_construct(ordering):
        """Greedy construction with given edge ordering."""
        G = 0
        subset_masks = [0] * num_subsets
        
        for e_idx in ordering:
            if time.time() > deadline - 5:
                break
            if can_add_edge(G, subset_masks, e_idx):
                G = add_edge_to_state(G, subset_masks, e_idx)
        
        return G, subset_masks

    def local_search_with_swaps(G, subset_masks, max_iterations=50):
        """
        Local search: try removing edges incident to blocked vertices and adding others.
        A vertex is "blocked" if removing its incident edges allows many new edges to be added.
        """
        current_G = G
        current_subset_masks = subset_masks[:]
        current_count = current_G.bit_count()
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find "blocked" vertices: vertices where removing incident edges could allow more additions
            # For each vertex, count how many edges incident to it are present,
            # and how many missing edges incident to it could be added if we removed some present edges.
            
            best_gain = 0
            best_swap = None  # (vertex, edges_to_remove, edges_to_add)
            
            for v in range(n):
                # Get edges incident to v
                incident_present = []
                incident_missing = []
                for u in range(n):
                    if u == v:
                        continue
                    e_idx = edge_index[v][u]
                    if e_idx < 0:
                        continue
                    if (current_G >> e_idx) & 1:
                        incident_present.append(e_idx)
                    else:
                        incident_missing.append(e_idx)
                
                if len(incident_present) == 0:
                    continue
                
                # Try removing 1-2 edges incident to v and see if we can add more
                for num_remove in [1, 2]:
                    if len(incident_present) < num_remove:
                        continue
                    
                    # Try subsets of incident_present of size num_remove
                    for remove_set in itertools.combinations(incident_present, num_remove):
                        # Trial: remove these edges
                        trial_G = current_G
                        trial_subset_masks = current_subset_masks[:]
                        for e in remove_set:
                            trial_G &= ~(1 << e)
                            for s_idx in edge_to_subsets[e]:
                                l_pos = local_idx[s_idx][e]
                                trial_subset_masks[s_idx] &= ~(1 << l_pos)
                        
                        # Try to add missing edges incident to v (and maybe others)
                        added = 0
                        added_edges = []
                        for e_add in incident_missing:
                            if can_add_edge(trial_G, trial_subset_masks, e_add):
                                trial_G = add_edge_to_state(trial_G, trial_subset_masks, e_add)
                                added += 1
                                added_edges.append(e_add)
                        
                        gain = added - num_remove
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (v, remove_set, added_edges, trial_G, trial_subset_masks)
            
            if best_swap is None or best_gain <= 0:
                break
            
            # Apply the best swap
            v, remove_set, added_edges, new_G, new_subset_masks = best_swap
            current_G = new_G
            current_subset_masks = new_subset_masks
            current_count = current_G.bit_count()
        
        return current_G, current_subset_masks

    def systematic_polish(G, subset_masks):
        """Systematic pass: try adding any edge that can be added."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for e_idx in range(total_edges):
                if time.time() > deadline - 5:
                    break
                if (G >> e_idx) & 1:
                    continue
                if can_add_edge(G, subset_masks, e_idx):
                    G = add_edge_to_state(G, subset_masks, e_idx)
                    improved = True
        return G, subset_masks

    best_G = 0
    best_count = 0
    best_subset_masks = None

    # Strategy 1: Multiple greedy passes with different orderings
    orderings = []
    
    # Random orderings
    for _ in range(20):
        edges = list(range(total_edges))
        random.shuffle(edges)
        orderings.append(edges)
    
    # Heuristic orderings
    # Order by: edges that appear in fewer subsets first? (all same)
    # Order by: random but with bias toward certain vertex pairs
    for trial in range(10):
        edges = list(range(total_edges))
        # Bias: prefer edges between lower-indexed vertices
        random.shuffle(edges)
        # Sort with some randomness
        edges.sort(key=lambda e: (edge_list[e][0] + random.random()*2, edge_list[e][1] + random.random()*2))
        orderings.append(edges)
    
    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        
        G, subset_masks = greedy_construct(ordering)
        count = G.bit_count()
        
        # Local search
        G, subset_masks = local_search_with_swaps(G, subset_masks)
        count = G.bit_count()
        
        # Systematic polish
        G, subset_masks = systematic_polish(G, subset_masks)
        count = G.bit_count()
        
        if count > best_count:
            best_count = count
            best_G = G
            best_subset_masks = subset_masks[:]

    # Final systematic polish on best
    if best_subset_masks is not None:
        best_G, best_subset_masks = systematic_polish(best_G, best_subset_masks)
        best_count = best_G.bit_count()

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
