"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin for 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation for speed
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add(frozenset((u, v)))
        
    all_edges = list(edge_set)
    
    # Helper: Check if a biclique (L, R) is valid (all cross edges exist in graph)
    # Note: We only care if it's a biclique in the original graph, not just remaining.
    # But for covering, we just need to know which edges it covers.
    def get_edges_covered(L, R):
        covered = set()
        for u in L:
            for v in R:
                covered.add(frozenset((u, v)))
        return covered

    # Helper: Generate a large biclique starting from a seed pair or vertex
    def expand_biclique(seed_u, seed_v, max_iters=100):
        """
        Greedily expand a biclique starting with u in Left and v in Right.
        Returns (Left, Right)
        """
        L = {seed_u}
        R = {seed_v}
        
        # Iterate to fixpoint
        changed = True
        while changed and (time.time() - t0) < deadline:
            changed = False
            
            # Try to add vertices to Left
            # A vertex w can join L if it is connected to ALL vertices in R
            # And w is not already in R
            candidates_L = []
            # We can only add w if w is connected to all in R
            # Optimization: Intersection of neighbors of all in R
            common_neighbors_R = (1 << n) - 1
            for v in R:
                common_neighbors_R &= adj[v]
                
            # Candidates must be in common_neighbors_R, not in R, and not in L
            # Also, to be a valid biclique in the graph, w must be connected to all in R.
            # We don't strictly require w to be connected to all in L (within-side edges allowed/ignored for cross constraint, 
            # but wait: "Every cross pair must be a host edge". 
            # If w is in L, and x is in R, edge (w,x) MUST exist.
            # So yes, w must be connected to all in R.
            
            # Also, we typically want to avoid putting a vertex in L if it's in R, as that splits it.
            # But the definition says L and R are disjoint.
            
            for w in range(n):
                if w in L or w in R:
                    continue
                if common_neighbors_R & (1 << w):
                    candidates_L.append(w)
            
            # Sort by degree in the "potential" sense or just random/greedy
            # Greedy: pick the one with highest degree in the remaining graph? 
            # Or just pick one that allows most further expansion?
            # Simple heuristic: pick random or highest degree in original graph
            if candidates_L:
                # Pick one with max degree in original graph to keep things dense
                best_w = max(candidates_L, key=lambda x: bin(adj[x]).count('1'))
                L.add(best_w)
                changed = True
            
            # Try to add vertices to Right
            common_neighbors_L = (1 << n) - 1
            for u in L:
                common_neighbors_L &= adj[u]
                
            candidates_R = []
            for w in range(n):
                if w in L or w in R:
                    continue
                if common_neighbors_L & (1 << w):
                    candidates_R.append(w)
            
            if candidates_R:
                best_w = max(candidates_R, key=lambda x: bin(adj[x]).count('1'))
                R.add(best_w)
                changed = True
                
        return L, R

    # Step 1: Generate a pool of candidate bicliques
    candidates = []
    
    # 1a. Maximal bicliques anchored at each edge
    for e in all_edges:
        u, v = tuple(e)
        L, R = expand_biclique(u, v)
        if L and R:
            # Normalize for deduplication
            # A biclique is defined by its edge set
            edges_covered = get_edges_covered(L, R)
            # Store as tuple of frozensets for hashability
            candidates.append((frozenset(L), frozenset(R), edges_covered))
            
    # 1b. Random sampling to find diverse bicliques
    random.seed(42)
    for _ in range(500):
        if (time.time() - t0) > deadline - 5: break
        u, v = random.sample(range(n), 2)
        if frozenset((u, v)) in edge_set:
            L, R = expand_biclique(u, v)
            if L and R:
                edges_covered = get_edges_covered(L, R)
                candidates.append((frozenset(L), frozenset(R), edges_covered))
                
    # Deduplicate candidates by their covered edge set
    unique_candidates = {}
    for L, R, edges_cov in candidates:
        key = frozenset(edges_cov)
        if key not in unique_candidates:
            unique_candidates[key] = (L, R, edges_cov)
            
    cand_list = list(unique_candidates.values())
    
    # If we have too many, keep the largest ones
    if len(cand_list) > 1000:
        cand_list.sort(key=lambda x: len(x[2]), reverse=True)
        cand_list = cand_list[:1000]
        
    # Step 2: Greedy Set Cover
    # We want to select a subset of cand_list that covers all_edges with minimum count.
    # Greedy: repeatedly pick the candidate that covers the most *uncovered* edges.
    
    remaining_edges = set(all_edges)
    selected_bicliques = []
    
    while remaining_edges and (time.time() - t0) < deadline - 10:
        best_cand = None
        best_gain = 0
        
        for L, R, edges_cov in cand_list:
            gain = len(edges_cov & remaining_edges)
            if gain > best_gain:
                best_gain = gain
                best_cand = (L, R, edges_cov)
                
        if best_cand is None or best_gain == 0:
            # Fallback: single edges
            for e in remaining_edges:
                u, v = tuple(e)
                selected_bicliques.append((frozenset([u]), frozenset([v]), frozenset([e])))
            break
            
        selected_bicliques.append(best_cand)
        remaining_edges -= best_cand[2]
        
    # Step 3: Local Search Refinement
    # Try to replace 1 biclique with 2 others if it reduces total count? No, that increases count.
    # We want to minimize count.
    # Move: If two bicliques in the solution can be merged into one, do it.
    # Move: If one biclique in the solution is redundant (its edges are covered by others), remove it.
    # Move: Swap a biclique B1 with B2 if B2 covers all edges of B1 (and maybe more) and B2 is not already in solution?
    # Actually, standard 2-opt for set cover:
    # Try replacing a set of k bicliques with a set of k-1 bicliques that covers the same union of edges.
    
    # Simple Redundancy Check
    changed = True
    while changed and (time.time() - t0) < deadline - 5:
        changed = False
        new_selected = []
        for i, (L, R, edges_cov) in enumerate(selected_bicliques):
            # Check if edges_cov is covered by the union of all other selected bicliques
            other_cover = set()
            for j, (_, _, e_cov_j) in enumerate(selected_bicliques):
                if i != j:
                    other_cover |= e_cov_j
            if edges_cov.issubset(other_cover):
                # Redundant, skip adding it
                changed = True
            else:
                new_selected.append((L, R, edges_cov))
        if changed:
            selected_bicliques = new_selected
            
    # Try Swapping: Replace one biclique with two? No, we want fewer.
    # Try Merging: If two bicliques B1, B2 form a valid biclique B3, replace them with B3.
    # This is hard to check generally.
    
    # Better Local Search: 
    # 1. Identify "weak" bicliques (small size).
    # 2. Try to replace a weak biclique with a combination of 2 stronger ones? No, count increases.
    # 3. Try to replace 2 bicliques with 1?
    #    If Union(Edges(B1), Edges(B2)) is a biclique, replace B1, B2 with that biclique.
    
    merged = True
    while merged and (time.time() - t0) < deadline - 2:
        merged = False
        new_selected = list(selected_bicliques)
        for i in range(len(new_selected)):
            for j in range(i+1, len(new_selected)):
                L1, R1, E1 = new_selected[i]
                L2, R2, E2 = new_selected[j]
                union_edges = E1 | E2
                # Check if union_edges forms a biclique
                # We need to find L, R such that all edges in union_edges are in L x R and no other edges?
                # Actually, just need to find L, R that cover union_edges and are valid bicliques.
                # Simple check: Does the subgraph induced by vertices in union_edges have a biclique cover of size 1?
                # i.e., is the bipartite graph of union_edges a complete bipartite graph?
                # Not necessarily, because union_edges might be disconnected or not complete.
                # But if we can find ANY biclique covering union_edges, we can merge.
                # Let's try to construct one.
                vertices_involved = set()
                for e in union_edges:
                    u, v = tuple(e)
                    vertices_involved.add(u)
                    vertices_involved.add(v)
                
                # Try to find a biclique covering union_edges
                # Heuristic: Pick a vertex u from union_edges, try to expand
                found = False
                for u in vertices_involved:
                    # Try u in Left
                    L, R = expand_biclique(u, next(iter(vertices_involved - {u}))) # Bad seed
                    # Better: just check if the set of edges is contained in a known candidate?
                    # Or just try to expand from u and see if it covers union_edges.
                    for v in vertices_involved:
                        if u == v: continue
                        L_try, R_try = expand_biclique(u, v)
                        if get_edges_covered(L_try, R_try) >= union_edges:
                            # Found a biclique covering both
                            new_selected[i] = (frozenset(L_try), frozenset(R_try), union_edges) # Approx
                            # Actually, we should use the exact edges covered by L_try, R_try that are in union_edges?
                            # No, the biclique covers ALL edges in L_try x R_try.
                            # If L_try x R_try contains edges NOT in union_edges, that's fine, as long as it covers union_edges.
                            # But wait, the partition must cover ALL graph edges exactly once.
                            # If we merge B1 and B2 into B3, B3 covers E1 U E2.
                            # But B3 might cover extra edges E3 that were covered by B4, B5...
                            # This breaks the partition property!
                            # The edges must be partitioned. 
                            # So we cannot just merge if the new biclique covers extra edges that are already assigned to others.
                            # We can only merge if the new biclique's edge set is EXACTLY E1 U E2? 
                            # Or if we re-assign the extra edges? That's complex.
                            # So, simple merging is dangerous.
                            pass
                    break
                if found:
                    break
            if merged:
                break
        # If no merge found, break
        
    # Final Fallback: Ensure all edges are covered
    covered = set()
    for L, R, E in selected_bicliques:
        covered |= E
        
    if covered != set(all_edges):
        # Add single edges for missing
        for e in set(all_edges) - covered:
            u, v = tuple(e)
            selected_bicliques.append((frozenset([u]), frozenset([v]), frozenset([e])))

    # Construct Result
    result = []
    for L, R, E in selected_bicliques:
        left = sorted([int(x) for x in L])
        right = sorted([int(x) for x in R])
        if left and right:
            result.append([left, right])
            
    return result
# EVOLVE-BLOCK-END
