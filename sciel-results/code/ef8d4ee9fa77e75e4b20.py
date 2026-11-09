"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Map vertices to indices for easier manipulation? 
    # Actually, just use 0..n-1 directly.
    
    # Helper: Check if a set of edges can be covered by a biclique [L, R]
    # A biclique [L, R] covers all edges (u,v) where u in L, v in R.
    # For this to be valid, all such pairs must be in the graph.
    # i.e., for all u in L, (adj[u] & R_mask) == R_mask (if we consider simple graph)
    # Wait, the biclique definition says "Every cross pair must be a host edge".
    # So if u in L and v in R, edge (u,v) MUST exist.
    
    def get_biclique_edges(L, R):
        # Returns list of edges covered by biclique [L, R]
        # And verifies validity
        edge_set = set()
        L_set = set(L)
        R_set = set(R)
        for u in L:
            for v in R:
                if u == v: continue
                # Check if edge exists
                if (adj[u] >> v) & 1:
                    edge_set.add((min(u,v), max(u,v)))
                else:
                    return None # Invalid biclique
        return edge_set

    # Strategy: Greedy with local search
    # 1. Initialize with a valid partition (e.g. star partition or greedy)
    # 2. Refine using 1-move (replace one biclique with a better one) or 2-move (merge/split)
    
    # Let's implement a strong greedy first.
    # State: list of remaining edges.
    
    def greedy_partition(randomness=0.0):
        # Returns a list of bicliques
        remaining_edges = set((min(u,v), max(u,v)) for u,v in edges)
        if not remaining_edges:
            return []
        
        partition = []
        
        # Precompute degrees for current remaining edges
        deg = [0] * n
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find a good seed edge
            # Pick edge with max sum of degrees, or random if randomness > 0
            if randomness > 0 and random.random() < randomness:
                # Pick a random edge
                e = random.choice(list(remaining_edges))
            else:
                # Pick edge with max (deg[u] + deg[v])
                max_score = -1
                best_e = None
                for e in remaining_edges:
                    u, v = e
                    score = deg[u] + deg[v]
                    if score > max_score:
                        max_score = score
                        best_e = e
                e = best_e
            
            if e is None:
                break
                
            u, v = e
            
            # Start biclique with L={u}, R={v}
            L = [u]
            R = [v]
            
            # Expand greedily
            # Try to add vertices to L or R that are connected to all vertices on the opposite side
            # and are incident to some remaining edge.
            
            changed = True
            while changed:
                changed = False
                
                # Try adding to L
                # Candidate w for L: w must be connected to all in R.
                # Also, w must have at least one edge to R that is in remaining_edges?
                # Actually, if we add w to L, we cover all edges (w, r) for r in R.
                # These edges MUST be in remaining_edges.
                # So for all r in R, (w, r) must be in remaining_edges.
                
                # Let's find candidates for L
                # A vertex w is a candidate for L if:
                # 1. w is not in L
                # 2. For all r in R, edge (w, r) exists in graph
                # 3. For all r in R, edge (w, r) is in remaining_edges (so we can cover it)
                
                # Efficient check using bitmasks
                R_mask = 0
                for r in R:
                    R_mask |= (1 << r)
                
                # Candidates for L:
                # Must have (adj[w] & R_mask) == R_mask
                # AND for all r in R, (w, r) is in remaining_edges.
                
                # Let's just iterate vertices. n=30 is small.
                for w in range(n):
                    if w in L: continue
                    # Check graph connectivity
                    if (adj[w] & R_mask) != R_mask:
                        continue
                    # Check remaining edges
                    valid = True
                    for r in R:
                        e_wr = (min(w,r), max(w,r))
                        if e_wr not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        L.append(w)
                        changed = True
                        # Update degrees? 
                        # We cover edges (w, r) for all r in R.
                        for r in R:
                            e_wr = (min(w,r), max(w,r))
                            if e_wr in remaining_edges:
                                remaining_edges.remove(e_wr)
                                deg[w] -= 1
                                deg[r] -= 1
                        # Note: deg updates are approximate if we don't track carefully, 
                        # but for greedy selection of NEXT seed, it's okay.
                        # Actually, we should update deg for the vertices in L and R properly.
                        # Let's just recalculate deg if we want to be safe, but n is small.
                        # Or just trust the decrement.
                        break # Restart loop to re-evaluate candidates with new L/R? 
                        # Actually, adding one vertex changes the constraints for others.
                        # Restarting the while loop is safer.
                
                if changed:
                    continue
                
                # Try adding to R
                L_mask = 0
                for l in L:
                    L_mask |= (1 << l)
                    
                for w in range(n):
                    if w in R: continue
                    if (adj[w] & L_mask) != L_mask:
                        continue
                    valid = True
                    for l in L:
                        e_wl = (min(w,l), max(w,l))
                        if e_wl not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        R.append(w)
                        changed = True
                        for l in L:
                            e_wl = (min(w,l), max(w,l))
                            if e_wl in remaining_edges:
                                remaining_edges.remove(e_wl)
                                deg[w] -= 1
                                deg[l] -= 1
                        break
                
                if changed:
                    continue
            
            # Add the biclique to partition
            # Clean up L and R (remove vertices that don't contribute edges? 
            # Actually, if a vertex in L has no edges to R in remaining_edges, 
            # it wouldn't have been added. But maybe it became redundant?
            # No, we only add if it covers remaining edges.
            # However, we should ensure L and R are non-empty and disjoint.)
            
            # Disjointness: 
            # If a vertex is in both L and R, it's a problem.
            # Our expansion logic: 
            # When adding to L, we check w not in L. But w could be in R.
            # If w is in R, and we add to L, then w is in both.
            # Is that allowed? "nonempty disjoint sides". NO.
            # So we must ensure L and R are disjoint.
            
            # Let's fix the expansion to prevent overlap.
            # Actually, if w is in R, and we try to add to L:
            # We require (w, r) in remaining_edges for all r in R.
            # If w is in R, then (w, w) is not an edge. 
            # My check `(adj[w] & R_mask) == R_mask` would fail if w is in R_mask 
            # because adj[w] doesn't have bit w set. 
            # So w cannot be added to L if w is in R.
            # Similarly for R.
            # So L and R should naturally be disjoint.
            
            if L and R:
                partition.append([L, R])
            else:
                # Fallback: single edge biclique
                partition.append([[u], [v]])
                remaining_edges.remove((min(u,v), max(u,v)))
                deg[u] -= 1
                deg[v] -= 1
                
        # If we timed out, we need to complete the partition with single edges
        if remaining_edges:
            for e in remaining_edges:
                partition.append([[e[0]], [e[1]]])
                
        return partition

    # Local Search Refinement
    # Try to improve the partition by merging or replacing bicliques.
    
    def refine(partition):
        if time.time() > deadline:
            return partition
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try merging two bicliques
            # If we have B1=[L1, R1] and B2=[L2, R2], 
            # can we merge them into one biclique?
            # A merged biclique would be [L1 U L2, R1 U R2] ?
            # No, that would require all cross edges between L1 U L2 and R1 U R2 to exist.
            # Specifically, edges between L1 and R2, and L2 and R1 must exist.
            # And all those edges must be covered by the union of B1 and B2 edges.
            # Actually, if we merge, the new biclique covers:
            # (L1 x R1) U (L2 x R2) U (L1 x R2) U (L2 x R1)
            # The original partition covered (L1 x R1) U (L2 x R2).
            # So we can only merge if the "new" edges (L1 x R2) and (L2 x R1) 
            # are ALREADY covered by other bicliques? No, that would be double covering.
            # The task requires EXACT partition. 
            # So we can only merge if the new edges are EMPTY (i.e., no cross edges exist) 
            # OR if the new edges are a subset of edges already covered by B1 or B2?
            # No, if an edge (u, v) with u in L1, v in R2 exists in the graph, 
            # it must be covered by exactly one biclique.
            # If we merge B1 and B2, the new biclique covers it.
            # If it was already covered by another biclique B3, we have a conflict.
            # So we can merge B1 and B2 ONLY IF:
            # 1. All edges in (L1 x R2) and (L2 x R1) exist in the graph.
            # 2. None of these edges are covered by any OTHER biclique in the partition.
            # 3. The edges in (L1 x R2) and (L2 x R1) are NOT covered by B1 or B2 (obviously).
            # Wait, if an edge (u, v) is in (L1 x R2), it wasn't covered by B1 (v not in R1) 
            # and wasn't covered by B2 (u not in L2).
            # So it must be covered by some other biclique Bk.
            # If we merge B1 and B2, the new biclique covers (u, v).
            # This would cause (u, v) to be covered by both Bk and the new merged biclique.
            # UNLESS Bk is removed or adjusted.
            # This is complex. 
            # 
            # Alternative: 1-move replacement.
            # Replace a biclique B with a set of bicliques B' that cover the same edges?
            # Or replace a biclique B with a LARGER biclique B' that covers B's edges 
            # plus some edges from other bicliques, and then remove those edges from the others?
            
            # Let's try a simpler heuristic: 
            # For each biclique B, try to expand it by taking edges from other bicliques.
            # If B=[L, R], can we add a vertex w to L such that all edges (w, r) for r in R 
            # are currently covered by OTHER bicliques, and we can "steal" them?
            # If we steal edges from other bicliques, those other bicliques must remain valid.
            # Removing edges from a biclique doesn't invalidate it (a subset of a biclique is a biclique? 
            # No, a biclique is defined by its vertex sets. If we remove edges, the vertex sets might 
            # still form a biclique, but the partition must cover ALL edges.
            # If we move an edge (w, r) from Bk to B, we must ensure:
            # 1. (w, r) is in B's coverage (w in L', r in R).
            # 2. Bk still covers all its remaining edges.
            # 3. No other biclique covers (w, r).
            
            # This is getting complicated. Let's stick to a robust greedy with multiple restarts 
            # and a simple local search that tries to merge bicliques if possible.
            
            # Let's implement a "try to merge" check.
            for i in range(len(partition)):
                for j in range(i+1, len(partition)):
                    L1, R1 = partition[i]
                    L2, R2 = partition[j]
                    
                    # Check if we can merge
                    # New L = L1 U L2, New R = R1 U R2
                    # Must be disjoint: (L1 U L2) & (R1 U R2) == empty
                    if set(L1) & set(R2) or set(L2) & set(R1) or set(L1) & set(R2) or set(L2) & set(R1):
                         # Wait, L1 and R1 are disjoint. L2 and R2 are disjoint.
                         # We need (L1 U L2) and (R1 U R2) to be disjoint.
                         # This requires L1 & R2 == empty, L2 & R1 == empty.
                         # (L1 & R1 is already empty, L2 & R2 is already empty)
                         pass
                     
                    # Check cross edges existence
                    valid_merge = True
                    # Edges to check: L1 x R2 and L2 x R1
                    for u in L1:
                        for v in R2:
                            if u == v: continue
                            if not ((adj[u] >> v) & 1):
                                valid_merge = False
                                break
                        if not valid_merge: break
                    if not valid_merge: continue
                    
                    for u in L2:
                        for v in R1:
                            if u == v: continue
                            if not ((adj[u] >> v) & 1):
                                valid_merge = False
                                break
                        if not valid_merge: break
                    if not valid_merge: continue
                    
                    # Check if these cross edges are covered by other bicliques
                    # If they are, we can't merge (would double cover).
                    # If they are NOT covered, we can't merge (would leave them uncovered? 
                    # No, the merged biclique would cover them. But they must be in the graph.
                    # If they are in the graph, they MUST be covered by SOME biclique.
                    # If they are not covered by B1 or B2, they must be covered by Bk.
                    # If we merge, B1 U B2 covers them. Bk also covers them. Double cover.
                    # So we can ONLY merge if the cross edges are EMPTY (no edges exist) 
                    # OR if the cross edges are covered by B1 or B2? Impossible.
                    # So merging is only possible if there are NO edges between L1-R2 or L2-R1.
                    
                    # Let's check if there are any edges between L1 and R2
                    has_cross_edge = False
                    for u in L1:
                        for v in R2:
                            if u != v and ((adj[u] >> v) & 1):
                                has_cross_edge = True
                                break
                        if has_cross_edge: break
                    if has_cross_edge: continue
                    
                    for u in L2:
                        for v in R1:
                            if u != v and ((adj[u] >> v) & 1):
                                has_cross_edge = True
                                break
                        if has_cross_edge: break
                    if has_cross_edge: continue
                    
                    # If no cross edges, we can merge!
                    # New biclique: [L1 U L2, R1 U R2]
                    new_L = L1 + L2
                    new_R = R1 + R2
                    
                    # Ensure disjointness
                    if set(new_L) & set(new_R):
                        continue
                        
                    # Replace partition[i] and partition[j] with the merged one
                    partition[i] = [new_L, new_R]
                    partition.pop(j)
                    improved = True
                    break
            if improved:
                continue
                
            # 2. Try to expand a biclique by stealing edges from another
            # This is harder. Let's try a simple heuristic:
            # For each biclique B, try to add a vertex to L or R.
            # If we add w to L, we need edges (w, r) for all r in R.
            # These edges must be covered by OTHER bicliques.
            # If we "steal" them, we remove them from those other bicliques.
            # The other bicliques must remain valid (which they do, as subsets of bicliques are bicliques? 
            # No, a biclique is a set of vertices. If we remove edges, the vertex set might still be a biclique.
            # But the partition must cover ALL edges.
            # If we move an edge, it's no longer covered by the old biclique.
            # This is a valid move IF the old biclique still covers all its remaining assigned edges.
            
            # Let's skip this for now and rely on greedy + merge.
            
        return partition

    best_partition = None
    best_count = float('inf')
    
    # Multi-start
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        randomness = i / num_starts
        partition = greedy_partition(randomness)
        partition = refine(partition)
        
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = partition
            
        # Early stop if we hit a very good score (e.g. < 10)
        if best_count <= 5:
            break

    if best_partition is None:
        # Fallback to seed
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    # Convert to required format: list of [left, right] with Python ints
    result = []
    for b in best_partition:
        L = [int(x) for x in b[0]]
        R = [int(x) for x in b[1]]
        result.append([L, R])
        
    return result
# EVOLVE-BLOCK-END
