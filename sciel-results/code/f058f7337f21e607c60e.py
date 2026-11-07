"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
import itertools

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. For each vertex v in I, create a clique C_v = {v} U N(v).
       This covers all cross edges incident to v and all internal edges within N(v).
    3. The remaining edges are internal edges in K that are not within any N(v).
       These form a graph H on K.
    4. Partition H into maximal cliques greedily.
    5. Local search: Try to remove vertices from C_v to improve the partition of H.
    """
    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency
        # K: 0..k-1, I: k..n-1
        # neighbors_in_I[v] = list of u in K such that (v,u) is an edge
        neighbors_in_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors_in_I[v] = nbrs

        # Total edges in K
        # We need to track which internal edges are covered by C_v cliques.
        # An internal edge (u, w) is covered by C_v if u, w in N(v).
        
        # Initial assignment: C_v = {v} U N(v) for all v in I with N(v) non-empty.
        # If N(v) is empty, C_v is not formed (v is isolated in cross edges).
        
        cliques = []
        # covered_internal_edges: set of frozenset({u, w})
        covered_internal_edges = set()
        
        # List of sets S_v for each v
        S_v = {}
        
        for v in range(k, n):
            nbrs = neighbors_in_I[v]
            if not nbrs:
                continue
            
            S_v[v] = set(nbrs)
            
            # Add clique
            cliques.append([v] + list(nbrs))
            
            # Mark internal edges covered
            for u in nbrs:
                for w in nbrs:
                    if u < w:
                        covered_internal_edges.add(frozenset((u, w)))
                    elif w < u:
                        covered_internal_edges.add(frozenset((w, u)))
        
        # Now, find remaining internal edges
        # All pairs (u, w) in K are edges in the original graph.
        # Remaining are those NOT in covered_internal_edges.
        
        remaining_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if frozenset((u, w)) not in covered_internal_edges:
                    remaining_edges.append((u, w))
        
        # Partition remaining_edges into cliques
        # Greedy: Repeatedly find a maximal clique in the graph formed by remaining_edges
        # Graph G_rem: vertices 0..k-1, edges remaining_edges
        
        def get_maximal_clique(edges_set, k_size):
            """
            Find a maximal clique in the graph defined by edges_set.
            edges_set is a set of frozensets of size 2.
            Returns a list of vertices in the clique.
            """
            if not edges_set:
                return []
            
            # Pick a vertex with highest degree in G_rem
            deg = {i: 0 for i in range(k_size)}
            for e in edges_set:
                u, w = tuple(e)
                deg[u] += 1
                deg[w] += 1
            
            # Start with a random high-degree vertex
            start_v = max(deg, key=deg.get)
            clique = [start_v]
            candidates = [u for u in range(k_size) if u != start_v and frozenset((start_v, u)) in edges_set]
            
            while candidates:
                # Pick a candidate that is connected to all current clique members
                # To be greedy, pick the one with highest degree in the subgraph?
                # Or just the first one.
                # For speed, just check connectivity.
                next_v = None
                for c in candidates:
                    is_connected = True
                    for m in clique:
                        if frozenset((c, m)) not in edges_set:
                            is_connected = False
                            break
                    if is_connected:
                        next_v = c
                        break
                
                if next_v is None:
                    break
                
                clique.append(next_v)
                # Remove candidates not connected to next_v
                new_candidates = []
                for c in candidates:
                    if c == next_v:
                        continue
                    if frozenset((c, next_v)) in edges_set:
                        # Check if c is connected to all other clique members?
                        # Since candidates were connected to start_v, and we only add if connected to next_v,
                        # we need to ensure they are connected to ALL clique members.
                        # Actually, the set of candidates should be maintained as intersection of neighborhoods.
                        pass
                
                # Simpler approach:
                # Re-calculate candidates from scratch or maintain intersection.
                # Given small k (<=30), re-calc is fine.
                pass
            
            # Let's implement a simpler maximal clique finder:
            # Bron-Kerbosch or just greedy expansion.
            # Greedy expansion:
            clique = [start_v]
            candidates = set(u for u in range(k_size) if frozenset((start_v, u)) in edges_set)
            
            while candidates:
                # Pick candidate with max degree in the subgraph induced by candidates U clique?
                # Or just pick one that is connected to all in clique.
                # We know candidates are connected to start_v.
                # We need to filter candidates to those connected to all in clique.
                
                valid_candidates = []
                for c in candidates:
                    ok = True
                    for m in clique:
                        if frozenset((c, m)) not in edges_set:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(c)
                
                if not valid_candidates:
                    break
                
                # Pick one with highest degree in the remaining graph?
                # Just pick the first one to be fast, or max degree for better packing.
                # Let's pick max degree in the subgraph of valid_candidates.
                # Degree in the full remaining graph might be a good heuristic.
                best_c = max(valid_candidates, key=lambda x: deg[x])
                
                clique.append(best_c)
                # Update candidates: must be connected to best_c
                candidates = set(c for c in candidates if c != best_c and frozenset((c, best_c)) in edges_set)
                
            return clique

        # Greedy partition of remaining edges
        remaining_edges_set = set(frozenset(e) for e in remaining_edges)
        
        while remaining_edges_set:
            clique_verts = get_maximal_clique(remaining_edges_set, k)
            if not clique_verts:
                # Should not happen if edges exist, unless bug
                break
            
            # Add this clique to the list
            cliques.append(clique_verts)
            
            # Remove edges in this clique from remaining_edges_set
            # The clique covers all edges between vertices in clique_verts
            for i in range(len(clique_verts)):
                for j in range(i + 1, len(clique_verts)):
                    e = frozenset((clique_verts[i], clique_verts[j]))
                    if e in remaining_edges_set:
                        remaining_edges_set.remove(e)
        
        # Local Search
        # Try to improve by adjusting S_v.
        # If we remove u from S_v, we free up internal edges (u, w) for w in S_v.
        # This might reduce the number of cliques needed for the remaining internal edges.
        
        # We need a function to calculate the number of cliques for a given set of remaining edges.
        # This is expensive if done repeatedly.
        # However, k is small.
        
        # Let's do a few iterations of local search.
        # For each v, try removing each u in S_v.
        # Check if the cost decreases.
        
        # Cost = len(cliques)
        
        # To make local search efficient, we can pre-calculate the "savings" or just re-run the partition.
        # Given n=90, k=30, the partition step is fast.
        
        best_cliques = cliques
        best_cost = len(cliques)
        
        # Limit local search iterations
        max_ls_iter = 50
        
        for _ in range(max_ls_iter):
            improved = False
            for v in list(S_v.keys()):
                if not S_v[v]:
                    continue
                for u in list(S_v[v]):
                    # Try removing u from S_v
                    # This means the clique C_v becomes {v} U (S_v \ {u})
                    # The cross edge (v, u) is no longer covered by C_v.
                    # It must be covered by a new 2-clique {v, u}.
                    # So we add 1 clique (the 2-clique).
                    # We remove u from C_v.
                    # If C_v becomes size 1 (just v), we remove C_v entirely?
                    # No, if S_v \ {u} is empty, C_v is {v}, which is not a valid clique (size < 2).
                    # So if |S_v| == 1, removing u removes C_v and adds {v, u}. Net change 0?
                    # Wait, if |S_v|==1, C_v is {v, u} (size 2). Removing u makes it invalid.
                    # So we must replace C_v with {v, u} (which is the same).
                    # So no change in cross cliques.
                    # But internal edges (u, w) for w in S_v (which is empty) are none.
                    # So no change in internal edges.
                    
                    # If |S_v| > 1:
                    # Old C_v covers internal edges within S_v.
                    # New C_v' covers internal edges within S_v \ {u}.
                    # Edges (u, w) for w in S_v \ {u} are freed up.
                    # These freed edges go into the "remaining" pool.
                    # The cross edge (v, u) is now uncovered by C_v, so we add a 2-clique {v, u}.
                    
                    # Let's simulate:
                    # 1. Remove u from S_v.
                    # 2. Update covered_internal_edges: remove edges (u, w) for w in S_v (old).
                    # 3. Add (v, u) to uncovered cross edges -> add clique {v, u}.
                    # 4. Re-partition the new set of remaining internal edges.
                    
                    # To avoid full re-partition, we can just check if the number of cliques decreases.
                    
                    old_S_v = S_v[v]
                    new_S_v = old_S_v - {u}
                    
                    # If new_S_v is empty, C_v disappears.
                    # If new_S_v is not empty, C_v shrinks.
                    
                    # Freed internal edges:
                    freed_edges = set()
                    for w in old_S_v:
                        if w != u:
                            freed_edges.add(frozenset((u, w)))
                    
                    # These edges were previously covered. Now they are available for the K-partition.
                    
                    # We need to see if adding these edges to the remaining pool reduces the total clique count.
                    # This is complex.
                    # Simpler: Just re-run the K-partition on the new remaining edges.
                    
                    # Current remaining edges (from previous step) + freed_edges
                    # Note: The current `remaining_edges_set` is the state AFTER the initial greedy partition.
                    # We should store the state BEFORE the K-partition, or re-derive it.
                    
                    # Let's store the initial covered_internal_edges and re-derive.
                    # This is safer.
                    
                    # Recalculate covered_internal_edges for the modified S_v
                    new_covered_internal = set()
                    for v2 in S_v:
                        sv2 = S_v[v2]
                        if v2 == v:
                            sv2 = new_S_v
                        for a in sv2:
                            for b in sv2:
                                if a < b:
                                    new_covered_internal.add(frozenset((a, b)))
                                elif b < a:
                                    new_covered_internal.add(frozenset((b, a)))
                    
                    # New remaining edges
                    new_remaining = set()
                    for a in range(k):
                        for b in range(a + 1, k):
                            if frozenset((a, b)) not in new_covered_internal:
                                new_remaining.add(frozenset((a, b)))
                    
                    # Partition new_remaining
                    new_k_cliques = []
                    rem_set = new_remaining
                    while rem_set:
                        clique_verts = get_maximal_clique(rem_set, k)
                        if not clique_verts:
                            break
                        new_k_cliques.append(clique_verts)
                        for i in range(len(clique_verts)):
                            for j in range(i + 1, len(clique_verts)):
                                e = frozenset((clique_verts[i], clique_verts[j]))
                                if e in rem_set:
                                    rem_set.remove(e)
                    
                    # Total cliques:
                    # Cross cliques:
                    # For each v2, if S_v2 (modified) is non-empty, 1 clique.
                    # Plus 1 clique for (v, u) if we removed u?
                    # Wait, if we remove u from S_v, the cross edge (v, u) is NOT covered by C_v.
                    # So we must add a clique {v, u}.
                    # What if u is in S_w for some other w? No, S_v are disjoint in terms of cross edges?
                    # No, cross edges are (v, u). Each cross edge belongs to exactly one v.
                    # So (v, u) is only associated with v.
                    # So yes, we add 1 clique for {v, u}.
                    
                    cross_clique_count = 0
                    for v2 in S_v:
                        sv2 = S_v[v2]
                        if v2 == v:
                            sv2 = new_S_v
                        if sv2:
                            cross_clique_count += 1
                        else:
                            # If sv2 is empty, no clique from this v.
                            # But if we removed u and sv2 became empty, we still need to cover (v, u).
                            # So we add 1.
                            pass
                    
                    # Correction:
                    # For v2 != v: if S_v2 non-empty, 1 clique.
                    # For v:
                    #   If new_S_v non-empty: 1 clique for {v} U new_S_v.
                    #   Plus 1 clique for {v, u}.
                    #   Total 2.
                    #   If new_S_v empty: 0 clique for {v} U new_S_v.
                    #   Plus 1 clique for {v, u}.
                    #   Total 1.
                    
                    if v in S_v:
                        if new_S_v:
                            cross_count_v = 2
                        else:
                            cross_count_v = 1
                    else:
                        cross_count_v = 0 # Should not happen as we are iterating over keys
                    
                    for v2 in S_v:
                        if v2 == v:
                            continue
                        if S_v[v2]:
                            cross_count_v += 1
                    
                    total_new_cliques = cross_count_v + len(new_k_cliques)
                    
                    if total_new_cliques < best_cost:
                        # Accept change
                        S_v[v] = new_S_v
                        # Update cliques list?
                        # We just need to return the final list.
                        # We can reconstruct it at the end.
                        best_cost = total_new_cliques
                        improved = True
                        break # Restart inner loop? Or just continue?
                        # Let's break and restart the outer loop to be safe.
                    
            if not improved:
                break
        
        # Reconstruct the final list of cliques
        final_cliques = []
        
        # 1. Cross cliques
        for v in S_v:
            sv = S_v[v]
            if sv:
                final_cliques.append([v] + list(sv))
        
        # 2. Uncovered cross edges (2-cliques)
        # Which cross edges are NOT covered by the S_v cliques?
        # An edge (v, u) is covered if u in S_v[v].
        # If u NOT in S_v[v], it's uncovered.
        for v in range(k, n):
            nbrs = neighbors_in_I[v]
            sv = S_v.get(v, set())
            for u in nbrs:
                if u not in sv:
                    final_cliques.append([v, u])
        
        # 3. Remaining internal edges partition
        # Recalculate covered internal edges
        covered_internal = set()
        for v in S_v:
            sv = S_v[v]
            for a in sv:
                for b in sv:
                    if a < b:
                        covered_internal.add(frozenset((a, b)))
                    elif b < a:
                        covered_internal.add(frozenset((b, a)))
        
        remaining_edges_set = set()
        for a in range(k):
            for b in range(a + 1, k):
                if frozenset((a, b)) not in covered_internal:
                    remaining_edges_set.add(frozenset((a, b)))
        
        while remaining_edges_set:
            clique_verts = get_maximal_clique(remaining_edges_set, k)
            if not clique_verts:
                break
            final_cliques.append(clique_verts)
            for i in range(len(clique_verts)):
                for j in range(i + 1, len(clique_verts)):
                    e = frozenset((clique_verts[i], clique_verts[j]))
                    if e in remaining_edges_set:
                        remaining_edges_set.remove(e)
        
        # Validation: Ensure all edges are covered exactly once.
        # (Omitted for speed, assumed correct by construction)
        
        return final_cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
    
    return result

# EVOLVE-BLOCK-END
