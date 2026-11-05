import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit (evaluator kills at 150s, keep 115s for safety)
    
    n = 20
    num_pairs = n * (n - 1) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Efficient K3,3 checker
    # A graph contains K3,3 iff there exist two disjoint sets U, V of size 3 
    # such that every u in U is connected to every v in V.
    # We can iterate over all pairs of disjoint triples? No, that's 20C6 * 10 (bipartitions) ~ 39k * 10 = 390k checks.
    # Each check is 9 edge lookups. Total ~ 3.5M ops. Fast enough for many calls.
    
    def is_k33_free(adj):
        # adj is a list of bitmasks
        # Precompute list of neighbors for each vertex for faster intersection?
        # Or just use bitmasks.
        # Iterate over all subsets of 6 vertices?
        # C(20, 6) = 38760.
        # For each subset of 6, check if it contains K3,3.
        # A 6-vertex graph contains K3,3 if and only if it IS K3,3 (since K3,3 has 9 edges, and is the unique 6-vertex 9-edge bipartite graph? No, K3,3 is the only 3-regular bipartite graph on 6 vertices? 
        # Actually, any 6-vertex graph with 9 edges that is bipartite with partition 3-3 is K3,3.
        # So we just need to check: Does the induced subgraph on these 6 vertices have 9 edges and is it bipartite with partition size 3?
        # Simpler: For each pair of disjoint triples (U, V), check if all 9 edges between U and V exist.
        
        # Optimization: 
        # Instead of all 6-subsets, iterate over all pairs of vertices (u, v).
        # Common neighbors of u, v: C_uv.
        # If |C_uv| >= 3, pick 3 common neighbors w1, w2, w3.
        # Then check if w1, w2, w3 are connected to... wait.
        # K3,3 has partitions {u1, u2, u3} and {v1, v2, v3}.
        # Every ui is connected to every vj.
        # So u1, u2, u3 have common neighbors v1, v2, v3.
        # So if we find a triple of vertices {u1, u2, u3} that have at least 3 common neighbors {v1, v2, v3}, 
        # AND v1, v2, v3 are connected to u1, u2, u3 (which is true by definition of common neighbors),
        # THEN we have a K3,3 subgraph?
        # Wait. The definition of K3,3 is a subgraph isomorphic to K3,3.
        # If {u1, u2, u3} and {v1, v2, v3} are disjoint, and every ui is connected to every vj, 
        # then the induced subgraph on these 6 vertices contains K3,3.
        # Do we need to check if u1, u2, u3 are connected to each other? No, K3,3 is a subgraph, not necessarily induced.
        # So, YES: If there exist 3 vertices U = {u1, u2, u3} such that |N(u1) \cap N(u2) \cap N(u3)| >= 3, 
        # AND the 3 common neighbors are distinct from U, then we have a K3,3.
        # Wait, the common neighbors must be disjoint from U.
        # If a common neighbor x is in U, say x=u1, then u1 is connected to u1? No, no self-loops.
        # So common neighbors are always disjoint from U.
        
        # So the check is:
        # For all triples (u, v, w):
        #   common = adj[u] & adj[v] & adj[w]
        #   if common.bit_count() >= 3:
        #       return False
        #       
        # This is exactly what the previous program tried to do, but it was slow due to Python loops.
        # Let's optimize this.
        
        # Precompute adj list as integers.
        # Iterate u < v < w.
        # C(20, 3) = 1140 triples.
        # For each, do 2 ANDs and a bit_count.
        # 1140 * (2 ANDs + 1 count) is extremely fast.
        
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                av = adj[v]
                common_uv = au & av
                # If common_uv has < 3 bits, no triple containing u,v can have >=3 common neighbors
                if common_uv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    common_uvw = common_uv & adj[w]
                    if common_uvw.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Initialization: Use a random graph or a specific seed.
    # Let's try a random graph with p=0.5, then repair.
    # Or better: Start with a known good structure?
    # The previous program used Paley. Let's try a different seed: 
    # Random regular graph?
    # Or just random dense graph and repair.
    
    def random_seed():
        adj = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.6: # Start dense
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def repair(adj):
        # Remove edges until K3,3-free
        # Greedy removal: Find a violating triple, remove a random edge incident to one of the common neighbors?
        # Actually, if (u,v,w) have >=3 common neighbors, we can remove an edge between one of {u,v,w} and one of the common neighbors.
        # Which edge to remove? 
        # Heuristic: Remove an edge that is part of many violations? 
        # For speed, just remove a random edge that fixes the current violation.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Find first violation
            for u in range(n):
                au = adj[u]
                for v in range(u + 1, n):
                    av = adj[v]
                    common_uv = au & av
                    if common_uv.bit_count() < 3:
                        continue
                    for w in range(v + 1, n):
                        common_uvw = common_uv & adj[w]
                        if common_uvw.bit_count() >= 3:
                            # Found violation: u,v,w share >=3 neighbors.
                            # Pick a random common neighbor x
                            common_list = []
                            temp = common_uvw
                            while temp:
                                low = temp & (-temp)
                                x = low.bit_length() - 1
                                common_list.append(x)
                                temp ^= low
                            
                            x = random.choice(common_list)
                            # Remove edge (u, x) or (v, x) or (w, x)?
                            # Removing (u, x) reduces common neighbors of (u,v,w) by 1 (if x was only connected to u among u,v,w? No, x is common to all).
                            # So removing (u, x) makes x not a neighbor of u, so x is no longer common to u,v,w.
                            # So the violation for this triple is fixed.
                            
                            adj[u] &= ~(1 << x)
                            adj[x] &= ~(1 << u)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj

    def local_search(adj):
        # 2-opt swap: remove (a,b), add (c,d).
        # Net change: 0 edges. But we want to MAXIMIZE edges.
        # So we should look for: remove 1 edge, add 2 edges? 
        # Or: remove 1 edge, add 1 edge if it allows adding MORE edges later?
        
        # Better Local Search: 
        # 1. Try to add any missing edge (Greedy Add).
        # 2. If stuck, try a "swap": Remove an edge e1, then greedily add edges. 
        #    If we can add > 1 edge, net gain.
        #    If we add 1 edge, net 0, but maybe it unlocks more?
        
        # Let's implement a "Remove and Re-add" strategy.
        # Pick a random edge to remove.
        # Greedily add edges until no more can be added.
        # If final edge count > initial, keep. Else revert.
        
        current_edges = count_edges(adj)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each edge? Too slow (190 edges).
            # Try removing random edges.
            for _ in range(20): # 20 attempts per pass
                if time.time() > deadline - 5:
                    break
                # Pick random edge to remove
                u, v = random.choice(all_pairs)
                if not (adj[u] & (1 << v)):
                    continue
                
                # Save state
                old_adj = adj[:]
                
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Greedy Add
                added = 0
                # Iterate over missing edges in random order? Or fixed order?
                # Fixed order is deterministic and faster to check.
                for x, y in all_pairs:
                    if not (adj[x] & (1 << y)):
                        # Try to add
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        if is_k33_free(adj):
                            added += 1
                        else:
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                            
                new_edges = count_edges(adj)
                
                if new_edges > current_edges:
                    current_edges = new_edges
                    improved = True
                else:
                    # Revert
                    adj = old_adj
                    # Note: If new_edges == current_edges, we might still want to keep the new state 
                    # if it's "better" in terms of potential? 
                    # But for now, only keep if strictly better.
                    # To avoid local optima, we might accept equal if random? 
                    # Let's stick to strict improvement for now to ensure progress.
                    
        return adj, current_edges

    best_adj = None
    best_edges = 0
    
    # Multi-start
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        adj = random_seed()
        adj = repair(adj)
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # If we reached a very high density, we might stop early?
        # Max possible is 190. 
        # Known ex(20, K3,3) is likely around 100-120?
        # If best_edges > 110, we can reduce restarts.
        
    # Final polish on best_adj
    if best_adj:
        best_adj, best_edges = local_search(best_adj)
        
    return adj_to_edges(best_adj) if best_adj else []
