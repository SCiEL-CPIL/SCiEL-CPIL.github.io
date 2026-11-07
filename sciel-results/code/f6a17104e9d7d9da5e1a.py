"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency list and edge set
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        cliques = []
        
        while edge_set:
            if time.time() > deadline:
                # Fallback: return remaining edges as individual cliques
                # This ensures validity even if we run out of time
                remaining = [list(e) for e in edge_set]
                cliques.extend(remaining)
                break
            
            # Heuristic: Pick edge with max sum of degrees
            # To speed up, we can just pick the first one or use a simple score
            # Let's pick the edge with the highest combined degree in the current residual graph
            best_edge = None
            best_score = -1
            
            # Optimization: Instead of iterating all edges every time, 
            # we can just pick one. For small graphs, iterating is fine.
            # But let's be smart: The clique size is bounded by k (the clique part).
            # Actually, we can just try to find a large clique from scratch.
            
            # Let's try a different strategy:
            # 1. Identify the "clique part" vertices (0 to k-1).
            # 2. Any clique in a split graph is either:
            #    a) A subset of the clique part (0..k-1)
            #    b) A subset of the clique part plus one vertex from the independent set (k..n-1)
            
            # Since vertices k..n-1 form an independent set, any clique can contain at most one such vertex.
            # So, we can iterate through vertices in the independent set, and for each,
            # find the largest clique in the clique part that is adjacent to it.
            # Also, we need to cover edges within the clique part (0..k-1) that are not covered by cliques containing independent set vertices.
            
            # Let's refine this structural approach.
            # Vertices 0..k-1 form a complete graph $K_k$.
            # Vertices k..n-1 are independent.
            # Edges are:
            # 1. All pairs in 0..k-1.
            # 2. Some pairs between 0..k-1 and k..n-1.
            
            # A clique can be:
            # - {u, v} where u, v in 0..k-1. (Size 2)
            # - {u, v, w} where u,v in 0..k-1, w in k..n-1 and w connected to u,v. (Size 3)
            # - Larger cliques involving multiple 0..k-1 vertices and one k..n-1 vertex.
            
            # Strategy:
            # 1. For each vertex w in k..n-1, let N(w) be its neighbors in 0..k-1.
            #    The edges incident to w are (u, w) for u in N(w).
            #    We can cover these edges by finding cliques in the subgraph induced by N(w) plus w.
            #    Since N(w) is a subset of a clique (0..k-1), N(w) itself is a clique.
            #    So {N(w) union {w}} is a clique? No, N(w) is a clique, and w is connected to all in N(w).
            #    So {N(w) union {w}} IS a clique.
            #    However, we don't necessarily want to use ALL of N(w) in one clique if it prevents better packing of edges within 0..k-1.
            #    But edges within 0..k-1 are plentiful.
            
            # Let's try a greedy approach specifically exploiting the structure:
            # 1. Sort independent set vertices by degree (descending).
            # 2. For each vertex w in k..n-1:
            #    Let C_w be the set of neighbors in 0..k-1 that are still connected to w (edge not covered).
            #    If C_w is not empty, form a clique {C_w union {w}}? 
            #    Wait, if we use all of C_w, we cover all edges incident to w.
            #    This removes those edges from the "internal" edge set of 0..k-1? No, the edges are (u,w).
            #    The edges (u,v) within 0..k-1 are separate.
            #    
            #    If we form clique {C_w union {w}}, we cover edges (u,w) for u in C_w.
            #    This does NOT cover edges (u,v) within 0..k-1.
            #    
            #    So, we can cover ALL edges incident to the independent set vertices very efficiently!
            #    For each w, the set of neighbors in the clique part is a clique. Adding w makes it a larger clique.
            #    So we can cover all "cross" edges with at most (n-k) cliques, one for each independent vertex (if it has edges).
            #    
            #    After covering all cross edges, we are left with the edges within the clique part (0..k-1).
            #    These edges form a complete graph $K_k$ minus any edges that were "used" in previous cliques?
            #    No, the cliques {C_w union {w}} contain edges (u,v) for u,v in C_w as well!
            #    Because C_w is a clique, any pair in C_w is an edge.
            #    So forming {C_w union {w}} covers:
            #    - All (u,w) for u in C_w.
            #    - All (u,v) for u,v in C_w.
            
            # This is a powerful observation.
            # Let's simulate:
            # 1. Start with all edges uncovered.
            # 2. For each w in k..n-1:
            #    Identify neighbors N_w in 0..k-1 such that edge (u,w) is still uncovered.
            #    If N_w is not empty:
            #       Form clique Q = N_w union {w}.
            #       Remove all edges in Q from the uncovered set.
            #       Add Q to cliques list.
            # 3. Now, handle remaining edges within 0..k-1.
            #    These form a subgraph of K_k.
            #    We need to cover this subgraph with minimum cliques.
            #    Since it's a subgraph of a clique, we can use a greedy algorithm for edge clique cover on this small graph.
            
            # Let's implement this.
            
            remaining_edges = set(edge_set)
            
            # Step 1: Cover cross edges
            # Sort independent vertices by number of neighbors in clique part (descending) to maximize coverage of internal edges?
            # Actually, the internal edges covered depend on the size of N_w.
            # Larger N_w covers more internal edges (pairs within N_w).
            # So we should process vertices with larger N_w first?
            # Or maybe it doesn't matter much, but let's try sorting by degree descending.
            
            indep_vertices = list(range(k, n))
            # Calculate initial degrees in clique part
            degs = {w: len([u for u in range(k) if (min(u,w), max(u,w)) in remaining_edges]) for w in indep_vertices}
            indep_vertices.sort(key=lambda w: degs[w], reverse=True)
            
            for w in indep_vertices:
                if time.time() > deadline:
                    break
                # Find neighbors in 0..k-1 with uncovered edge to w
                neighbors = []
                for u in range(k):
                    e = (u, w)
                    if e in remaining_edges:
                        neighbors.append(u)
                
                if not neighbors:
                    continue
                
                # The clique is neighbors + [w]
                clique = neighbors + [w]
                # Remove all edges in this clique from remaining_edges
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        e = (u, v) if u < v else (v, u)
                        remaining_edges.discard(e)
                
                cliques.append(clique)
            
            # Step 2: Cover remaining edges within 0..k-1
            # The remaining edges form a graph G' on vertices 0..k-1.
            # We need to cover G' with cliques.
            # G' is a subgraph of a complete graph.
            
            # Greedy for G':
            # While remaining_edges not empty:
            #   Pick an edge (u,v) from remaining_edges.
            #   Find max clique containing (u,v) in G'.
            #   Remove its edges.
            
            # Since k is small (10, 20, 30), we can afford a decent search.
            # Let's use a simple heuristic: pick edge with max degree sum, expand to max clique.
            
            while remaining_edges:
                if time.time() > deadline:
                    # Fallback
                    for e in remaining_edges:
                        cliques.append(list(e))
                    break
                
                # Pick an edge
                # Heuristic: max sum of degrees in G'
                # Degrees in G'
                deg = {u: 0 for u in range(k)}
                for u, v in remaining_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                best_edge = None
                best_score = -1
                for u, v in remaining_edges:
                    score = deg[u] + deg[v]
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
                
                u, v = best_edge
                
                # Find max clique containing u, v
                # Common neighbors of u and v in G'
                common = adj[u].intersection(adj[v])
                # Filter to those in remaining graph? 
                # Actually, adj is the original graph. We need to ensure edges exist in remaining_edges.
                # Let's build a local adjacency for remaining_edges restricted to 0..k-1.
                # For efficiency, let's just try to expand greedily.
                
                # Start with {u, v}
                current_clique = {u, v}
                # Candidates: common neighbors
                candidates = list(common)
                
                # Greedy expansion
                # Sort candidates by degree in current clique? Or just pick one that maximizes clique size?
                # Simple: add vertex with highest degree in the subgraph induced by current_clique + candidates?
                # Let's try: repeatedly add the vertex from candidates that has the most neighbors in current_clique.
                
                while candidates:
                    if time.time() > deadline:
                        break
                    # Find candidate with max connections to current_clique
                    best_cand = None
                    max_conn = 0
                    for c in candidates:
                        conn = len([x for x in current_clique if (min(c,x), max(c,x)) in remaining_edges])
                        if conn > max_conn:
                            max_conn = conn
                            best_cand = c
                    if best_cand is None or max_conn == 0:
                        break
                    current_clique.add(best_cand)
                    candidates.remove(best_cand)
                    # Update candidates: remove those not connected to best_cand in remaining_edges
                    # Actually, we just need to ensure the new clique is valid.
                    # If we add best_cand, it must be connected to all in current_clique.
                    # The check `conn == len(current_clique)` would be strict, but we used `conn > max_conn`.
                    # We should ensure best_cand is connected to ALL in current_clique.
                    # Let's re-verify.
                    # If we just pick the one with max connections, it might not be connected to ALL.
                    # We need a valid clique.
                    # So, we should only consider candidates that are connected to ALL vertices in current_clique.
                    
                    # Let's restart the expansion logic more carefully.
                    # This inner loop is getting complex. Let's simplify.
                    
                    # For small k, we can just check all subsets? No.
                    # Let's use a simpler heuristic for the final part:
                    # Just take the edge (u,v) and try to add vertices one by one if they are connected to all existing.
                    
                    # Re-do expansion for (u,v)
                    current_clique = {u, v}
                    candidates = [c for c in common if (min(u,c), max(u,c)) in remaining_edges and (min(v,c), max(v,c)) in remaining_edges]
                    # Actually, common already implies connected to u and v in ORIGINAL graph.
                    # We need connected in REMAINING graph.
                    candidates = [c for c in common if (min(u,c), max(u,c)) in remaining_edges and (min(v,c), max(v,c)) in remaining_edges]
                    
                    while candidates:
                        # Find candidate connected to all in current_clique
                        valid_cands = []
                        for c in candidates:
                            is_connected = all((min(c,x), max(c,x)) in remaining_edges for x in current_clique)
                            if is_connected:
                                valid_cands.append(c)
                        
                        if not valid_cands:
                            break
                        
                        # Pick one (e.g., first one, or one with highest degree)
                        # Let's pick the one with highest degree in remaining_edges
                        best_c = max(valid_cands, key=lambda x: deg[x])
                        current_clique.add(best_c)
                        candidates.remove(best_c)
                        # Update candidates: remove those not connected to best_c
                        # We'll just re-filter in next iteration.
                    
                    # Add clique
                    clique_list = list(current_clique)
                    cliques.append(clique_list)
                    
                    # Remove edges
                    for i in range(len(clique_list)):
                        for j in range(i+1, len(clique_list)):
                            a, b = clique_list[i], clique_list[j]
                            e = (a, b) if a < b else (b, a)
                            remaining_edges.discard(e)
            
            return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, we have to return something valid.
            # The safest is to return individual edges for the remaining instances.
            # But we need to return 3 lists.
            # If we haven't solved an instance, solve it with the trivial method.
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            # Ensure types are correct
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END
