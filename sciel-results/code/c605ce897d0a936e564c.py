import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_bottleneck_partition(n, edges, seed_heur):
        # Build adjacency list for quick neighbor access
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                # Fallback: return current cliques plus edges as pairs if we run out of time
                # But we must ensure validity. If we break early, we have uncovered edges.
                # The contract says "every edge must lie in exactly one listed clique".
                # So we cannot return an invalid partition.
                # We must finish or have a valid partition.
                # If time is tight, we should have pre-computed or use a faster fallback.
                # Let's just let it run, the inner loop is fast enough for these sizes.
                pass

            # Find the edge with the minimum common neighborhood size
            min_common = float('inf')
            target_edge = None
            
            # Optimization: If the set is large, sampling might be faster, 
            # but for n=64, E~1100, full scan is okay.
            for e in remaining_edges:
                u, v = e
                # Common neighbors in the remaining graph
                # Note: adj[u] contains all neighbors, not just those in remaining_edges.
                # We need common neighbors that are connected to BOTH u and v in the REMAINING graph.
                # Actually, the clique must be formed from vertices connected in the remaining graph.
                # So common neighbors must be in adj[u] AND adj[v].
                # But we also need to ensure that the edges between these common neighbors 
                # are present if we want to extend further.
                # For the "bottleneck" metric, we just count potential candidates.
                
                # A vertex w is a candidate if (u,w) in remaining and (v,w) in remaining.
                # This is equivalent to w in adj[u] and w in adj[v] and (u,w) in rem and (v,w) in rem.
                # Since remaining_edges only contains edges present in the original graph,
                # and we remove edges, adj should ideally be updated or we check membership.
                # Checking membership in a set is O(1).
                
                # Let's count candidates w such that (u,w) in remaining and (v,w) in remaining.
                # This is | {w | (u,w) in remaining} \cap {w | (v,w) in remaining} |
                # Which is | N_rem(u) \cap N_rem(v) |
                
                # We can compute N_rem(u) on the fly? No, that's O(E).
                # Better to maintain N_rem(u) as a set.
                
                # Let's refactor to maintain remaining neighbors.
                pass
            
            # Refactored approach: Maintain remaining neighbors
            # This was not done in the first pass. Let's do it properly.
            
            break # Placeholder to restart logic below

    def greedy_bottleneck_partition_v2(n, edges, seed_heur):
        adj_rem = [set() for _ in range(n)]
        for u, v in edges:
            adj_rem[u].add(v)
            adj_rem[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            # Find edge with min common neighbors in remaining graph
            min_common = float('inf')
            target_edge = None
            
            for e in remaining_edges:
                u, v = e
                # Common neighbors are those w in adj_rem[u] that are also in adj_rem[v]
                # Note: w must be connected to both u and v in the remaining graph.
                common = len(adj_rem[u] & adj_rem[v])
                if common < min_common:
                    min_common = common
                    target_edge = e
                    # Early exit if we find a very constrained edge? 
                    # Min possible is 0.
                    if common == 0:
                        break
            
            if target_edge is None:
                # Should not happen if remaining_edges is not empty
                target_edge = next(iter(remaining_edges))
            
            u, v = target_edge
            # Find maximal clique containing u, v
            C = [u, v]
            P = adj_rem[u] & adj_rem[v]
            
            while P:
                # Pick vertex in P with max degree in P
                max_deg = -1
                best_candidate = None
                # Use a deterministic tie-breaker or random?
                # Random helps in restarts.
                candidates = list(P)
                rng_heur.shuffle(candidates) # Optional, or just pick random from max degree group
                
                # Find max degree in P
                degs = [(len(adj_rem[x] & P), x) for x in candidates]
                max_d = max(d for d, _ in degs)
                # Pick one of the max degree vertices
                # To allow variation, pick randomly among max degree
                max_cands = [x for d, x in degs if d == max_d]
                w = rng_heur.choice(max_cands)
                
                C.append(w)
                P = P & adj_rem[w]
            
            # Remove edges of C from remaining_edges and adj_rem
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        adj_rem[a].discard(b)
                        adj_rem[b].discard(a)
            
            cliques.append(C)
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        while time.time() < time_limit:
            heur_seed = seed + attempt * 1000
            cliques = greedy_bottleneck_partition_v2(n, edges, heur_seed)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: single edge cliques
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time
    t1_limit = start_time + 30.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
