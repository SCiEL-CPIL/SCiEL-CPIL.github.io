import time
import sys

def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed, time_limit=110):
        start_time = time.time()
        edges = random_graph(n, p, seed)
        remaining_edges = set(edges)
        # adj_remaining[i] is a set of vertices j such that (i,j) is in remaining_edges
        adj_remaining = [set() for _ in range(n)]
        for u, v in edges:
            adj_remaining[u].add(v)
            adj_remaining[v].add(u)
        
        cliques = []
        
        while remaining_edges:
            if time.time() - start_time > time_limit:
                # Fallback: If we run out of time, we cannot return an invalid partition.
                # We must ensure we finish. The loop is O(E * N^2) roughly.
                # For N=64, E~1100. 1100 * 4096 ~ 4.5M ops. Should be fast in Python if optimized.
                # If we hit timeout, we just break, but the result will be invalid (missing edges).
                # To be safe, we should optimize the inner loop.
                break
            
            # Pick a seed edge.
            # Heuristic: Pick edge (u,v) that maximizes |N_rem(u) ∩ N_rem(v)|.
            # This suggests a large clique potential.
            best_edge = None
            max_common = -1
            # Iterate over a snapshot of keys to avoid modification issues during iteration, 
            # though we don't modify inside the selection loop.
            for u, v in remaining_edges:
                # u < v is guaranteed by construction? No, set doesn't guarantee order.
                # But adj is symmetric.
                common = len(adj_remaining[u].intersection(adj_remaining[v]))
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)
            
            if best_edge is None:
                break # Should not happen if remaining_edges is not empty

            u, v = best_edge
            
            # Greedy clique expansion starting from edge (u, v) using adj_remaining
            K = {u, v}
            # Candidates must be connected to ALL vertices in K in the remaining graph
            candidates = adj_remaining[u].intersection(adj_remaining[v])
            
            while candidates:
                # Pick vertex w in candidates that maximizes |N_rem(w) ∩ K|
                # Since we maintain candidates as intersection of neighbors, 
                # any w in candidates is connected to all in K.
                # We want to pick w that allows for further expansion?
                # Or just pick one that has many neighbors in candidates?
                # Let's pick w that maximizes |N_rem(w) ∩ candidates|
                best_w = None
                max_next = -1
                for w in candidates:
                    next_count = len(adj_remaining[w].intersection(candidates))
                    if next_count > max_next:
                        max_next = next_count
                        best_w = w
                
                if best_w is None:
                    break
                    
                K.add(best_w)
                # Update candidates: must be neighbor of best_w in remaining graph
                candidates = candidates.intersection(adj_remaining[best_w])
                
            cliques.append(list(K))
            
            # Remove all edges within K from remaining_edges and adj_remaining
            K_list = list(K)
            for i in range(len(K_list)):
                a = K_list[i]
                for j in range(i + 1, len(K_list)):
                    b = K_list[j]
                    # Ensure we remove the edge in both directions from adj
                    # And from the set
                    if (a, b) in remaining_edges:
                        remaining_edges.remove((a, b))
                    elif (b, a) in remaining_edges:
                        remaining_edges.remove((b, a))
                    
                    # Update adj_remaining
                    if b in adj_remaining[a]:
                        adj_remaining[a].remove(b)
                    if a in adj_remaining[b]:
                        adj_remaining[b].remove(a)
            
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
