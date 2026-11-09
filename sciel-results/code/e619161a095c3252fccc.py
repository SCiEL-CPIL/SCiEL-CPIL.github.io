"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(n, k, p, seed):
        if time.time() > deadline:
            # Fallback: just return per-edge cliques (valid but not optimal)
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in sorted(edges)]

        edges = instance(n, k, p, seed)
        
        # Precompute neighbors for each vertex
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Candidate cliques:
        # 1. The big clique C (vertices 0..k-1)
        # 2. For each v in I (k..n-1), the clique {v} U N(v)
        # Note: {v} U N(v) is a clique because N(v) is a subset of C, which is a clique.
        
        candidates = []
        
        # Add the big clique C
        if k >= 2:
            candidates.append(list(range(k)))
            
        # Add cliques for each vertex in I
        for v in range(k, n):
            neighbors = sorted(adj[v]) # Only neighbors in C, but sorted for consistency
            if len(neighbors) >= 1:
                clique = [v] + neighbors
                # Ensure it's a clique in the original graph (it is, by construction of split graph)
                candidates.append(clique)
        
        # Greedy Set Partition
        uncovered_edges = set(edges)
        partition = []
        
        # To speed up, we can pre-compute the set of edges for each candidate
        # But checking if a candidate is valid (no edge already covered) is O(E_c)
        # We can simply iterate and pick the best valid one.
        
        while uncovered_edges:
            if time.time() > deadline:
                # Fallback for remaining edges
                for u, v in sorted(uncovered_edges):
                    partition.append([u, v])
                break
            
            best_clique = None
            best_count = -1
            
            for clique in candidates:
                # Check if this clique can be used
                # It must not contain any edge that is already covered
                # i.e., all edges in this clique must be in uncovered_edges
                valid = True
                count = 0
                # Iterate through pairs in the clique
                # Optimize: if the clique is large, this is slow.
                # But k is small (10, 20, 30).
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        if u > v:
                            u, v = v, u
                        edge = (u, v)
                        if edge in uncovered_edges:
                            count += 1
                        else:
                            # If an edge is not in uncovered_edges, it's already covered
                            # So this clique is invalid
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and count > best_count:
                    best_count = count
                    best_clique = clique
            
            if best_clique is None:
                # Should not happen if we have per-edge cliques as fallback candidates?
                # We didn't add per-edge cliques to candidates.
                # If no candidate is valid, it means we are stuck?
                # Actually, if uncovered_edges is not empty, there is at least one edge (u,v).
                # The clique [u,v] is always a candidate?
                # We didn't explicitly add all size-2 cliques.
                # But {v} U N(v) includes size-2 cliques if N(v) has size 1.
                # And C includes all size-2 cliques in C.
                # So if an edge is uncovered, it must be in some candidate?
                # Yes, every edge is either in C or in some {v} U N(v).
                # So best_clique should not be None.
                # However, if best_count is 0, it means no edges in the clique are uncovered.
                # We require count > best_count (initialized -1).
                # If count is 0, it's valid but covers nothing. We should skip it.
                # So we should only pick if count > 0.
                if best_count == 0:
                     # Pick any edge
                     u, v = next(iter(uncovered_edges))
                     partition.append([u, v])
                     uncovered_edges.discard((u, v))
                     continue
                else:
                    break # Should not reach here

            partition.append(best_clique)
            
            # Remove edges in best_clique from uncovered_edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v:
                        u, v = v, u
                    uncovered_edges.discard((u, v))
        
        return partition

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback for remaining instances
            for s in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results):]:
                edges = instance(*s)
                results.append([[u, v] for u, v in sorted(edges)])
            break
        results.append(partition_cliques(*spec))

    # Ensure all values are Python int and lists are lists
    final = []
    for clique_list in results:
        cleaned = []
        for clique in clique_list:
            cleaned.append([int(v) for v in clique])
        final.append(cleaned)

    return final
# EVOLVE-BLOCK-END
