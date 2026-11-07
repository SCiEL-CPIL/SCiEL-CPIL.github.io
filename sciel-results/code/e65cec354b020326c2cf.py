import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, keep 5s margin

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    results = []

    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Fallback to seed behavior if time is up
            # But seed behavior is just edges, which is valid but bad.
            # Better to return what we have? No, must return 3 lists.
            # If we timeout, we should probably just return the seed result for remaining.
            pass

        edges = instance(n, k, p, seed)
        
        # Reconstruct graph structure
        # K = 0..k-1, I = k..n-1
        # adj_K is complete.
        # adj_I is empty.
        # adj_KI: for v in I, neighbors in K.
        
        neighbors_in_K = {v: set() for v in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                neighbors_in_K[v].add(u)
            elif v < k and u >= k:
                neighbors_in_K[u].add(v)
            # Edges in K are all present, we don't need to store them explicitly for logic, 
            # but we need to know which are covered.
            
        cliques = []
        
        # 1. Create cliques for each v in I
        # C_v = N(v) U {v}
        # This covers all edges (v, u) for u in N(v)
        # And all edges (u, w) for u, w in N(v)
        
        # Build L: graph on K where (u, w) in L if exists v s.t. u, w in N(v)
        # L_adj[u][w] = True
        
        L_adj = [[False] * k for _ in range(k)]
        
        for v in range(k, n):
            Nv = neighbors_in_K[v]
            if len(Nv) >= 1:
                clique_v = list(Nv) + [v]
                cliques.append(clique_v)
                
                # Update L
                Nv_list = list(Nv)
                for i in range(len(Nv_list)):
                    for j in range(i+1, len(Nv_list)):
                        u = Nv_list[i]
                        w = Nv_list[j]
                        L_adj[u][w] = True
                        L_adj[w][u] = True

        # 2. Find clique partition of complement of L on K
        # Graph G_rem has vertices 0..k-1
        # Edge (u, w) in G_rem if not L_adj[u][w]
        
        # Greedy partition
        # Maintain set of remaining edges
        remaining_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                if not L_adj[u][w]:
                    remaining_edges.add((u, w))
                    
        while remaining_edges:
            # Find a maximal clique in G_rem
            # Simple greedy: pick a vertex, add neighbors, etc?
            # Or just find a clique that covers max edges?
            # Given small k, we can just pick a random edge, expand to maximal clique.
            
            # To be deterministic and robust:
            # Pick the edge with smallest "degree" in remaining? Or just first one?
            # Let's try to find a large clique.
            
            # Simple heuristic:
            # Candidate set = all vertices
            # Start with a vertex from an edge in remaining_edges
            u0, w0 = next(iter(remaining_edges))
            
            # Build clique starting from u0
            # Potential clique vertices must be connected to all current clique vertices in G_rem
            # And also, the clique must be a clique in G (which is K, so always true)
            # And it must not contain any edge from L.
            # So we are finding a clique in G_rem.
            
            # Greedy expansion:
            clique = [u0]
            candidates = set(range(k))
            candidates.discard(u0)
            
            # Filter candidates to those connected to u0 in G_rem
            candidates = {x for x in candidates if not L_adj[u0][x]}
            
            # Add w0 if possible? w0 is connected to u0.
            # Let's just do a simple greedy maximal clique construction.
            
            # Re-iterate:
            # We want a clique C in G_rem.
            # Start with empty set.
            # Add vertices one by one if they are connected to all existing in G_rem.
            
            # To maximize size, maybe sort candidates by degree in G_rem?
            # For now, just simple greedy.
            
            current_clique = [u0]
            # Candidates must be connected to u0
            candidates = {x for x in range(k) if x != u0 and not L_adj[u0][x]}
            
            # Sort candidates by degree in G_rem (number of neighbors in G_rem)
            # deg(x) = number of y in candidates s.t. not L_adj[x][y] ? No, in whole G_rem.
            # Let's just pick arbitrarily or by index for simplicity, k is small.
            
            # Better: Try to build a large clique.
            # This is Max Clique problem, NP-hard.
            # But k=30. We can use a simple branch and bound or just greedy.
            # Greedy:
            # 1. Start with u0.
            # 2. Candidates = neighbors of u0 in G_rem.
            # 3. While candidates not empty:
            #      Pick vertex v in candidates with max degree in G_rem (within candidates?)
            #      Add v to clique.
            #      Candidates = candidates ∩ neighbors(v) in G_rem.
            
            # Let's implement this.
            
            def get_neighbors_in_G_rem(v, node_set):
                # Returns set of nodes in node_set connected to v in G_rem
                return {x for x in node_set if not L_adj[v][x]}
            
            # Initial candidates
            candidates = get_neighbors_in_G_rem(u0, set(range(k)))
            
            # Sort candidates by degree in G_rem (global) to pick "central" ones first?
            # Or just iterate.
            # Let's sort by index for determinism.
            
            while candidates:
                # Pick vertex with highest degree in G_rem among candidates
                # deg(x) = count of y in candidates s.t. not L_adj[x][y]
                # This is O(|candidates|^2)
                best_v = None
                best_deg = -1
                for x in candidates:
                    deg = sum(1 for y in candidates if y != x and not L_adj[x][y])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = x
                # Tie-break by index
                # Actually, let's just pick the first one with max deg
                
                current_clique.append(best_v)
                # Update candidates
                candidates = get_neighbors_in_G_rem(best_v, candidates)
                
            # Now current_clique is a maximal clique in G_rem
            # Remove its edges from remaining_edges
            C = current_clique
            if len(C) >= 2:
                cliques.append(C)
                for i in range(len(C)):
                    for j in range(i+1, len(C)):
                        e = (C[i], C[j])
                        if e in remaining_edges:
                            remaining_edges.remove(e)
            # If len(C) < 2, no edges removed? But we started with an edge, so len(C)>=2.
            
        results.append(cliques)

    return results
