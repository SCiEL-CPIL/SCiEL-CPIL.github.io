import time
import random
import numpy as np
from scipy.sparse.linalg import eigsh
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Adjacency matrix for spectral analysis
        # We will maintain a set of remaining edges and an adjacency structure
        # For spectral, we need the original graph structure or dynamic?
        # Dynamic spectral is expensive. Let's use static spectral for ordering,
        # but dynamic degrees for selection.
        
        # Initial adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
            
        # Compute eigenvector centrality (principal eigenvector)
        try:
            # k=1 for largest eigenvalue
            # sigma=0 is default for largest magnitude
            vals, vecs = eigsh(A, k=1, which='LA')
            centrality = np.abs(vecs[:, 0])
        except Exception:
            # Fallback to degree centrality if eigsh fails
            centrality = np.sum(A, axis=1)
        
        # Normalize centrality
        if np.max(centrality) > 0:
            centrality = centrality / np.max(centrality)

        cliques = []
        remaining_edges = set(edges)
        
        # Helper to get neighbors of u in remaining graph
        def get_neighbors(u, rem_edges):
            return {v for (a, b) in rem_edges if a == u for v in [b] if v != u} | \
                   {v for (a, b) in rem_edges if b == u for v in [a] if v != u}

        # Bron-Kerbosch with pivot to find maximum clique in a subgraph defined by a candidate set
        # Returns the largest clique found
        def find_max_clique(candidates, rem_edges):
            # candidates is a set of vertices
            # We want to find the max clique within candidates
            # Use Bron-Kerbosch
            best_clique = []
            
            def bronk(R, P, X):
                nonlocal best_clique
                if time.time() > deadline:
                    return
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = list(R)
                    return
                
                # Pivot selection: choose u in P union X with max neighbors in P
                # For dense graphs, P is small
                U = P.union(X)
                if not U:
                    return
                pivot = max(U, key=lambda u: len(P.intersection(get_neighbors(u, rem_edges))))
                
                for v in list(P - get_neighbors(pivot, rem_edges)):
                    bronk(R.union({v}), P.intersection(get_neighbors(v, rem_edges)), X.intersection(get_neighbors(v, rem_edges)))
                    P.remove(v)
                    X.add(v)
            
            # We are looking for clique in the subgraph induced by candidates
            # But we must ensure edges exist in rem_edges
            # The get_neighbors function uses rem_edges, so it's correct.
            # However, Bron-Kerbosch assumes a static graph. 
            # Since rem_edges doesn't change during this search, it's fine.
            
            # Optimization: Only consider vertices in candidates
            # We need to filter P to be candidates
            P = set(candidates)
            X = set()
            R = set()
            
            # Pre-filter: Remove vertices not connected to all others? No, BK handles that.
            # But we can prune P to only those with degree > 0 in the subgraph
            # Actually, just run BK.
            
            bronk(R, P, X)
            return best_clique

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges with pairs
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find vertex with max degree in remaining graph
            degs = {u: 0 for u in range(n)}
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
            
            active_vertices = [u for u in range(n) if degs[u] > 0]
            if not active_vertices:
                break
            
            # Select seed: Highest centrality among active, tie-break by degree
            # Or simply highest degree?
            # Let's use a score: centrality * 0.5 + degree * 0.5
            def score(u):
                return centrality[u] * 0.5 + degs[u] * 0.5
            
            seed = max(active_vertices, key=score)
            
            # Get candidates for clique: neighbors of seed
            candidates = get_neighbors(seed, remaining_edges)
            
            # Try to find a large clique containing seed
            # We can run find_max_clique on candidates U {seed}
            # But find_max_clique as defined finds max clique in the induced subgraph.
            # We specifically want one containing seed.
            # We can force seed to be in R.
            
            # Let's modify find_max_clique to force a vertex
            def find_max_clique_with_seed(seed, candidates, rem_edges):
                best_clique = [seed]
                
                def bronk(R, P, X):
                    nonlocal best_clique
                    if time.time() > deadline:
                        return
                    if not P and not X:
                        if len(R) > len(best_clique):
                            best_clique = list(R)
                        return
                    
                    U = P.union(X)
                    if not U:
                        return
                    pivot = max(U, key=lambda u: len(P.intersection(get_neighbors(u, rem_edges))))
                    
                    for v in list(P - get_neighbors(pivot, rem_edges)):
                        bronk(R.union({v}), P.intersection(get_neighbors(v, rem_edges)), X.intersection(get_neighbors(v, rem_edges)))
                        P.remove(v)
                        X.add(v)
                
                P = set(candidates)
                X = set()
                R = {seed}
                
                bronk(R, P, X)
                return best_clique

            clique = find_max_clique_with_seed(seed, candidates, remaining_edges)
            
            # If the found clique is just {seed} (no neighbors added), 
            # it means seed is isolated in the remaining graph? 
            # But degs[seed] > 0, so it has neighbors.
            # If BK failed to find any extension, it might be due to timeout or logic.
            # If clique size is 1, we must at least form an edge.
            if len(clique) < 2:
                # Pick any neighbor
                if candidates:
                    v = next(iter(candidates))
                    clique = [seed, v]
                else:
                    # Should not happen if deg > 0
                    break

            # Remove edges of the clique
            clique_set = set(clique)
            edges_to_remove = set()
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u_c, v_c = clique[i], clique[j]
                    if u_c > v_c:
                        u_c, v_c = v_c, u_c
                    edges_to_remove.add((u_c, v_c))
            
            # Only remove edges that are actually present
            for e in edges_to_remove:
                if e in remaining_edges:
                    remaining_edges.remove(e)
            
            cliques.append(clique)
            
            # Update centrality? 
            # Recomputing eigenvectors is O(n^3) or similar.
            # n=64, so it's fast. But we do it in a loop.
            # Number of cliques might be ~50-100.
            # 100 * O(64^3) is fine.
            # Let's update A and recompute centrality every 10 cliques or if significant change.
            # For simplicity and structural adherence, let's just update A.
            # We won't recompute eigenvectors every step to save time, 
            # but the initial centrality guides the first few large cliques.
            # Actually, let's just recompute every step for n=64.
            A_sub = np.zeros((n, n), dtype=np.float64)
            for u, v in remaining_edges:
                A_sub[u, v] = 1.0
                A_sub[v, u] = 1.0
            
            try:
                if np.any(A_sub):
                    vals, vecs = eigsh(A_sub, k=1, which='LA')
                    centrality = np.abs(vecs[:, 0])
                    if np.max(centrality) > 0:
                        centrality = centrality / np.max(centrality)
                    else:
                        centrality = np.sum(A_sub, axis=1)
            except Exception:
                centrality = np.sum(A_sub, axis=1)

        return cliques

    # Instance 1
    cliques1 = solve_instance(32, 0.6, 601)
    
    # Instance 2
    cliques2 = solve_instance(64, 0.55, 602)
    
    return [cliques1, cliques2]
