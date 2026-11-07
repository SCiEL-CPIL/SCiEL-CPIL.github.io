import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # Vertices 0..k-1 are the clique K
        # Vertices k..n-1 are the independent set I
        # Edges in K are all pairs
        # Edges in K x I are random with prob p
        # Edges in I are none
        
        # We need to return a list of cliques (lists of vertices)
        # such that every edge is in exactly one clique.
        
        # 1. Handle edges incident to I
        # For each v in I, let N_K(v) be its neighbors in K.
        # The edges incident to v are (v, u) for u in N_K(v).
        # We cover these with the clique C_v = N_K(v) U {v}.
        # This also covers edges (u, w) in K where u, w in N_K(v).
        
        cliques = []
        
        # Build neighbor lists for v in I
        # neighbors_in_K[v] is a set of u in K
        neighbors_in_K = {}
        for v in range(k, n):
            neigh = []
            for u in range(k):
                if rng.random() < p:
                    neigh.append(u)
            if neigh:
                neighbors_in_K[v] = neigh
        
        # Add cliques for I vertices
        for v, neigh in neighbors_in_K.items():
            clique = [v] + neigh
            cliques.append(clique)
            
        # 2. Handle residual edges in K
        # An edge (u, w) in K is covered if there exists v in I such that u, w in neighbors_in_K[v].
        # We need to cover edges (u, w) where no such v exists.
        
        # Precompute which pairs in K are covered
        # covered[u][w] = True if covered
        # Since k is small, we can use a set of tuples or a 2D array.
        covered_pairs = set()
        for v, neigh in neighbors_in_K.items():
            # All pairs in neigh are covered
            for i in range(len(neigh)):
                for j in range(i+1, len(neigh)):
                    u, w = neigh[i], neigh[j]
                    covered_pairs.add((min(u,w), max(u,w)))
                    
        # Find residual edges
        residual_edges = []
        for u in range(k):
            for w in range(u+1, k):
                if (u, w) not in covered_pairs:
                    residual_edges.append((u, w))
                    
        # 3. Cover residual edges with minimum cliques
        # The residual graph H is on vertices K.
        # We want to cover E(H) with cliques.
        # Greedy approach:
        # - Find triangles in H, cover them, remove edges.
        # - Cover remaining edges as single edges.
        
        # Build adjacency for H
        adj_H = {u: set() for u in range(k)}
        for u, w in residual_edges:
            adj_H[u].add(w)
            adj_H[w].add(u)
            
        # List of edges to cover
        edges_to_cover = set(residual_edges)
        
        # Greedy triangle finding
        # Since H is sparse, we can iterate through edges
        # To find triangles efficiently, we can check for each edge (u,w) if they have a common neighbor
        # But we must be careful not to double count or use removed edges.
        
        # Simple heuristic:
        # 1. Find all triangles in the current graph.
        # 2. Pick one, add as clique, remove its 3 edges.
        # 3. Repeat until no triangles.
        # 4. Add remaining edges as 2-cliques.
        
        # Optimization: Since k <= 30, we can just check all triples?
        # O(k^3) is 27000. Very fast.
        # But we need to ensure edges exist.
        
        while True:
            found_triangle = None
            # Check for triangles
            # Iterate u < w < x
            for u in range(k):
                if not adj_H[u]: continue
                for w in adj_H[u]:
                    if w <= u: continue
                    # Check common neighbors of u and w
                    common = adj_H[u].intersection(adj_H[w])
                    for x in common:
                        if x > w: # Ensure u < w < x to avoid duplicates
                            # Check if edges (u,w), (w,x), (u,x) are in edges_to_cover
                            if (u,w) in edges_to_cover and (w,x) in edges_to_cover and (u,x) in edges_to_cover:
                                found_triangle = (u, w, x)
                                break
                    if found_triangle: break
                if found_triangle: break
            if not found_triangle:
                break
                
            # Add triangle clique
            u, w, x = found_triangle
            cliques.append([u, w, x])
            # Remove edges
            edges_to_cover.discard((u, w))
            edges_to_cover.discard((w, x))
            edges_to_cover.discard((u, x))
            adj_H[u].discard(w)
            adj_H[w].discard(u)
            adj_H[w].discard(x)
            adj_H[x].discard(w)
            adj_H[u].discard(x)
            adj_H[x].discard(u)
            
            if time.time() > DEADLINE:
                break

        # Add remaining edges as 2-cliques
        for u, w in edges_to_cover:
            cliques.append([u, w])
            
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            # If we run out of time, we need to return something valid.
            # But we are likely done.
            break
        res = instance(*spec)
        # Convert to ensure types are correct
        # Vertices are already ints from range()
        # Cliques are lists of ints
        results.append(res)
        
    # If we broke early, we might have fewer than 3 results.
    # The task requires 3 lists.
    # However, the computation is so fast it won't happen.
    # To be safe, if we didn't finish, we can't just return partial.
    # But given the complexity, it's safe.
    
    return results
