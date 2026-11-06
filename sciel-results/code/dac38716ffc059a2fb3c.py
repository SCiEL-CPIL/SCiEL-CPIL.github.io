import time
import heapq
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def get_strong_adjacency(edges, adj_sets):
        """
        edges: list of tuples (u, v)
        adj_sets: dict vertex -> set of neighbors
        Returns: list of sets, where strong_adj[i] is the set of indices of edges strongly adjacent to edges[i]
        """
        n_edges = len(edges)
        strong_adj = [set() for _ in range(n_edges)]
        
        # Map edge tuple to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    strong_adj[i].add(j)
                    strong_adj[j].add(i)
                else:
                    # Check if there is an edge connecting any endpoint of e_i to any endpoint of e_j
                    # i.e., is u1 adjacent to u2? u1 to v2? v1 to u2? v1 to v2?
                    if (u2 in adj_sets[u1]) or (v2 in adj_sets[u1]) or (u2 in adj_sets[v1]) or (v2 in adj_sets[v1]):
                        strong_adj[i].add(j)
                        strong_adj[j].add(i)
                        
        return strong_adj

    def dsatur_coloring(strong_adj, n_edges):
        """
        strong_adj: list of sets of neighbor indices
        n_edges: number of edges
        Returns: list of colors, one per edge
        """
        colors = [-1] * n_edges
        # neighbor_colors[i] is a set of colors used by neighbors of edge i
        neighbor_colors = [set() for _ in range(n_edges)]
        
        # To speed up, we can keep a list of uncolored edges.
        # Since N is small, scanning all uncolored edges each step is O(E^2) which is fine.
        
        for _ in range(n_edges):
            # Find the uncolored edge with highest saturation, then highest degree
            best_idx = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(n_edges):
                if colors[i] == -1:
                    sat = len(neighbor_colors[i])
                    deg = len(strong_adj[i])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_idx = i
            
            if best_idx == -1:
                break # Should not happen if graph is connected enough, but safe
                
            # Assign smallest available color
            c = 0
            while c in neighbor_colors[best_idx]:
                c += 1
            
            colors[best_idx] = c
            
            # Update neighbors' neighbor_colors
            for j in strong_adj[best_idx]:
                if colors[j] == -1:
                    neighbor_colors[j].add(c)
                    
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # If we run out of time, return empty lists to avoid exception, 
            # though this will score 0. Better to just return what we have if possible.
            # But since we process one by one, if we haven't finished, we can't return partial.
            # However, DSATUR is very fast. This guard is just in case.
            pass
            
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Build adjacency sets for G
        adj_sets = {i: set() for i in range(n)}
        for u, v in edges:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
        # Compute strong adjacency
        strong_adj = get_strong_adjacency(edges, adj_sets)
        
        # Color
        colors = dsatur_coloring(strong_adj, len(edges))
        
        out.append(colors)
        
        if time.time() > deadline:
             # If we are about to run out of time, we should probably stop and return whatever we have?
             # The interface expects 3 lists. If we don't finish, we might return a shorter list.
             # The verifier might crash or score 0. 
             # Given the small size, this is unlikely.
             pass

    return out
