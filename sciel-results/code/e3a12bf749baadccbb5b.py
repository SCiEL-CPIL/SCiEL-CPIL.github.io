"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
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

    def strong_coloring(edges, n, max_iter=100):
        if not edges:
            return []
        
        # Build adjacency list for vertices
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        
        # Build conflict graph adjacency list
        # Two edges e1=(u,v) and e2=(x,y) conflict if:
        # 1. They share a vertex
        # 2. There is an edge connecting a vertex of e1 to a vertex of e2
        # i.e., u in adj[x] or u in adj[y] or v in adj[x] or v in adj[y]
        # Note: If they share a vertex, it's already a conflict.
        
        conflict_adj = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check conflict
                if u == x or u == y or v == x or v == y:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)
                else:
                    if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                    
        # DSATUR Heuristic for Coloring Conflict Graph
        # State: color[i] = -1 if uncolored
        best_colors = None
        best_count = float('inf')
        
        for attempt in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            color = [-1] * m
            # Track available colors for each uncolored vertex? 
            # Simpler: For each uncolored vertex, calculate set of used colors in its neighborhood.
            
            # Priority queue: (-available_colors_count, -degree, index)
            # We want to pick the vertex with fewest available colors (most constrained).
            # If tie, highest degree in uncolored subgraph.
            
            uncolored = set(range(m))
            
            # Initial priority
            pq = []
            for i in range(m):
                # Degree in full conflict graph is a static tie-breaker
                heapq.heappush(pq, (-len(conflict_adj[i]), -i, i)) 
                # Note: We will update this dynamically or re-evaluate.
                # For simplicity and speed, we might just re-scan or use a simpler metric.
                # Let's use a simpler greedy: pick uncolored vertex with most colored neighbors.
                
            # Actually, a standard efficient DSATUR implementation:
            # 1. Pick uncolored vertex with highest saturation (distinct colors in neighbors).
            # 2. Tie-break by degree in uncolored subgraph.
            # 3. Assign smallest available color.
            
            color = [-1] * m
            uncolored = set(range(m))
            
            while uncolored:
                if time.time() > deadline - 2.0:
                    # Return partial or fail? Better to return best so far or valid partial?
                    # The task requires valid coloring. If we run out of time, we must return a valid one.
                    # We will ensure we finish the current coloring if possible, or fallback to simple greedy.
                    pass
                    
                best_v = -1
                best_sat = -1
                best_deg = -1
                
                # Find vertex with max saturation
                # Optimization: We can maintain a list, but m is small (< 300 edges approx)
                # Max edges for n=80, d=6 is approx 80*6/2 = 240.
                # So scanning all uncolored is O(M^2) which is 240^2 = 57600. Very fast.
                
                for v in uncolored:
                    # Calculate saturation: distinct colors in neighbors
                    neigh_colors = set()
                    for u in conflict_adj[v]:
                        if color[u] != -1:
                            neigh_colors.add(color[u])
                    sat = len(neigh_colors)
                    
                    # Degree in uncolored subgraph
                    deg = 0
                    for u in conflict_adj[v]:
                        if u in uncolored:
                            deg += 1
                            
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                        
                if best_v == -1:
                    break # Should not happen
            
                # Assign smallest available color
                used_colors = set()
                for u in conflict_adj[best_v]:
                    if color[u] != -1:
                        used_colors.add(color[u])
                        
                c = 0
                while c in used_colors:
                    c += 1
                color[best_v] = c
                uncolored.remove(best_v)
                
            # Count colors used
            if -1 not in color:
                max_c = max(color) + 1
                if max_c < best_count:
                    best_count = max_c
                    best_colors = color[:]
                    
            # Simple randomization: shuffle tie-breaks?
            # The current implementation is deterministic. 
            # To explore, we can randomize the order of checking vertices in ties.
            # But for now, let's see if deterministic is good enough.
            # Given the small size, we can also try a few random restarts with different tie-breaking.
            
        if best_colors is None:
            # Fallback to simple greedy if something went wrong
            color = [-1] * m
            for i in range(m):
                used = set()
                for u in conflict_adj[i]:
                    if color[u] != -1:
                        used.add(color[u])
                c = 0
                while c in used:
                    c += 1
                color[i] = c
            best_colors = color
            
        return best_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = strong_coloring(edges, n)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
