"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    n = P['n']
    # Adjacency matrix for fast access
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        
    added_edges = []
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def get_diameter():
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_diam:
                    max_diam = d
        return max_diam

    def get_sum_distances():
        total = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            total += sum(d for d in dist if d != -1)
        return total

    def is_valid(u, v):
        if adj[u][v]:
            return False
        # Check for triangle: no common neighbor
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                return False
        return True

    while True:
        if time.time() > deadline - 5.0:
            break
            
        current_diam = get_diameter()
        if current_diam <= 4:
            break
            
        best_edge = None
        best_metric = (2, float('inf'), float('inf')) # (priority, diameter, sum_dist)
        
        for u, v in all_pairs:
            if not is_valid(u, v):
                continue
                
            # Add edge
            adj[u][v] = True
            adj[v][u] = True
            
            new_diam = get_diameter()
            
            if new_diam <= 4:
                # If we achieve the goal, we prefer this edge
                # We still want to minimize sum distances among those that achieve diam <= 4
                # But actually, any edge that achieves diam <= 4 is good.
                # To minimize total edges, we just need one.
                # However, a better choice now might lead to fewer edges later?
                # Usually, achieving the goal is the primary objective.
                # Let's pick the one with min sum distances among those achieving diam <= 4
                sum_dist = get_sum_distances()
                metric = (0, new_diam, sum_dist)
            else:
                # If not achieved, minimize diameter, then sum distances
                sum_dist = get_sum_distances()
                metric = (1, new_diam, sum_dist)
                
            # Remove edge
            adj[u][v] = False
            adj[v][u] = False
            
            if metric < best_metric:
                best_metric = metric
                best_edge = (u, v)
                
        if best_edge is None:
            # No valid edge found to improve
            break
            
        u, v = best_edge
        adj[u][v] = True
        adj[v][u] = True
        added_edges.append([u, v])
        
    # Ensure return type is list of lists of ints
    return added_edges

# EVOLVE-BLOCK-END
