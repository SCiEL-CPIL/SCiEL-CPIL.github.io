import random
import time

def construct():
    n = 15
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    # Initialize best result
    best_edges = []
    best_count = 0
    
    # Helper to check validity
    def is_valid(edges):
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        degrees = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        
        # Check all paths s-u-v-t
        # For each edge (u, v)
        for u, v in edges:
            # s is neighbor of u, t is neighbor of v
            # s != v, t != u, s != t
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if degrees[s] == degrees[t]:
                        return False
        return True

    # Helper to get edge set for fast lookup
    def to_set(edges):
        return set(tuple(sorted((u, v))) for u, v in edges)

    # Initialize with a valid structure: 5 disjoint triangles
    # Triangles: (0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14)
    initial_edges = []
    for i in range(0, n, 3):
        initial_edges.append([i, i+1])
        initial_edges.append([i+1, i+2])
        initial_edges.append([i, i+2])
    
    if is_valid(initial_edges):
        best_edges = initial_edges[:]
        best_count = len(initial_edges)
    else:
        # Fallback to star if triangles fail (they shouldn't)
        best_edges = [[0, v] for v in range(1, n)]
        best_count = n - 1

    # Local Search
    random.seed(42)
    current_edges = best_edges[:]
    current_set = to_set(current_edges)
    
    max_iters = 100000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        
        # Try to add a random edge
        added = False
        # Generate random non-edge
        attempts = 0
        while attempts < 10:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            edge_key = (min(u,v), max(u,v))
            if edge_key not in current_set:
                # Try adding
                new_edges = current_edges + [[u, v]]
                if is_valid(new_edges):
                    current_edges = new_edges
                    current_set.add(edge_key)
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = current_edges[:]
                    added = True
                    break
            attempts += 1
        
        if not added:
            # Try a random move: remove one edge, add another
            # This helps escape local optima where no single edge can be added
            if len(current_edges) > 0:
                remove_idx = random.randint(0, len(current_edges) - 1)
                rem_u, rem_v = current_edges[remove_idx]
                rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
                
                # Try to add a different edge
                attempts = 0
                while attempts < 10:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v:
                        continue
                    add_key = (min(u,v), max(u,v))
                    if add_key == rem_key:
                        continue
                    if add_key in current_set:
                        continue
                    
                    # Simulate move
                    temp_edges = current_edges[:]
                    temp_edges.pop(remove_idx)
                    temp_edges.append([u, v])
                    
                    if is_valid(temp_edges):
                        # Accept move (edge count stays same, structure changes)
                        current_edges = temp_edges
                        current_set.discard(rem_key)
                        current_set.add(add_key)
                        # We don't update best_count here as it's same count
                        break
                    attempts += 1

    # Final check and format
    # Ensure types are correct
    final_edges = []
    for e in best_edges:
        final_edges.append([int(e[0]), int(e[1])])
        
    return final_edges
