import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Function to check if adding (u, v) creates a C6
    def would_create_c6(u, v, adj):
        # We need to find a simple path of length 5 from u to v in the graph defined by adj
        # Path: u - a - b - c - d - v
        # a is neighbor of u
        # b is neighbor of a
        # c is neighbor of b
        # d is neighbor of c
        # v is neighbor of d
        
        # Optimization: If deg(u) == 0 or deg(v) == 0, return False
        if not adj[u] or not adj[v]:
            return False
            
        # We can do a DFS from u up to depth 4, checking if any node at depth 4 is connected to v
        # And ensuring the path is simple.
        
        # visited set for the path
        # Since depth is small (5), we can just pass the path or a visited mask/set.
        
        # To make it faster, we can use a set of visited nodes in the current path.
        
        def dfs(current, depth, visited):
            if depth == 4:
                # Check if v is a neighbor of current
                if v in adj[current]:
                    return True
                return False
            
            for nxt in adj[current]:
                if nxt in visited:
                    continue
                if nxt == v:
                    # We can't go to v until depth 4
                    # Wait, the path is u-a-b-c-d-v.
                    # So at depth 4 (node d), we check if v is a neighbor.
                    # If we reach v at depth < 4, that's a shorter path, not a C6.
                    # But we are looking for a path of length exactly 5.
                    # If we hit v early, that path doesn't form a C6 with (u,v) because the cycle would be shorter.
                    # However, we just need to know if ANY simple path of length 5 exists.
                    # So we should avoid going to v early?
                    # If we go to v at depth 2, that's a path of length 2. Adding (u,v) makes a C3.
                    # But we are checking for C6.
                    # So we should not consider paths that hit v before depth 4.
                    continue
                
                visited.add(nxt)
                if dfs(nxt, depth + 1, visited):
                    visited.remove(nxt)
                    return True
                visited.remove(nxt)
            
            return False
        
        visited = {u}
        return dfs(u, 0, visited)

    # Greedy construction
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.shuffle(all_edges)
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if would_create_c6(u, v, adj):
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # Optional: Try to improve best_edges with local search?
            # For now, just keep the best greedy result.
            
        # If we found a very high number, maybe we can stop early?
        # Best known is 105. Total edges 780.
        # If we reach 105, we are at the known bound.
        # Let's aim for > 90 to be safe.
        
    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
