import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # 1. Construct Base Edges (Fan Graph)
    edges = set()

    # Base Hamiltonian Cycle: 0-1-...-39-0
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # Chords from 0 to m for m=2..20
    # These create cycles of length m+1 and 41-m
    for m in range(2, 21):
        u = 0
        v = m
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # Initial edge count is 59. Value = 19.
    
    # 2. Attempt to reduce edge count via greedy removal
    # We check if removing an edge preserves pancyclic property.
    
    def is_pancyclic(edge_set):
        # Build adjacency list
        adj = {i: set() for i in range(N)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for cycles of length 3..N
        for k in range(3, N + 1):
            if not has_cycle_of_length(adj, k):
                return False
        return True

    def has_cycle_of_length(adj, k):
        # Simple DFS with depth limit
        # For each start node, try to find a path of length k-1 that returns to start
        # Optimization: Only start from smallest node in the cycle to avoid duplicates? 
        # Or just check existence.
        
        # For k=3, 4, etc., DFS is fast.
        # For larger k, it might be slow if we do naive DFS.
        # But N=40 is small.
        
        for start in range(N):
            # Try to find a cycle starting and ending at 'start'
            # Path: start -> v1 -> ... -> v_{k-1} -> start
            # We can use a set for visited nodes in the current path.
            
            # To speed up, we can prune.
            # If k is large, we can just check if the graph is connected? No.
            
            # Let's use a simple recursive search with backtracking.
            # Since we only need ONE cycle, we can stop early.
            
            # Optimization: For k > N/2, checking N-k might be faster?
            # No, cycle length is specific.
            
            # Use a stack for DFS
            # State: (current_node, path_list)
            # Path list contains nodes in order.
            
            # To avoid re-visiting same sets, we can use a memo of (current_node, frozenset(path))?
            # Might be too heavy.
            
            # Simple DFS:
            def dfs(u, depth, visited, path):
                if depth == k:
                    # Check if u is connected to start
                    if start in adj[u]:
                        return True
                    return False
                
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        if dfs(v, depth + 1, visited, path):
                            return True
                        path.pop()
                        visited.remove(v)
                return False
            
            # Start DFS
            visited = {start}
            if dfs(start, 1, visited, [start]):
                return True
        
        return False

    # Greedy Removal
    # Sort edges: Chords first (more likely to be redundant if interactions exist), then cycle edges.
    # Actually, cycle edges are critical for the Hamiltonian cycle (length 40).
    # If we remove a cycle edge, we might lose length 40.
    # So we should try removing chords first.
    
    chord_edges = [(u, v) for u, v in edges if u == 0 or v == 0]
    cycle_edges = [(u, v) for u, v in edges if u != 0 or v != 0]
    
    # Sort chords by 'm' value? 
    # Let's just try all edges.
    all_edges = list(edges)
    # Heuristic: Try removing chords first
    all_edges.sort(key=lambda e: 0 if 0 in e else 1)
    
    for e in all_edges:
        if time.time() > deadline:
            break
            
        # Try removing e
        new_edges = edges - {e}
        
        # Quick check: If we removed a cycle edge, do we still have a Hamiltonian cycle?
        # We can check connectivity or just run is_pancyclic.
        # is_pancyclic includes length 40 check.
        
        if is_pancyclic(new_edges):
            edges = new_edges
            # If we found a 58-edge graph, we can stop or continue?
            # Continue to see if we can get 57? Unlikely.
            # But 58 is the target.
            # We can break if len(edges) == 58 to save time?
            # Or continue to try for 57.
            # Let's continue.
            
    # 3. Construct Cycles for the final edge set
    # We need to provide explicit cycles for the returned edge set.
    # The seed logic for cycles might not work if edges were removed.
    # We need to find cycles for the final graph.
    
    # Rebuild adjacency for cycle finding
    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycles = []
    
    for k in range(3, N + 1):
        # Find a cycle of length k
        cycle = None
        for start in range(N):
            # DFS to find cycle of length k starting/ending at start
            def dfs_cycle(u, depth, visited, path):
                if depth == k:
                    if start in adj[u]:
                        return path + [start]
                    return None
                
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        res = dfs_cycle(v, depth + 1, visited, path)
                        if res:
                            return res
                        path.pop()
                        visited.remove(v)
                return None
            
            visited = {start}
            cycle = dfs_cycle(start, 1, visited, [start])
            if cycle:
                break
        
        if cycle:
            cycles.append(cycle)
        else:
            # Fallback: This should not happen if is_pancyclic was correct
            # But just in case, return a dummy or raise error?
            # The task requires valid cycles.
            # If we failed to find, the graph is not pancyclic.
            # We should revert to a known good graph if this happens.
            # For now, we assume is_pancyclic is correct.
            pass

    edge_list = list(edges)

    return {'edges': edge_list, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    for c in res['cycles']:
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (u,v) not in edges and (v,u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c}")
                break
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
