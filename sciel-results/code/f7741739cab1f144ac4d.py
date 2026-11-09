import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Deadline guard: 60s limit, keep a safety margin
    deadline = time.time() + 55.0
    
    n = P['n']
    path_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Candidate edges: all non-path undirected edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in path_edges:
                candidates.append((u, v))
                
    # Pre-compute path neighbors for triangle-free check
    path_neighbors = {i: {i-1, i+1} & set(range(n)) for i in range(n)}
    
    # Initial graph: path
    current_edges = set(path_edges)
    current_adj = {i: set() for i in range(n)}
    for u, v in current_edges:
        current_adj[u].add(v)
        current_adj[v].add(u)
        
    def is_triangle_free(u, v):
        # Check if adding (u,v) creates a triangle
        # A triangle exists if there is a w adjacent to both u and v
        # Since the graph is bipartite (path + bipartite edges), we can just check intersection
        # However, to be safe and general, check intersection of neighbors
        # Optimization: if u and v are in the same parity (both even or both odd), 
        # they cannot be connected in a bipartite graph without creating an odd cycle (triangle if dist=2)
        # Actually, the path is bipartite. Any edge added between same parity nodes creates an odd cycle.
        # Is it necessarily a triangle? 
        # If u, v same parity, dist(u,v) is even. If dist(u,v) == 2, adding edge creates triangle.
        # If dist(u,v) > 2, adding edge creates odd cycle of length > 3. 
        # The requirement is "triangle-free". It does NOT say "bipartite". 
        # However, the seed solution uses bipartite edges. 
        # Let's stick to the strict definition: no 3-cycles.
        # But wait, if we add an edge between same parity nodes, do we create a triangle?
        # Only if there is a path of length 2 between them. 
        # In a path, dist(u,v) is fixed. If dist(u,v)==2, yes. 
        # If dist(u,v)>2, no triangle, but an odd cycle. 
        # The problem only forbids triangles. So same-parity edges are allowed IF dist != 2.
        # However, the "bipartite" strategy was successful. 
        # Let's check the constraint: "Triangle-free". 
        # Let's just check the intersection of neighbors.
        
        # If we add (u,v), a triangle is formed if there exists w such that (u,w) in E and (v,w) in E.
        # This is equivalent to: current_adj[u] & current_adj[v] is not empty.
        # Note: u and v are not yet connected in current_adj (assuming we check before adding).
        # Actually, if u,v are already connected, we wouldn't add it.
        
        # Optimization: 
        # If u and v have the same parity in the path, they are at even distance.
        # If dist_path(u,v) == 2, then they share a neighbor in the path. So adding (u,v) creates a triangle.
        # If dist_path(u,v) > 2, they don't share a path-neighbor. 
        # But they might share an added-edge neighbor.
        
        # General check:
        common = current_adj[u] & current_adj[v]
        return len(common) == 0

    def get_diameter():
        # BFS from all nodes to find max shortest path
        # Since n=24, Floyd-Warshall or BFS from each node is fast.
        # BFS from each node is O(N*(N+E)). E is small.
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            for u in queue:
                for v in current_adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_d:
                            max_d = dist[v]
        return max_d

    # Initial diameter
    current_diam = get_diameter()
    
    # Greedy Set Cover / Diameter Reduction
    # Goal: Reduce diameter to <= 4.
    # Strategy: While diameter > 4, find the edge that most reduces the diameter.
    # If multiple edges reduce it, pick one.
    # If no single edge reduces it? 
    # The path has diameter 23. 
    # Adding one edge can reduce it significantly.
    # E.g. adding (0, 23) makes it a cycle C24, diam 12.
    # Adding (0, 12) makes two paths of length 12, diam 12.
    # We need to get to 4.
    
    # It's possible that no single edge reduces diameter below a certain threshold, 
    # or that we need to add multiple edges simultaneously to see the benefit?
    # No, diameter is a monotonic property w.r.t edge addition (adding edges cannot increase diameter).
    # So we can always find the "best" next edge.
    
    # However, "best" in terms of diameter reduction might not lead to the minimal number of edges.
    # But it's a good heuristic.
    
    added_edges = []
    
    while current_diam > 4:
        if time.time() > deadline:
            break
            
        best_edge = None
        best_diam = current_diam
        
        # To speed up, we might not need to check ALL candidates if we have a good guess.
        # But N=24, candidates ~ 276. BFS is fast. 276 * BFS(24 nodes) is very fast.
        
        for u, v in candidates:
            # Skip if already added
            if (u, v) in current_edges:
                continue
                
            # Check triangle-free
            if not is_triangle_free(u, v):
                continue
                
            # Tentatively add
            current_edges.add((u, v))
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            new_diam = get_diam()
            
            # Undo
            current_edges.remove((u, v))
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            # No single edge reduces diameter?
            # This should not happen for diameter > 4 on a path-like graph.
            # If it does, we are stuck.
            break
            
        # Commit best edge
        u, v = best_edge
        current_edges.add((u, v))
        current_adj[u].add(v)
        current_adj[v].add(u)
        added_edges.append([u, v])
        current_diam = best_diam
        
    # Now diameter <= 4 (or we timed out)
    # If we timed out and diam > 4, the result is invalid.
    # We should check validity at the end.
    
    # Greedy Removal: Try to remove edges to minimize count while maintaining constraints.
    # Constraints: Triangle-free, Connected, Diameter <= 4.
    
    # We can only remove added edges. Path edges are pinned.
    # Iterate through added edges (maybe reverse order of addition, or random)
    
    # To be safe, try removing in reverse order of addition (last added is least "essential"?)
    # Or try all.
    
    # Let's try removing each edge and checking validity.
    # If valid, keep it removed.
    
    # Since we want to minimize count, we should try to remove as many as possible.
    # Order doesn't matter much for final count if we check all, but might affect which set we end up with.
    
    # For efficiency, we can try removing edges one by one.
    
    # Convert to list to iterate
    # We need to be careful: removing one edge might make another edge removable.
    # So we should loop until no more edges can be removed.
    
    removed = False
    while not removed:
        removed = True
        # Iterate over a copy of the added edges currently present
        current_added = [e for e in added_edges if (e[0], e[1]) in current_edges]
        
        for u, v in current_added:
            if time.time() > deadline:
                break
                
            # Tentatively remove
            current_edges.remove((u, v))
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            # Check validity
            # 1. Connected? (Diameter check will fail if disconnected, as dist will be -1)
            # 2. Diameter <= 4?
            # 3. Triangle-free? (Removing edges cannot create triangles, so this is always true if it was true before)
            
            new_diam = get_diam()
            
            if new_diam <= 4 and new_diam != -1: # -1 indicates disconnected
                # Keep it removed
                added_edges.remove([u, v])
                removed = False # Restart loop to check other edges again
            else:
                # Put it back
                current_edges.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        if time.time() > deadline:
            break

    # Final validity check
    # If invalid, return empty or best effort? 
    # The task says "Any broken rule makes the whole result invalid".
    # So we must ensure it's valid.
    
    final_diam = get_diam()
    if final_diam > 4 or final_diam == -1:
        # Fallback to a known valid solution if possible?
        # The seed solution is valid.
        return [
            [12, 1],
            [12, 3],
            [12, 5],
            [12, 7],
            [12, 9],
            [12, 15],
            [12, 17],
            [12, 19],
            [12, 21],
            [12, 23]
        ]
        
    # Ensure types are Python int and list
    result = [[int(u), int(v)] for u, v in added_edges]
    return result

# EVOLVE-BLOCK-END
