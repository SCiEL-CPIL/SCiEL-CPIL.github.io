import time
import sys
from collections import deque
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    # Initialize graph with the path
    adj = [set() for _ in range(n)]
    initial_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        initial_edges.add((u, v) if u < v else (v, u))
    
    # Precompute all valid candidate edges (triangle-free, not existing)
    # An edge (u,v) is valid if u,v not adjacent and no common neighbor
    # For a path, two vertices u < v have a common neighbor iff |u-v| <= 2
    # Actually, let's be precise: in a path, u and v share a neighbor w iff
    # w is adjacent to both. In a path, each vertex has at most 2 neighbors.
    # So u and v share a neighbor iff they are distance 2 apart (w is between them)
    # or if one of them is adjacent to the other's neighbor... 
    # Let's just compute it properly.
    
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in initial_edges:
                continue
            # Check triangle-free: no common neighbor
            if adj[u] & adj[v]:
                continue
            all_candidates.append((u, v))
    
    # We need to find the minimum number of edges from all_candidates
    # such that the resulting graph has diameter <= 4.
    # 
    # Key insight: The path P24 has diameter 23. We need to reduce it to <= 4.
    # With k added edges, we can create "shortcuts". 
    # 
    # Let's think about what structure is needed.
    # If we add edges, we're creating a graph that is a path plus some chords.
    # The diameter is the max shortest path between any two vertices.
    #
    # A known result: For a path on n vertices, the minimum number of edges to add
    # to make the diameter <= d is related to covering the path with "jumps".
    #
    # Let's use a different approach: BFS/DFS over the space of edge sets is too large.
    # Instead, let's use a greedy approach with a good heuristic, or try to find
    # a small set of edges that "bridges" the path.
    #
    # Actually, let's think about it this way:
    # If we add an edge (i, j) with j > i, it creates a shortcut from i to j.
    # The path can then be traversed with jumps.
    #
    # Let's try a different structural approach:
    # The graph must be triangle-free. The added edges must form a "matching" or
    # at least not create triangles with the path edges or each other.
    #
    # Actually, the added edges can share endpoints as long as no triangle is formed.
    # A triangle is formed if we have edges (a,b), (b,c), (c,a) all present.
    # Since the path already has (a,a+1) and (a+1,a+2), adding (a, a+2) would create
    # a triangle. So we cannot add edges of length 2.
    # Similarly, if we add (a, b) and (b, c), we need to ensure (a, c) is not present
    # and not added.
    #
    # Let's try a constructive approach:
    # We want to reduce the diameter to 4. 
    # Consider adding edges that create a "hub" structure or a "ladder" structure.
    #
    # Actually, let's try to find the minimum k by trying k=1, k=2, ... and checking
    # if there exists a set of k edges that achieves diameter <= 4.
    #
    # For small k, we can enumerate all combinations of k edges from all_candidates.
    # The number of candidates is O(n^2) = ~276. 
    # For k=1: 276 choices.
    # For k=2: 276*275/2 ~ 38k choices.
    # For k=3: ~3.5M choices. This is feasible.
    # For k=4: ~300M choices. This is too many.
    #
    # So let's try k=1, k=2, k=3 first. If none work, we fall back to a greedy approach.
    
    def check_diameter(adj, added_edges):
        # Compute diameter of the graph
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check connectivity
            if any(d == -1 for d in dist):
                return -1  # Not connected
            max_diam = max(max_diam, max(dist))
            if max_diam > 4:
                return max_diam
        return max_diam
    
    # First, let's try to find a solution with k=1, 2, 3 edges by enumeration.
    # We'll use a smarter check: instead of full BFS for each candidate,
    # we can precompute distances in the path and use the added edges as shortcuts.
    
    # Actually, let's just do the enumeration for small k.
    # For each candidate set, we build the graph and check diameter.
    
    def try_k_edges(k):
        if k == 0:
            return None
        
        for combo in combinations(range(len(all_candidates)), k):
            # Build the graph
            test_adj = [set() for _ in range(n)]
            for u, v in P['edges']:
                test_adj[u].add(v)
                test_adj[v].add(u)
            
            # Check if adding these edges creates a triangle
            valid = True
            for idx in combo:
                u, v = all_candidates[idx]
                # Check triangle with path edges
                # This is already ensured by all_candidates being triangle-free
                # But we also need to check triangles between added edges
                # Actually, all_candidates are individually triangle-free with the path.
                # But two added edges might create a triangle with a path edge or each other.
                # Let's just add them and check.
                test_adj[u].add(v)
                test_adj[v].add(u)
            
            # Check for triangles in the resulting graph
            for i in range(n):
                for j in test_adj[i]:
                    if j > i:
                        for k2 in test_adj[j]:
                            if k2 > j and k2 in test_adj[i]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check diameter
            diam = check_diameter(test_adj, combo)
            if 0 <= diam <= 4:
                # Found a solution
                return [list(all_candidates[idx]) for idx in combo]
        
        return None
    
    # Try k=1, 2, 3
    for k in range(1, 4):
        result = try_k_edges(k)
        if result is not None:
            return result
    
    # If no solution found with <= 3 edges, use a greedy approach
    # This should not happen, but let's be safe.
    
    added_edges = []
    deadline = time.time() + 60.0
    
    while time.time() < deadline:
        # Check current diameter
        diam = check_diameter(adj, added_edges)
        if diam <= 4:
            break
        
        # Find the pair with maximum distance
        max_pair = (0, 0)
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for end in range(n):
                if dist[end] > max_dist:
                    max_dist = dist[end]
                    max_pair = (start, end)
        
        u_far, v_far = max_pair
        
        # Find the best edge to add
        best_edge = None
        best_new_diam = diam + 1
        
        for u, v in all_candidates:
            # Skip if already added
            if (u, v) in [(e[0], e[1]) for e in added_edges]:
                continue
            if (v, u) in [(e[0], e[1]) for e in added_edges]:
                continue
            
            # Temporarily add
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam = check_diameter(adj, added_edges)
            
            # Remove
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
            
            if new_diam <= 4:
                break
        
        if best_edge is None:
            break
        
        adj[best_edge[0]].add(best_edge[1])
        adj[best_edge[1]].add(best_edge[0])
        added_edges.append(list(best_edge))
    
    # Return the result
    return [[int(e[0]), int(e[1])] for e in added_edges]
# EVOLVE-BLOCK-END
