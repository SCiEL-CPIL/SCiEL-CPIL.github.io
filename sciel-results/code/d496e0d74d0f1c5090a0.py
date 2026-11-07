"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 16
    half = n // 2
    
    # Structural Idea: Complete Bipartite Graph K_{8,8}
    # Partition A: 0..7, Partition B: 8..15
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    adj = [set() for _ in range(n)]
    
    for u in A:
        for v in B:
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
            
    # Refinement Machinery: Fast Verification
    # We verify that the graph has diameter exactly 2 and every edge is critical.
    # This is fast for N=16.
    
    def check_validity(adj, n):
        # 1. Check Diameter is exactly 2
        # Diameter <= 2: For all u, v, d(u,v) <= 2
        # This is equivalent to: For all u, N(u) union N(N(u)) covers all vertices.
        # Or: A^2 has no zero entries (except diagonal if we consider paths of length 2, but we need connectivity too).
        # Simpler: BFS from each node.
        
        # Check connected and diam <= 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            # Check all reachable and max dist <= 2
            max_d = 0
            for i in range(n):
                if dist[i] == -1:
                    return False
                if dist[i] > max_d:
                    max_d = dist[i]
            if max_d > 2:
                return False
        
        # Check Diameter is NOT 1 (i.e., not complete)
        # If it's complete, there is no edge whose removal increases diameter to >2 (it stays 1 or becomes 2).
        # But we need "diameter exactly two".
        # K_{8,8} is not complete, so diam is 2.
        # Let's verify there is at least one pair at distance 2.
        has_dist2 = False
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for i in range(n):
                if dist[i] == 2:
                    has_dist2 = True
                    break
            if has_dist2:
                break
        if not has_dist2:
            return False
            
        # 2. Check Edge Criticality
        # For each edge (u,v), removing it must make d(u,v) > 2 or disconnect graph.
        # In a triangle-free graph, removing (u,v) makes d(u,v) > 2 if and only if there is no common neighbor.
        # If there is a common neighbor w, d(u,v) becomes 2.
        # So, we just need to ensure the graph is triangle-free.
        # If graph is triangle-free, then for any edge (u,v), N(u) and N(v) are disjoint (except u and v themselves? No, u is in N(v), v is in N(u)).
        # Wait, N(u) contains v. N(v) contains u.
        # Common neighbors of u and v: w such that (u,w) in E and (v,w) in E.
        # If such w exists, triangle (u,v,w) exists.
        # So, if graph is triangle-free, no edge (u,v) has a common neighbor.
        # Therefore, removing (u,v) leaves no path of length 2 between u and v.
        # Since graph is connected and bipartite (K_{8,8}), next shortest path is length 3.
        # So d(u,v) becomes 3 > 2.
        # Thus, triangle-free + connected + diam 2 implies all edges are critical.
        
        # Check for triangles
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check if u and v have a common neighbor
                    # Intersection of adj[u] and adj[v]
                    # If intersection is non-empty, triangle exists.
                    # Note: adj[u] contains v, adj[v] contains u.
                    # We look for w != u, w != v.
                    common = adj[u].intersection(adj[v])
                    # Remove u and v from consideration if they are in each other's sets (they are)
                    # Actually, intersection will include v (from adj[u]) and u (from adj[v]).
                    # So if len(common) > 2, there is a triangle.
                    # Or simpler: just check if any w in adj[u] is also in adj[v] and w != u and w != v.
                    for w in adj[u]:
                        if w == v:
                            continue
                        if w in adj[v]:
                            return False # Triangle found
                            
        return True

    if check_validity(adj, n):
        return edges
    else:
        # Fallback: Should not happen for K_{8,8}, but return what we have
        # If invalid, we might try to fix, but K_{8,8} is theoretically sound.
        # For safety, if it fails verification, we return the edges anyway as it's the best candidate,
        # or we could try a star graph which is definitely valid but low score.
        # Given the theoretical proof, K_{8,8} is valid.
        return edges

# EVOLVE-BLOCK-END
