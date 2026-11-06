import random
import time
import sys

def construct():
    n = 40
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    def has_path_length_5(graph, u, v):
        # Check if there is a simple path of length 5 from u to v in graph
        # graph is a set of edges or adjacency list
        # We use adjacency list for speed
        if u == v:
            return False
        visited = [set() for _ in range(n)]
        visited[0].add(u)
        
        for step in range(1, 6):
            for node in visited[step - 1]:
                for neighbor in graph[node]:
                    if neighbor not in visited[step]:
                        # Check if neighbor was visited in any previous step
                        # If so, it's not part of a simple path of this exact length 
                        # that avoids previous nodes? 
                        # Actually, we just need *a* path of length 5.
                        # If neighbor is in visited[k] for k < step, then there is a shorter path.
                        # But we want a path of length exactly 5.
                        # If we reach v in 5 steps, and the path is simple, it's a C6.
                        # The BFS layering ensures simplicity if we don't revisit nodes.
                        # However, if a node is reachable in 2 steps and 5 steps, 
                        # the 5-step path might not be simple if it loops.
                        # But BFS finds shortest paths. 
                        # If shortest path is < 5, there might still be a longer simple path.
                        # Example: u-a-b-v (len 3). Also u-c-d-e-f-v (len 5).
                        # My BFS above only tracks the *first* time a node is seen?
                        # No, I'm adding to visited[step] if not in visited[step].
                        # But I should check if it's in ANY previous visited set to ensure simplicity?
                        # Actually, if I just want to know if *any* simple path of length 5 exists,
                        # the standard BFS for shortest path is not enough.
                        # I need to check if v is reachable in exactly 5 steps via a simple path.
                        
                        # A simpler check: 
                        # Is there a path u=x0, x1, x2, x3, x4, x5=v with distinct xi?
                        # This is equivalent to checking if v is in the set of vertices 
                        # reachable from u in exactly 5 steps without repeating vertices.
                        
                        # Given n=40, we can just do a DFS with pruning or 
                        # use the fact that if the shortest path is < 5, 
                        # we still need to check for longer paths.
                        
                        # However, for C6 detection, if there is ANY path of length 5, 
                        # does it imply a C6? 
                        # If the path is not simple, it contains a cycle. 
                        # If the path u...v has length 5 and is not simple, 
                        # it might contain a C4 or C3. 
                        # But if the graph is C3-free and C4-free, then any path of length 5 is simple?
                        # We are not maintaining C3/C4 freedom.
                        
                        # Let's stick to a robust check: 
                        # Check if there exists a path of length 5.
                        # We can use DP: dp[k][v] = True if there is a path of length k from u to v.
                        # But this allows repeated vertices.
                        # If a path of length 5 exists, does it create a C6?
                        # If the path is u-a-b-c-d-v, and all distinct, yes.
                        # If not distinct, e.g., u-a-b-a-c-v, then u-a-b-a is a C3? No, u-a, a-b, b-a is not a cycle in simple graph (no multi-edges). 
                        # In a simple graph, a path is a sequence of distinct vertices.
                        # So "path" usually implies simple path.
                        # So we need to check for a simple path of length 5.
                        
                        pass

        return False

    # The above logic is getting complicated. 
    # Let's use a simpler sufficient condition.
    # If N(u) intersect N(v) contains a C4, adding (u,v) creates C6.
    # This is a subset of cases.
    # But maybe good enough?
    
    # Let's try a different check: 
    # Adding (u,v) creates a C6 iff there is a path of length 5 between u and v.
    # We can check this by looking at neighbors.
    # Let A = N(u), B = N(v).
    # We need a path u-a-b-c-d-v.
    # a in A, d in B.
    # Path a-b-c-d of length 3.
    # So we need to check if there is a path of length 3 between any a in A and any d in B.
    # This is still hard.
    
    # Given the time, let's use the BFS check for path length 5, 
    # assuming that if a path of length 5 exists, it's simple enough to form a C6 
    # OR we can just check for path length 5 and assume it's safe to block it.
    # If we block all edges that create a path of length 5, we are safe.
    
    def creates_c6(adj, u, v):
        # Check if there is a simple path of length 5 from u to v
        # Using BFS with state (node, depth) is not enough for simplicity.
        # But for n=40, we can just do a DFS from u, limiting depth to 5, 
        # and checking if we reach v.
        # To ensure simplicity, we track visited nodes in the current path.
        
        def dfs(node, depth, visited):
            if depth == 5:
                return node == v
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1, visited):
                        return True
                    visited.remove(neighbor)
            return False
        
        return dfs(u, 0, {u})

    # This DFS might be slow if called 780 * 1000 times.
    # 780 * 1000 = 780,000 calls.
    # Each DFS explores up to 5 levels. Branching factor ~10. 10^5 = 100,000 nodes.
    # 780,000 * 100,000 is too slow.
    
    # We need a faster check.
    # How about checking if the number of paths of length 5 is > 0?
    # We can compute the number of walks of length 5 using matrix multiplication?
    # A^5[u][v] > 0.
    # This counts walks, not simple paths.
    # If A^5[u][v] > 0, there is a walk.
    # If there is a walk, there is a simple path? 
    # Not necessarily. But if there is a walk of length 5, and the graph is small, 
    # it's likely there is a simple path.
    # However, if the walk repeats vertices, it might not form a C6.
    # But if we block edges that create ANY walk of length 5, we might be too conservative?
    # No, we want to AVOID C6. If a walk exists, it might contain a C6 or a smaller cycle.
    # If it contains a C6, we must block.
    # If it contains a C4, we might not care? No, we only care about C6.
    # But if a walk of length 5 exists, does it imply a C6?
    # If the walk is simple, yes.
    # If not, it has a cycle.
    # If the cycle is C6, yes.
    # If the cycle is C4, then the walk is u...C4...v.
    # This doesn't necessarily create a C6 when adding (u,v).
    # Example: u-a-b-c-a-d-v. Walk length 5. Cycle a-b-c-a is C3? No, C3.
    # If graph has C3, this is possible.
    
    # Let's assume the DFS is fast enough for a few hundred iterations.
    # Or we can optimize the DFS.
    
    # Alternative: Use the property that if G is C4-free, then C6 check is easier.
    # But we don't enforce C4-free.
    
    # Let's try the greedy with the DFS check and see if it fits in time.
    # We can limit the number of iterations.
    
    for _ in range(100): # Try 100 random orders
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        adj = {i: set() for i in range(n)}
        edges = []
        
        for u, v in all_edges:
            if creates_c6(adj, u, v):
                continue
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges
