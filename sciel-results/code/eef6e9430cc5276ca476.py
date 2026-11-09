import sys
import time
import random

P = {'n': 15}

def _construct_graph(n):
    """
    Construct a graph on n vertices maximizing edges such that
    no 3-edge path has endpoints of equal degree.
    """
    start_time = time.time()
    deadline = 50.0  # Leave 10s margin for 60s limit
    
    def is_valid(adj, n):
        """Check validity using bitsets for speed."""
        deg = [sum(adj[i]) for i in range(n)]
        # Precompute neighbors as bitsets
        # adj[i] is a list of 0/1, but for speed we can use int bitmasks
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
        
        # For each edge (u, v), check paths s-u-v-t
        # s in N(u) \ {v}, t in N(v) \ {u}
        # s != t
        # Check if deg[s] == deg[t]
        
        # Optimization: Group vertices by degree
        # But simple check is O(E * deg(u) * deg(v))
        # With n=15, this is small.
        
        for u in range(n):
            for v in range(n):
                if u != v and adj[u][v]:
                    # Neighbors of u excluding v
                    # Neighbors of v excluding u
                    # Iterate s in N(u)
                    for s in range(n):
                        if s != u and s != v and adj[s][u]:
                            # Iterate t in N(v)
                            for t in range(n):
                                if t != s and t != u and t != v and adj[v][t]:
                                    if deg[s] == deg[t]:
                                        return False
        return True

    def count_edges(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
        return count

    # Start with complete graph
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0

    best_adj = [row[:] for row in adj]
    best_count = count_edges(adj, n)
    
    # If K_n is invalid (it is), we need to remove edges.
    # Greedy removal
    current_adj = [row[:] for row in adj]
    current_count = best_count
    
    # We will iteratively remove edges that are part of violations.
    # To keep edge count high, we remove edges "smartly".
    
    for iteration in range(10000):
        if time.time() - start_time > deadline:
            break
            
        if is_valid(current_adj, n):
            # Valid, try to add edges?
            # Or just break if we are doing removal only.
            # Let's try to add back removed edges if possible.
            # For now, just record best.
            if current_count > best_count:
                best_count = current_count
                best_adj = [row[:] for row in current_adj]
            break
        
        # Find a violation
        deg = [sum(current_adj[i]) for i in range(n)]
        violation_edge = None
        for u in range(n):
            if violation_edge: break
            for v in range(n):
                if violation_edge: break
                if u != v and current_adj[u][v]:
                    for s in range(n):
                        if violation_edge: break
                        if s != u and s != v and current_adj[s][u]:
                            for t in range(n):
                                if violation_edge: break
                                if t != s and t != u and t != v and current_adj[v][t]:
                                    if deg[s] == deg[t]:
                                        # Violation found: path s-u-v-t
                                        # Remove one of the edges (s,u), (u,v), (v,t)
                                        # Choose (u,v) as it's central?
                                        # Or random?
                                        # Let's remove (u,v)
                                        violation_edge = (u, v)
                                        break
                    if violation_edge: break
                if violation_edge: break
            if violation_edge: break
        
        if not violation_edge:
            break
            
        u, v = violation_edge
        current_adj[u][v] = 0
        current_adj[v][u] = 0
        current_count -= 1

    # If the greedy removal stopped at a valid graph, use it.
    # If it timed out, use the best valid one found?
    # The loop breaks if valid or time out.
    # If it broke due to time, current_adj might be invalid.
    # We must return a VALID graph.
    # So we need to keep track of the last VALID graph.
    
    # Re-run logic to ensure we return valid
    # Actually, the above loop only updates best_adj when valid.
    # If it never becomes valid, best_adj is K_n (invalid).
    # We need a fallback valid graph.
    
    # Let's do a second pass: Randomized repair
    if not is_valid(best_adj, n):
         # Fallback: Star
         best_adj = [[0]*n for _ in range(n)]
         for v in range(1, n):
             best_adj[0][v] = 1
             best_adj[v][0] = 1
         best_count = n - 1

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges

def construct():
    edges = _construct_graph(P['n'])
    result = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    return result
