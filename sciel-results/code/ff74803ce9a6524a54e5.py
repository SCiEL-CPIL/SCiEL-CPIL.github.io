import random
import time

def construct():
    n = 15
    
    def get_edges_set():
        # Returns list of edges as (u, v) with u < v
        pass

    def solve():
        # Start with complete graph
        edges = set((i, j) for i in range(n) for j in range(i+1, n))
        
        # Precompute adjacency for speed
        adj = [[False]*n for _ in range(n)]
        deg = [0]*n
        
        def update_graph():
            # Rebuild adj and deg from edges
            for i in range(n):
                for j in range(n):
                    adj[i][j] = False
                deg[i] = 0
            for u, v in edges:
                adj[u][v] = True
                adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
        
        update_graph()
        
        while True:
            # Find violations
            # A violation is a path s-u-v-t with s!=t, deg[s]==deg[t]
            # Edges involved: (s,u), (u,v), (v,t)
            # We count how many times each edge appears in a violation
            violation_counts = {}
            
            # Iterate over all middle edges (u,v)
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v]:
                        continue
                    # s must be neighbor of u, s != v
                    # t must be neighbor of v, t != u
                    # s != t
                    # deg[s] == deg[t]
                    
                    # To speed up, group neighbors by degree
                    # But n=15 is small, O(n^4) is fine (15^4 ~ 50k)
                    for s in range(n):
                        if s == v or not adj[s][u]:
                            continue
                        for t in range(n):
                            if t == u or t == s or not adj[v][t]:
                                continue
                            if deg[s] == deg[t]:
                                # Violation found
                                # Mark edges (s,u), (u,v), (v,t)
                                e1 = (min(s,u), max(s,u))
                                e2 = (u,v)
                                e3 = (min(v,t), max(v,t))
                                violation_counts[e1] = violation_counts.get(e1, 0) + 1
                                violation_counts[e2] = violation_counts.get(e2, 0) + 1
                                violation_counts[e3] = violation_counts.get(e3, 0) + 1
                                
            if not violation_counts:
                return list(edges)
            
            # Remove edge with max count
            # Random tie-breaking
            max_count = max(violation_counts.values())
            candidates = [e for e, c in violation_counts.items() if c == max_count]
            edge_to_remove = random.choice(candidates)
            
            edges.remove(edge_to_remove)
            update_graph()
            
        return list(edges)

    best_edges = []
    start_time = time.time()
    while time.time() - start_time < 50: # Leave 10s margin
        res = solve()
        if len(res) > len(best_edges):
            best_edges = res
            
    # Ensure types are correct
    final_edges = []
    for u, v in best_edges:
        if isinstance(u, bool) or isinstance(v, bool):
            continue
        if not isinstance(u, int) or not isinstance(v, int):
            continue
        final_edges.append([int(u), int(v)])
        
    return final_edges
