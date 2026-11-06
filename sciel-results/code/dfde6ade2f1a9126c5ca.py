import random
import time
import itertools

def construct():
    n = 18
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Helper to check if a subset is triangle-free
    def is_triangle_free(subset, adj):
        # subset is a tuple of vertices
        # Check all triples
        s = list(subset)
        for i in range(len(s)):
            u = s[i]
            for j in range(i + 1, len(s)):
                v = s[j]
                if v in adj[u]:
                    for k in range(j + 1, len(s)):
                        w = s[k]
                        if w in adj[u] and w in adj[v]:
                            return False
        return True

    # Helper to find the largest triangle-free induced subgraph size
    def get_score(adj):
        # Check from n down to 1
        for k in range(n, 0, -1):
            # Iterate all subsets of size k
            for subset in itertools.combinations(range(n), k):
                if is_triangle_free(subset, adj):
                    return k
        return 0

    best_score = n
    best_edges = []
    
    # Run multiple random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Random greedy construction of a maximal K4-free graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # Generate random order of edges
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
                
            # Check if adding (u, v) creates a K4
            # A K4 is formed if there is an edge (x, y) in common neighbors of u and v
            common = adj[u] & adj[v]
            creates_k4 = False
            # Optimization: iterate common neighbors
            # If any two common neighbors are connected, K4 is formed
            # Actually, we just need to check if the subgraph induced by common neighbors has an edge.
            # common is a set.
            # We can check pairs in common.
            # Size of common is at most 16.
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    if y in adj[x]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        # Evaluate score
        score = get_score(adj)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            
        # If we hit a very low score, we can stop early
        if best_score <= 10:
            break

    return best_edges
