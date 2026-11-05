import random
import itertools

def construct():
    n = 20
    
    def count_common_neighbors(adj, u, v):
        return len(adj[u] & adj[v])
    
    def is_valid(adj, u, v):
        if count_common_neighbors(adj, u, v) > 2:
            return False
        for w in adj[u]:
            if w != v and count_common_neighbors(adj, u, w) > 1:
                return False
        for w in adj[v]:
            if w != u and count_common_neighbors(adj, v, w) > 1:
                return False
        return True

    def local_search(adj, max_iter=2000):
        current_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges.add((u, v))
        
        best_edges = set(current_edges)
        best_count = len(best_edges)
        
        for _ in range(max_iter):
            # Try adding an edge
            all_pairs = list(itertools.combinations(range(n), 2))
            random.shuffle(all_pairs)
            added = False
            for u, v in all_pairs:
                if (u, v) in current_edges:
                    continue
                if is_valid(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    current_edges.add((u, v))
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                    added = True
                    break
            
            if not added:
                # Try a swap: remove (a,b), add (c,d)
                # To keep it simple, we just break.
                # A full swap search is O(E^2) per step, which might be too slow.
                break
                
        return best_edges

    best_edges_overall = set()
    best_count_overall = 0
    
    for restart in range(100):
        random.seed(restart)
        adj = [set() for _ in range(n)]
        current_edges = set()
        
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        for u, v in edges:
            if is_valid(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.add((u, v))
        
        improved_edges = local_search(adj)
        
        if len(improved_edges) > best_count_overall:
            best_count_overall = len(improved_edges)
            best_edges_overall = improved_edges
            
    return [[u, v] for u, v in best_edges_overall]
