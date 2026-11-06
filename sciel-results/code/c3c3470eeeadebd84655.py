import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def _get_30_vertex_edges():
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
        return edges

    def _is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def _has_independent_set(adj, n, target):
        """Check if there's an independent set of size target."""
        if n < target:
            return False
        
        # Use a simple backtracking with pruning
        adj_list = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                adj_list[u].add(v)
        
        # Try to find independent set of size target
        # Use greedy coloring bound
        def color_bound(remaining):
            """Upper bound on independent set size in remaining graph."""
            if not remaining:
                return 0
            # Greedy coloring
            uncolored = set(remaining)
            colors = 0
            while uncolored:
                colors += 1
                # Find maximal independent set
                current = []
                while uncolored:
                    v = min(uncolored, key=lambda x: len(uncolored & adj_list[x]))
                    current.append(v)
                    uncolored -= adj_list[v]
                    uncolored.discard(v)
                # All vertices in current get this color
                # But we need to remove them from uncolored for next color
            return colors
        
        found = [False]
        
        def backtrack(candidates, current_size):
            if found[0]:
                return
            if current_size == target:
                found[0] ==True
                return
            if not candidates:
                return
            if time.time() > deadline:
                return
            
            # Pruning: if we can't reach target
            max_possible = current_size + len(candidates)
            if max_possible < target:
                return
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_candidates = [x for x in rest if x not in adj_list[v]]
            backtrack(new_candidates, current_size + 1)
            
            # Exclude v
            if not found[0]:
                backtrack(rest, current_size)
        
        backtrack(list(range(n)), 0)
        return found[0]

    def _verify_graph(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        if not _is_c4_free(adj, n):
            return False
        
        if _has_independent_set(adj, n, 11):
            return False
        
        return True

    best_result = None
    best_n = 0
    
    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + n * 104729)
            
            adj = [set() for _ in range(n)]
            edges = []
            
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                common_uv = adj[u] & adj[v]
                if len(common_uv) >= 2:
                    continue
                
                ok = True
                for w in adj[u]:
                    if w == v:
                        continue
                    if adj[w] & adj[v]:
                        ok = False
                        break
                if not ok:
                    continue
                
                for w in adj[v]:
                    if w == u:
                        continue
                    if adj[w] & adj[u]:
                        ok = False
                        break
                if not ok:
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
            
            if not _has_independent_set(adj, n, 11):
                best_result = {'n': n, 'edges': edges}
                best_n = n
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    edges = _get_30_vertex_edges()
    return {'n': 30, 'edges': edges}
