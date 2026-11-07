    def solve(n, edges, lists):
        import sys
        sys.setrecursionlimit(10000)
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sort vertices by degree descending for a good static order
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
            
        color = [None] * n
        avail = [list(lists[v]) for v in range(n)]
        
        # Precompute neighbors in order of assignment (only those assigned before v)
        # Actually, for backtracking we check all neighbors, but only colored ones matter.
        # We can optimize by only checking neighbors that come earlier in the order.
        pred_neighbors = [[] for _ in range(n)]
        for v in range(n):
            for u in adj[v]:
                if pos[u] < pos[v]:
                    pred_neighbors[v].append(u)
                    
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Try each available color
            # Sort colors by how many they conflict with already colored neighbors?
            # Or just static list order.
            # Forward checking: when assigning c to v, we can remove c from avail of uncolored neighbors.
            # But simple backtracking is often sufficient for 3-coloring bipartite graphs which are easy.
            # The Alon-Tarsi theorem guarantees a solution.
            
            # Heuristic: try colors that conflict with the least already-colored neighbors?
            # Or just iterate.
            for c in avail[v]:
                # Check consistency with already colored predecessors
                conflict = False
                for u in pred_neighbors[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                color[v] = c
                
                # Recurse
                if backtrack(idx + 1):
                    return True
                
                color[v] = None
            
            return False
        
        if backtrack(0):
            return color
        else:
            # Fallback: should not happen
            return [0] * n
