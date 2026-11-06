"""Rewritten program: 4-partite Turan base + local search to push n above 32."""
import random

def construct():
    def has_c5(adj, n):
        # Precompute neighbor lists
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # For each edge (u,v), check if there's a path of length 4 from u to v not using edge (u,v)
        # Path: u-a-b-c-v with a,b,c distinct from u,v and from each other
        for u in range(n):
            for v in neighbors[u]:
                if v <= u:
                    continue
                # a in neighbors[u] \ {v}
                for a in neighbors[u]:
                    if a == v:
                        continue
                    # c in neighbors[v] \ {u, a}
                    for c in neighbors[v]:
                        if c == u or c == a:
                            continue
                        # b in neighbors[a] ∩ neighbors[c], b distinct from u,v,a,c
                        common = neighbors[a] & neighbors[c]
                        for b in common:
                            if b != u and b != v and b != a and b != c:
                                return True
        return False

    def has_independent_set_of_size_k(adj, n, k):
        # Branch and bound with pruning
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Order by degree
        order = sorted(range(n), key=lambda v: len(neighbors[v]))
        
        def dfs(idx, chosen, size):
            if size == k:
                return True
            if idx == n:
                return False
            if size + (n - idx) < k:
                return False
            v = order[idx]
            # Don't include v
            if dfs(idx+1, chosen, size):
                return True
            # Include v if independent from chosen
            can = True
            for u in chosen:
                if u in neighbors[v]:
                    can = False
                    break
            if can:
                if dfs(idx+1, chosen+[v], size+1):
                    return True
            return False
        
        return dfs(0, [], 0)

    def is_valid(adj, n):
        if has_c5(adj, n):
            return False
        if has_independent_set_of_size_k(adj, n, 9):
            return False
        return True

    def turan(n, r):
        parts = [[] for _ in range(r)]
        for i in range(n):
            parts[i % r].append(i)
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if i % r != j % r:
                    adj[i][j] = adj[j][i] = True
        return adj, parts

    adj, parts = turan(32, 4)
    n = 32
    r = 4
    
    for target_n in range(33, 36):
        new_adj = [[False]*target_n for _ in range(target_n)]
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
        
        # Try different edge patterns for new vertex
        # Pattern 1: connect to all in largest part
        part_sizes = [len(parts[i]) for i in range(r)]
        largest_part = max(range(r), key=lambda i: part_sizes[i])
        
        for v in parts[largest_part]:
            new_adj[n][v] = new_adj[v][n] = True
        
        if is_valid(new_adj, target_n):
            adj = new_adj
            n = target_n
            # Recompute parts approximately
            parts = [[] for _ in range(r)]
            for i in range(n):
                parts[i % r].append(i)
            continue
        else:
            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
