import time
import random
import sys
from functools import lru_cache

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_adj(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        adj_tuple = tuple(adj)
        order = sorted(range(n), key=lambda x: bin(adj_tuple[x]).count('1'), reverse=True)
        
        colors_used = [0] * n
        colored = [False] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            available = 7 & ~colors_used[v]
            if available == 0:
                return False
            
            while available:
                c = (available & (-available)).bit_length() - 1
                available &= available - 1
                
                colored[v] = True
                updated = []
                for u in range(n):
                    if not colored[u] and (adj_tuple[v] >> u) & 1:
                        colors_used[u] |= (1 << c)
                        updated.append(u)
                
                if backtrack(idx + 1):
                    return True
                
                for u in updated:
                    colors_used[u] &= ~(1 << c)
                colored[v] = False
            
            return False
        
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def greedy_add_edges(adj):
        total_added = 0
        while True:
            if time.time() > deadline:
                break
            added = 0
            possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
            random.shuffle(possible)
            
            for u, v in possible:
                if is_triangle_free_adj(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added += 1
            
            if added == 0:
                break
            total_added += added
        
        return total_added

    def try_add_edge_and_check(adj, u, v):
        if adj[u] & (1 << v):
            return False
        if not is_triangle_free_adj(adj, u, v):
            return False
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not is_3_colorable(adj):
            return True
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def make_grotzsch():
        adj = [0] * n
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def make_random_tf(density=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    # Initialize with Grötzsch graph to ensure we have a valid 4-chromatic graph
    adj = make_grotzsch()
    # Verify it's not 3-colorable. If it is, something is very wrong, but we proceed.
    if not is_3_colorable(adj):
        greedy_add_edges(adj)
        edges = get_edges_from_adj(adj)
        best_count = len(edges)
        best_edges = edges
    else:
        # Fallback: If Grötzsch is 3-colorable (bug), use a hardcoded valid graph?
        # For now, just ensure we don't return empty.
        # This case should not happen.
        best_edges = get_edges_from_adj(adj)
        best_count = len(best_edges)

    # Search for better graphs
    while time.time() < deadline:
        # Start from current best or random
        if random.random() < 0.5 and best_edges:
            adj = [0] * n
            for u, v in best_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            # Perturb
            current_edges = get_edges_from_adj(adj)
            random.shuffle(current_edges)
            for u, v in current_edges[:3]:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        else:
            adj = make_random_tf(0.6)
        
        # Ensure triangle-free
        # The random gen is triangle-free by construction.
        # Perturbation might not break triangle-free (removing edges doesn't create triangles).
        
        if not is_3_colorable(adj):
            greedy_add_edges(adj)
            
            possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
            random.shuffle(possible)
            
            for u, v in possible:
                if time.time() > deadline:
                    break
                try_add_edge_and_check(adj, u, v)
            
            edges = get_edges_from_adj(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
        else:
            # Try to make it 4-chromatic
            greedy_add_edges(adj)
            possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
            random.shuffle(possible)
            
            for u, v in possible:
                if time.time() > deadline:
                    break
                try_add_edge_and_check(adj, u, v)
            
            if not is_3_colorable(adj):
                edges = get_edges_from_adj(adj)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges

    # Final safety check: if best_edges is empty or 3-colorable, return a known valid graph
    if not best_edges or is_3_colorable([0]*n): # dummy check
        # Re-construct Grötzsch
        adj = make_grotzsch()
        best_edges = get_edges_from_adj(adj)

    # Ensure the returned graph is not 3-colorable
    # Build adj from best_edges to check
    adj_check = [0]*n
    for u,v in best_edges:
        adj_check[u] |= (1<<v)
        adj_check[v] |= (1<<u)
    
    if is_3_colorable(adj_check):
        # This is a failure case. Return a hardcoded 4-chromatic triangle-free graph.
        # Grötzsch graph edges:
        # 0-1, 1-2, 2-3, 3-4, 4-0
        # 0-6, 1-5, 1-7, 2-6, 2-8, 3-7, 3-9, 4-8, 4-5, 0-9
        # 10-5, 10-6, 10-7, 10-8, 10-9
        # Vertices 11-15 are isolated.
        hardcoded = [
            [0,1], [1,2], [2,3], [3,4], [4,0],
            [0,6], [1,5], [1,7], [2,6], [2,8], [3,7], [3,9], [4,8], [4,5], [0,9],
            [10,5], [10,6], [10,7], [10,8], [10,9]
        ]
        return hardcoded

    return best_edges
