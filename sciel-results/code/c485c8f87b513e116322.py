import time
import random
import numpy as np

def construct():
    n = 40
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    # Use numpy for fast adjacency operations
    adj = np.zeros((n, n), dtype=np.int8)
    
    def edge_count():
        return int(np.sum(adj) // 2)
    
    def would_create_c6(u, v):
        """Check if adding edge (u,v) would create a C6.
        A C6 is formed if there's a path of length 5 from u to v in current graph.
        Path: u - a - b - c - d - v (5 edges)
        We can check: for each neighbor a of u, for each neighbor d of v,
        check if there's a path of length 3 from a to d (a-x-y-d) where
        x,y are not u,v and the path doesn't use u or v.
        """
        # Get neighbors of u and v
        neigh_u = np.where(adj[u] == 1)[0]
        neigh_v = np.where(adj[v] == 1)[0]
        
        # For each a in neigh_u, d in neigh_v, check path a-x-y-d of length 3
        # This is equivalent to: (A^2)[a][d] > 0 where A is adjacency with u,v removed
        # But we need to be more careful - the path must not go through u or v
        
        for a in neigh_u:
            if a == v:
                continue
            for d in neigh_v:
                if d == u or d == a:
                    continue
                # Check if there's a path of length 2 from a to d (a-x-d)
                # where x is not u, v, a, d
                # i.e., sum over x of adj[a][x] * adj[x][d] > 0, x not in {u,v,a,d}
                common = np.where((adj[a] & adj[:, d]) > 0)[0]
                for x in common:
                    if x != u and x != v and x != a and x != d:
                        return True
        return False
    
    def try_add_edge(u, v):
        if u == v or adj[u][v]:
            return False
        if would_create_c6(u, v):
            return False
        adj[u][v] = adj[v][u] = 1
        return True
    
    def remove_edge(u, v):
        adj[u][v] = adj[v][u] = 0
    
    def is_c6_free_check():
        """Full validation: check no C6 exists."""
        # For each pair of non-adjacent vertices u,v, check no path of length 5
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                # Check path u-a-b-c-d-v of length 5
                neigh_u = np.where(adj[u] == 1)[0]
                for a in neigh_u:
                    if a == v:
                        continue
                    neigh_a = np.where(adj[a] == 1)[0]
                    for b in neigh_a:
                        if b == u or b == v or b == a:
                            continue
                        neigh_b = np.where(adj[b] == 1)[0]
                        for c in neigh_b:
                            if c == u or c == v or c == a or c == b:
                                continue
                            neigh_c = np.where(adj[c] == 1)[0]
                            for d in neigh_c:
                                if d == u or d == v or d == a or d == b or d == c:
                                    continue
                                if adj[d][v]:
                                    return False
        return True
    
    # Phase 1: Greedy construction
    # Start with random edges, then greedily add more
    random.seed(42)
    
    # Initial: try to build a dense graph greedily
    # Use a random order for vertices to break symmetry
    order = list(range(n))
    random.shuffle(order)
    
    # Try adding edges in a smart order: prefer high-degree vertices
    for u in order:
        for v in order:
            if u >= v:
                continue
            try_add_edge(u, v)
    
    best_count = edge_count()
    best_adj = adj.copy()
    
    # Phase 2: Local search - try to add edges
    while time.time() < deadline - 10.0:
        progress = False
        # Try all non-edges
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u][v]:
                    if try_add_edge(u, v):
                        progress = True
        if not progress:
            break
    
    best_count = max(best_count, edge_count())
    if edge_count() > best_count:
        best_count = edge_count()
        best_adj = adj.copy()
    
    # Phase 3: Local search with kicks (edge removal + re-addition)
    while time.time() < deadline:
        # Try random non-edges
        improved = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not adj[u][v]:
                if try_add_edge(u, v):
                    improved = True
                    if edge_count() > best_count:
                        best_count = edge_count()
                        best_adj = adj.copy()
        
        if not improved:
            # Kick: remove a random edge and try to re-add different edges
            if edge_count() > 0:
                # Pick a random edge to remove
                edges_list = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u][v]:
                            edges_list.append((u, v))
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Now try to add edges greedily
                    for u2 in range(n):
                        for v2 in range(u2 + 1, n):
                            if not adj[u2][v2]:
                                try_add_edge(u2, v2)
                    if edge_count() > best_count:
                        best_count = edge_count()
                        best_adj = adj.copy()
    
    # Final validation
    adj = best_adj
    if not is_c6_free_check():
        # Fallback: rebuild from scratch with greedy only
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in range(u + 1, n):
                try_add_edge(u, v)
    
    # Convert to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                result.append([u, v])
    
    return result
