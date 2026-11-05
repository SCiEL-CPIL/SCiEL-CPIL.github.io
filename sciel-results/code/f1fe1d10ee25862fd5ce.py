import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 40
    # Initial graph: 8 disjoint K5
    # Vertices 0-4 in K5_1, 5-9 in K5_2, etc.
    edges = set()
    for k in range(8):
        base = k * 5
        for a in range(5):
            for b in range(a + 1, 5):
                edges.add((base + a, base + b))
    
    best_edges = list(edges)
    
    # Precompute all possible edges
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                all_possible.append((u, v))
    
    # Helper to check if adding (u,v) creates C6
    # We need to check if dist(u,v) == 5 in the current graph
    # Since graph is small, we can do BFS
    
    def has_path_len_5(u, v, current_edges):
        # BFS from u, limit depth 5
        # If v is reached at depth exactly 5, return True
        # Note: If v is reached at depth < 5, adding edge creates C3-C5, which is OK.
        # We only care about C6.
        # However, if there is a path of length 5, adding edge creates C6.
        # We must ensure the path is simple. BFS finds shortest path.
        # If shortest path is 5, then there is a path of length 5.
        # If shortest path is < 5, there might still be a path of length 5?
        # If dist(u,v) = 2, there is a path u-x-v.
        # Is there a path of length 5?
        # u-x-y-z-w-v?
        # If such a path exists, adding (u,v) creates a C6?
        # The cycle would be u-v-w-z-y-x-u? No, the edge is (u,v).
        # The cycle is u-v-...-u.
        # If we add (u,v), any path from v to u of length 5 creates a C6.
        # So we need to check if there is ANY path of length 5 from u to v.
        # BFS gives shortest. If shortest is 5, yes.
        # If shortest is 3, is there a path of length 5?
        # Yes, if the graph is dense enough.
        # So BFS depth 5 is necessary.
        
        from collections import deque
        q = deque()
        q.append((u, 0, {u}))
        found = False
        while q:
            node, depth, visited = q.popleft()
            if depth == 5:
                if node == v:
                    return True
                continue
            if depth >= 5:
                continue
            for neighbor in current_edges.get(node, []):
                if neighbor not in visited:
                    new_visited = visited | {neighbor}
                    q.append((neighbor, depth + 1, new_visited))
        return False

    # To speed up, maintain adjacency list
    def build_adj(edges_set):
        adj = {i: set() for i in range(n)}
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    best_count = len(best_edges)
    
    while time.time() < deadline:
        # Random restart
        current_edges = set(best_edges) # Start from best found so far? Or initial?
        # Starting from initial is safer to avoid bias, but starting from best might improve.
        # Let's start from initial K5s for diversity, or mix.
        # Actually, starting from best found and trying to improve is good.
        # But we need to allow "moves" that reduce edge count? No, greedy only adds.
        # So we should restart from initial or a random subset.
        
        # Let's restart from initial K5s
        current_edges = set()
        for k in range(8):
            base = k * 5
            for a in range(5):
                for b in range(a + 1, 5):
                    current_edges.add((base + a, base + b))
        
        candidates = list(all_possible)
        random.shuffle(candidates)
        
        adj = build_adj(current_edges)
        
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            # Check if adding creates C6
            # We need to check if there is a path of length 5 from u to v
            # Using adj
            # BFS
            from collections import deque
            q = deque()
            q.append((u, 0, {u}))
            creates_c6 = False
            while q:
                node, depth, visited = q.popleft()
                if depth == 5:
                    if node == v:
                        creates_c6 = True
                        break
                    continue
                if depth >= 5:
                    continue
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        q.append((neighbor, depth + 1, visited | {neighbor}))
            
            if not creates_c6:
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
    
    return [list(e) for e in best_edges]
