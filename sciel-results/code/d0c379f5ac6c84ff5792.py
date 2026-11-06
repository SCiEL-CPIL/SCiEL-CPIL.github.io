import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    def is_valid_addition(adj, u, v):
        # adj is a list of sets or bitmasks. Let's use sets for clarity, n is small.
        # Or list of lists? Sets are faster for intersection.
        # Let's use Python sets.
        
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if len(common_uv) >= 3:
            return False
        
        # 2. Check pairs (u, w) and (v, w)
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            if v in adj[w]:
                # v is a neighbor of w, so v is in N'(w)
                # u is in N'(w)? No, we are adding u-v.
                # Wait, N'(u) = adj[u] U {v}.
                # N'(w) = adj[w] (unchanged).
                # Common neighbors of u and w in new graph:
                # (adj[u] U {v}) & adj[w]
                # = (adj[u] & adj[w]) U ({v} & adj[w])
                # Since v in adj[w], v is added to common neighbors.
                # Old common: adj[u] & adj[w]
                # New common size = len(adj[u] & adj[w]) + 1
                if len(adj[u] & adj[w]) + 1 > 2:
                    return False
            
            # Check pair (v, w)
            if u in adj[w]:
                # u is a neighbor of w.
                # N'(v) = adj[v] U {u}.
                # N'(w) = adj[w].
                # Common neighbors: (adj[v] U {u}) & adj[w]
                # = (adj[v] & adj[w]) U ({u} & adj[w])
                # Since u in adj[w], u is added.
                if len(adj[v] & adj[w]) + 1 > 2:
                    return False
            
        return True

    def greedy_add(adj):
        # Try to add edges in random order
        added = True
        while added:
            added = False
            # Shuffle pairs
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
            random.shuffle(pairs)
            for u, v in pairs:
                if is_valid_addition(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Initial best is seed
    seed_adj = [set() for _ in range(n)]
    for k in range(5):
        for a in range(4):
            for b in range(a+1, 4):
                u = 4*k + a
                v = 4*k + b
                seed_adj[u].add(v)
                seed_adj[v].add(u)
    best_edges = get_edges(seed_adj)
    best_count = len(best_edges)
    
    # Local search with restarts
    while time.time() < deadline:
        # Start with random sparse graph or empty
        # Let's try starting with random graph with p=0.1
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.1:
                    adj[i].add(j)
                    adj[j].add(i)
        
        greedy_add(adj)
        
        # Try some swaps if stuck?
        # For now, just count
        curr_edges = get_edges(adj)
        if len(curr_edges) > best_count:
            best_count = len(curr_edges)
            best_edges = curr_edges
            
        # If we hit a high count, we might want to polish
        # But greedy is fast. Let's just restart.
        
    return best_edges
