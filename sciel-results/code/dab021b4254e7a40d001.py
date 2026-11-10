import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check pair (u,v): common neighbors <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # For each w in N(v)\{u}: check (u,w) pair
        mask = adj[v] & ~(1 << u)
        while mask:
            low = mask & (-mask)
            w = low.bit_length() - 1
            mask ^= low
            if popcount(adj[u] & adj[w]) > 1:
                return False
        # For each w in N(u)\{v}: check (v,w) pair
        mask = adj[u] & ~(1 << v)
        while mask:
            low = mask & (-mask)
            w = low.bit_length() - 1
            mask ^= low
            if popcount(adj[v] & adj[w]) > 1:
                return False
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2
    
    best_adj = [0] * n
    best_count = 0
    
    random.seed(12345)
    
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for attempt in range(500):
        if time.time() > deadline - 15.0:
            break
        
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline - 10.0:
                break
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Local search
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try adding edges
            missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
            
            # Try 1-for-2 swaps
            if not improved:
                existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
                random.shuffle(existing)
                for u, v in existing[:30]:
                    if time.time() > deadline - 5.0:
                        break
                    remove_edge(adj, u, v)
                    added = 0
                    candidates = [(a, b) for a, b in all_pairs 
                                  if not ((adj[a] >> b) & 1) and not (a==u and b==v)]
                    random.shuffle(candidates)
                    for a, b in candidates:
                        if added >= 2:
                            break
                        if can_add(adj, a, b):
                            add_edge(adj, a, b)
                            added += 1
                    if added >= 1:
                        improved = True
                    else:
                        add_edge(adj, u, v)
                    if improved:
                        break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
