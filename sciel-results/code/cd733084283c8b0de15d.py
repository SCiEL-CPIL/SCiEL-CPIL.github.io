import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40

    def time_ok():
        return time.time() < deadline

    # Bitset adjacency
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def can_add(adj, u, v):
        # Check if adding (u,v) creates a C6
        # C6: u-a-b-c-d-v
        # a in N(u)\{v}, d in N(v)\{u}, a!=d
        # Path a-b-c-d length 3 in G\{u,v}
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate a in N(u)
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                B = Na & ~(1 << d)
                if B == 0: continue
                
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0: continue
                
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    
                    if adj[b] & C & ~(1 << b):
                        return False
        return True

    def greedy_add(adj, edge_set, edges, candidates):
        added = True
        while added and time_ok():
            added = False
            remaining = []
            for u, v in candidates:
                if (min(u,v), max(u,v)) in edge_set: continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added = True
                else:
                    remaining.append((u, v))
            candidates = remaining
        return edges

    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for trial in range(50):
        if not time_ok(): break
        random.seed(trial)
        
        # Init: Random sparse graph
        edges = []
        adj = [0]*n
        edge_set = set()
        
        # Add random edges
        random.shuffle(all_edges)
        for u, v in all_edges:
            if len(edges) > 80: break # Keep sparse to avoid C6
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                edges.append([u, v])
        
        # Greedy fill
        remaining = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        greedy_add(adj, edge_set, edges, remaining)
        
        # Local Search: Try to add, if fail, try swap
        improved = True
        while improved and time_ok():
            improved = False
            # Try add
            for u, v in all_edges:
                if (u, v) in edge_set: continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    improved = True
                    break
            
            if improved: continue
            
            # Try swap: remove (a,b), add (u,v)
            # This is expensive. Just try removing random edge and re-greedy?
            # Let's try a simple perturbation
            if edges:
                rem_idx = random.randint(0, len(edges)-1)
                ra, rb = edges.pop(rem_idx)
                adj[ra] &= ~(1 << rb)
                adj[rb] &= ~(1 << ra)
                edge_set.discard((min(ra,rb), max(ra,rb)))
                
                # Re-greedy
                remaining = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
                greedy_add(adj, edge_set, edges, remaining)
                improved = True
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return best_edges
