import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute pairs
    all_4_subsets = list(combinations(vertices, 4))
    pairs = []
    for A in all_4_subsets:
        mask_A = 0
        for v in A:
            mask_A |= (1 << v)
        remaining = [v for v in vertices if v not in A]
        for B in combinations(remaining, 4):
            mask_B = 0
            for v in B:
                mask_B |= (1 << v)
            pairs.append((mask_A, mask_B, A, B))
            
    def is_q3_free(adj):
        for mask_A, mask_B, A, B in pairs:
            # Check A degrees
            fail = False
            for v in A:
                if (adj[v] & mask_B).bit_count() != 3:
                    fail = True
                    break
            if fail: continue
            # Check B degrees
            for v in B:
                if (adj[v] & mask_A).bit_count() != 3:
                    fail = True
                    break
            if not fail:
                return False
        return True

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    best_adj = None
    best_count = -1
    
    initial_adj = [(1 << n) - 2 for _ in range(n)]
    
    def local_search(adj):
        adj = adj[:]
        while True:
            if time.time() > deadline:
                return adj
            bad_found = False
            for mask_A, mask_B, A, B in pairs:
                # Check if this pair is a Q3
                ok = True
                for v in A:
                    if (adj[v] & mask_B).bit_count() != 3:
                        ok = False
                        break
                if not ok: continue
                for v in B:
                    if (adj[v] & mask_A).bit_count() != 3:
                        ok = False
                        break
                if ok:
                    # Find edges to remove
                    edges_to_remove = []
                    for u in A:
                        for v in B:
                            if adj[u] & (1 << v):
                                edges_to_remove.append((u, v))
                    if edges_to_remove:
                        u, v = random.choice(edges_to_remove)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        bad_found = True
                    break
            if not bad_found:
                break
        return adj

    def polish(adj):
        # Try to add edges back
        adj = adj[:]
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    # Try adding
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_q3_free(adj):
                        # Keep
                        pass
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        return adj

    # Main loop
    while time.time() < deadline - 10: # Reserve 10s for final check/polish
        adj = initial_adj[:]
        adj = local_search(adj)
        
        # Polish
        if time.time() < deadline - 5:
            adj = polish(adj)
            
        if is_q3_free(adj):
            c = count_edges(adj)
            if c > best_count:
                best_count = c
                best_adj = adj[:]
                
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    return get_edges(best_adj)
