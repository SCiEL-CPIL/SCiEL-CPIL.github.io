import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40

    def time_ok():
        return time.time() < deadline

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        C6: u-a-b-c-d-v, all distinct.
        Equivalent: exists a in N(u)\{v}, d in N(v)\{u}, a!=d,
        and a path a-b-c-d of length 3 in G-{u,v}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a
            
            Na_base = adj[a] & ~(1 << u) & ~(1 << v)
            
            mask_d = Nv & ~(1 << a)
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d
                
                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue
                
                # Check if any b in B has a neighbor in C (excluding b itself)
                # For each b in B, check N(b) & C & ~(1<<b)
                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    if adj[b] & C & ~(1 << b):
                        return True
        
        return False

    def greedy_add(adj, edge_set, edges, candidate_edges):
        added_any = True
        while added_any and time_ok():
            added_any = False
            remaining = []
            for item in candidate_edges:
                if len(item) == 2:
                    u, v = item
                else:
                    u, v = item[1], item[2]
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if would_create_c6(adj, u, v):
                    remaining.append(item)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((min(u,v), max(u,v)))
                    edges.append([u, v])
                    added_any = True
            candidate_edges = remaining
        return edges

    def local_search_add(adj, edge_set, edges):
        """Try to add edges one by one."""
        improved = True
        while improved and time_ok():
            improved = False
            for u in range(n):
                if not time_ok():
                    break
                for v in range(u + 1, n):
                    if not time_ok():
                        break
                    if adj[u] & (1 << v):
                        continue
                    if not would_create_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add((u, v))
                        edges.append([u, v])
                        improved = True
        return edges

    def local_search_swap(adj, edge_set, edges):
        """Try edge swaps: remove (a,b), add (c,d) if it doesn't create C6 and improves density.
        We do this by: for each edge (a,b), try removing it, then try adding all possible new edges."""
        improved = True
        while improved and time_ok():
            improved = False
            edges_copy = edges[:]
            for i in range(len(edges_copy)):
                if not time_ok():
                    break
                a, b = edges_copy[i]
                # Remove edge (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                edge_set.discard((min(a,b), max(a,b)))
                edges.remove([a, b])
                
                # Try to add any new edge
                best_new = None
                for u in range(n):
                    if not time_ok():
                        break
                    for v in range(u + 1, n):
                        if not time_ok():
                            break
                        if (u, v) in edge_set:
                            continue
                        if (u, v) == (min(a,b), max(a,b)):
                            continue
                        if not would_create_c6(adj, u, v):
                            best_new = (u, v)
                            break
                    if best_new:
                        break
                
                if best_new:
                    u, v = best_new
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append([u, v])
                    improved = True
                else:
                    # Restore the original edge
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    edge_set.add((min(a,b), max(a,b)))
                    edges.append([a, b])
        return edges

    # Strategy: Start with K5 blocks, then greedy add, then local search
    best_edges = []
    best_count = 0

    # Initial: 8 disjoint K5s
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    adj = make_adj(edges)
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    # Greedy add all possible edges
    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    random.seed(42)
    random.shuffle(all_possible)
    
    greedy_add(adj, edge_set, edges, all_possible)
    
    best_count = len(edges)
    best_edges = edges[:]
    
    # Local search: add edges
    if time_ok():
        local_search_add(adj, edge_set, edges)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Local search: swaps
    if time_ok():
        local_search_swap(adj, edge_set, edges)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Try more random restarts if time permits
    for trial in range(10):
        if not time_ok():
            break
        random.seed(1000 + trial)
        
        # Random 3-regular start
        adj_t = [0] * n
        edges_t = []
        edge_set_t = set()
        
        # Build a random sparse graph
        for i in range(n):
            candidates = [j for j in range(n) if j != i and (min(i,j), max(i,j)) not in edge_set_t]
            random.shuffle(candidates)
            for j in candidates[:3]:
                if (min(i,j), max(i,j)) not in edge_set_t:
                    adj_t[i] |= (1 << j)
                    adj_t[j] |= (1 << i)
                    edge_set_t.add((min(i,j), max(i,j)))
                    edges_t.append([i, j])
        
        all_possible_t = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set_t]
        random.shuffle(all_possible_t)
        
        greedy_add(adj_t, edge_set_t, edges_t, all_possible_t)
        
        if len(edges_t) > best_count:
            best_count = len(edges_t)
            best_edges = edges_t[:]
        
        # Local search on this solution
        if time_ok():
            local_search_add(adj_t, edge_set_t, edges_t)
            if len(edges_t) > best_count:
                best_count = len(edges_t)
                best_edges = edges_t[:]
        
        if time_ok():
            local_search_swap(adj_t, edge_set_t, edges_t)
            if len(edges_t) > best_count:
                best_count = len(edges_t)
                best_edges = edges_t[:]
    
    return best_edges
