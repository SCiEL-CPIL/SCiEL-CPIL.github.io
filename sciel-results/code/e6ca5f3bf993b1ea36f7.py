import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    adj = [0] * n
    edges = [] # List of (u, v) with u < v

    def get_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def add_edge(u, v):
        if u > v: u, v = v, u
        if adj[u] & (1 << v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append((u, v))
        return True

    def remove_edge(u, v):
        if u > v: u, v = v, u
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Remove from edges list
        for i, e in enumerate(edges):
            if e[0] == u and e[1] == v:
                edges.pop(i)
                break

    def count_edges():
        return len(edges)

    def can_add(u, v):
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        
        n_u = adj[u]
        n_v = adj[v]
        
        n2_u = 0
        for a in get_bits(n_u):
            n2_u |= adj[a]
        n2_u &= ~n_u
        n2_u &= ~(1 << u)
        
        n2_v = 0
        for d in get_bits(n_v):
            n2_v |= adj[d]
        n2_v &= ~n_v
        n2_v &= ~(1 << v)
        
        # Check existing edges
        for b, c in edges:
            # Case 1: b in N2(u), c in N2(v)
            if (n2_u >> b) & 1 and (n2_v >> c) & 1:
                # Check disjoint paths
                cand_a = n_u & adj[b]
                cand_d = n_v & adj[c]
                if cand_a and cand_d:
                    # Need a != c, d != b, a != d
                    # a from cand_a, d from cand_d
                    # If we can find such pair, C6 exists
                    for a in get_bits(cand_a):
                        if a == c: continue
                        for d in get_bits(cand_d):
                            if d == b or d == a: continue
                            return False
            # Case 2: c in N2(u), b in N2(v)
            if (n2_u >> c) & 1 and (n2_v >> b) & 1:
                cand_a = n_u & adj[c]
                cand_d = n_v & adj[b]
                if cand_a and cand_d:
                    for a in get_bits(cand_a):
                        if a == b: continue
                        for d in get_bits(cand_d):
                            if d == c or d == a: continue
                            return False
                            
        return True

    def greedy_fill():
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add(u, v):
                            add_edge(u, v)
                            changed = True

    # Initialize with 8 K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    greedy_fill()
    current_edges = count_edges()
    best_edges = current_edges
    best_adj = adj[:]
    best_edges_list = edges[:]
    
    while time.time() < deadline:
        added = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if can_add(u, v):
                        add_edge(u, v)
                        added = True
                        break
            if added:
                break
        
        if added:
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                best_edges_list = edges[:]
            continue
        
        # Kick
        if not edges:
            break
        u, v = random.choice(edges)
        
        old_adj = adj[:]
        old_edges = edges[:]
        
        remove_edge(u, v)
        greedy_fill()
        
        new_edges = count_edges()
        
        if new_edges > current_edges:
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                best_edges_list = edges[:]
        else:
            adj = old_adj[:]
            edges = old_edges[:]
            
    return [[u, v] for u, v in best_edges_list]
