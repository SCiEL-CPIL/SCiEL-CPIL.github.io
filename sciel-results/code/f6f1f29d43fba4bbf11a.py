import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 16
    
    # Build Groetzsch graph
    edges_set = set()
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        edges_set.add((u, v))
    
    for i in range(5):
        add_edge(i, (i + 1) % 5)
    for i in range(5):
        u, w = i, (i + 1) % 5
        add_edge(u, 5 + w)
        add_edge(w, 5 + u)
    for u in range(5):
        add_edge(5 + u, 10)
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def build_adj(es):
        adj = [0] * n
        for (u, v) in es:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def check_3color(adj):
        color = [-1] * n
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def bt(idx):
            if idx == n:
                return True
            v = order[idx]
            nb = adj[v]
            for c in range(3):
                ok = True
                m = nb
                while m:
                    lsb = m & (-m)
                    ni = lsb.bit_length() - 1
                    if color[ni] == c:
                        ok = False
                        break
                    m ^= lsb
                if ok:
                    color[v] = c
                    if bt(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return bt(0)
    
    current_edges = set(edges_set)
    current_adj = build_adj(current_edges)
    assert not check_3color(current_adj)
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    random.shuffle(all_edges)
    
    T = 3.0
    T_min = 0.01
    cooling = 0.9998
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.55:
            # Add edge
            cands = [(u,v) for (u,v) in all_edges if (u,v) not in current_edges and (current_adj[u] & current_adj[v]) == 0]
            if not cands: continue
            u, v = random.choice(cands)
            current_edges.add((u,v))
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            if check_3color(current_adj):
                current_edges.discard((u,v))
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        elif r < 0.85:
            # Remove edge
            if not current_edges: continue
            u, v = random.choice(list(current_edges))
            current_edges.discard((u,v))
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            if check_3color(current_adj):
                # Became 3-colorable, revert
                current_edges.add((u,v))
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                if random.random() > T:
                    current_edges.add((u,v))
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        else:
            # Swap
            if not current_edges: continue
            u1, v1 = random.choice(list(current_edges))
            current_edges.discard((u1,v1))
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            cands = [(u,v) for (u,v) in all_edges if (u,v) not in current_edges and (current_adj[u] & current_adj[v]) == 0]
            if not cands:
                current_edges.add((u1,v1))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                continue
            u2, v2 = random.choice(cands)
            current_edges.add((u2,v2))
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            if check_3color(current_adj):
                current_edges.discard((u2,v2))
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_edges.add((u1,v1))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
            else:
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
        
        if T > T_min:
            T *= cooling
    
    return [list(e) for e in best_edges]
