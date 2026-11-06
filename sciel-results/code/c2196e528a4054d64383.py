import time
import random

def construct():
    n = 40
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Init 8 K5s
    for k in range(8):
        base = 5 * k
        for i in range(5):
            for j in range(i + 1, 5):
                u, v = base + i, base + j
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    missing_list = [e for e in all_edges if e not in edges]
    
    def find_c6(u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Ensure nu is smaller for iteration?
        if len(nu) > len(nv):
            nu, nv = nv, nu
            u, v = v, u
            
        for a in nu:
            if a == v: continue
            for d in nv:
                if d == u or d == a: continue
                na = adj[a]
                nd = adj[d]
                
                for b in na:
                    if b in (u, v, a, d): continue
                    # Check connection b-c-d
                    # c in N(b) and N(d)
                    common = adj[b] & adj[d]
                    for c in common:
                        if c in (u, v, a, d, b): continue
                        return (u, a, b, c, d, v)
        return None

    def add_edge(u, v):
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u, v):
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)

    def remove_from_missing(edge):
        # Swap remove
        if edge in missing_set:
            idx = missing_index[edge]
            last = len(missing_list) - 1
            missing_list[idx] = missing_list[last]
            missing_index[missing_list[idx]] = idx
            missing_list.pop()
            del missing_index[edge]
            
    # Index for fast removal
    missing_set = set(missing_list)
    missing_index = {e: i for i, e in enumerate(missing_list)}
    
    deadline = time.time() + 140
    best_count = len(edges)
    
    while time.time() < deadline:
        if not missing_list:
            break
            
        # Random pick
        idx = random.randint(0, len(missing_list) - 1)
        u, v = missing_list[idx]
        
        c6 = find_c6(u, v)
        
        if c6 is None:
            add_edge(u, v)
            remove_from_missing((u, v))
            best_count = len(edges)
        else:
            # Swap
            cycle_edges = [(c6[0], c6[1]), (c6[1], c6[2]), (c6[2], c6[3]), (c6[3], c6[4]), (c6[4], c6[5])]
            rem_u, rem_v = random.choice(cycle_edges)
            
            remove_edge(rem_u, rem_v)
            add_edge(u, v)
            
            remove_from_missing((u, v))
            
            # Add removed edge back to missing
            missing_list.append((rem_u, rem_v))
            missing_index[(rem_u, rem_v)] = len(missing_list) - 1
            missing_set.add((rem_u, rem_v))

    return [list(e) for e in edges]
