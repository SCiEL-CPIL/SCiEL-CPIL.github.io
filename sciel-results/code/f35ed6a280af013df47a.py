import time
import random

def construct():
    start_time = time.time()
    n = 16
    vertices = list(range(n))
    
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
            
    def creates_q3_fast(u, v, adj):
        mask_u = adj[u]
        mask_v = adj[v]
        
        # If degree < 2, cannot form Q3
        if bin(mask_u).count('1') < 2 or bin(mask_v).count('1') < 2:
            return False
            
        neighbors_u = [i for i in range(n) if (mask_u >> i) & 1]
        neighbors_v = [i for i in range(n) if (mask_v >> i) & 1]
        
        for a in neighbors_u:
            for b in neighbors_u:
                if a == b: continue
                for c in neighbors_v:
                    if c == a or c == b: continue
                    for d in neighbors_v:
                        if d == a or d == b or d == c: continue
                        
                        # Pairing 1: a-c, b-d
                        if ((adj[a] >> c) & 1) and ((adj[b] >> d) & 1):
                            mask_e = adj[a] & adj[c] & ~(1 << u) & ~(1 << v)
                            mask_f = adj[b] & adj[d] & ~(1 << u) & ~(1 << v)
                            # e, f must not be a,b,c,d
                            mask_e &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                            mask_f &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                            
                            # Check if there exists e in mask_e, f in mask_f such that (e,f) is edge
                            # e_list and f_list are small
                            e_list = [i for i in range(n) if (mask_e >> i) & 1]
                            f_list = [i for i in range(n) if (mask_f >> i) & 1]
                            
                            for e in e_list:
                                for f in f_list:
                                    if e == f: continue
                                    if (adj[e] >> f) & 1:
                                        return True
                                        
                        # Pairing 2: a-d, b-c
                        if ((adj[a] >> d) & 1) and ((adj[b] >> c) & 1):
                            mask_e = adj[a] & adj[d] & ~(1 << u) & ~(1 << v)
                            mask_f = adj[b] & adj[c] & ~(1 << u) & ~(1 << v)
                            mask_e &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                            mask_f &= ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                            
                            e_list = [i for i in range(n) if (mask_e >> i) & 1]
                            f_list = [i for i in range(n) if (mask_f >> i) & 1]
                            
                            for e in e_list:
                                for f in f_list:
                                    if e == f: continue
                                    if (adj[e] >> f) & 1:
                                        return True
        return False

    best_edges = set()
    best_count = 0
    
    # Greedy + Local Search
    restarts = 50
    for _ in range(restarts):
        if time.time() - start_time > 115:
            break
            
        adj = [0] * n
        current_edges = set()
        
        edges = all_edges[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if not creates_q3_fast(u, v, adj):
                current_edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Local Search: Add more if possible
        improved = True
        while improved and (time.time() - start_time < 115):
            improved = False
            non_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if (time.time() - start_time > 115):
                    break
                if not creates_q3_fast(u, v, adj):
                    current_edges.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    return [list(e) for e in best_edges]
