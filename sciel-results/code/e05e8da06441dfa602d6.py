import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 512
    
    # Precompute compatibility bitsets
    compat = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for y in range(1, N + 1):
            g = gcd(x, y)
            lcm_val = x // g * y
            if lcm_val <= N:
                mask |= (1 << (y - 1))
        compat[x] = mask
        
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    best_clique = []
    best_size = 0
    
    all_mask = (1 << N) - 1
    
    # Quick greedy to get a good initial lower bound
    def greedy_clique():
        R = []
        P = all_mask
        while P:
            # Pick vertex with max degree in P
            best_v = -1
            best_deg = -1
            p = P
            idx = 0
            while p:
                if p & 1:
                    v = idx + 1
                    deg = popcount(compat[v] & P)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                p >>= 1
                idx += 1
            
            if best_v == -1:
                break
                
            R.append(best_v)
            P = P & compat[best_v]
            P = P & ~(1 << (best_v - 1))
            
        return R

    init_clique = greedy_clique()
    if len(init_clique) > best_size:
        best_size = len(init_clique)
        best_clique = init_clique
        
    # Tomita's algorithm
    def tomita(P, R):
        nonlocal best_clique, best_size
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        if popcount(P) + len(R) <= best_size:
            return
            
        # Coloring phase
        vertices_in_P = []
        p = P
        idx = 0
        while p:
            if p & 1:
                vertices_in_P.append(idx + 1)
            p >>= 1
            idx += 1
            
        if not vertices_in_P:
            return

        # Sort by degree in P descending for better coloring
        def degree_in_P(v):
            return popcount(compat[v] & P)
        
        vertices_in_P.sort(key=degree_in_P, reverse=True)
        
        colors = {}
        max_color = 0
        for v in vertices_in_P:
            forbidden = set()
            neighbors = compat[v] & P
            n = neighbors
            idx = 0
            while n:
                if n & 1:
                    u = idx + 1
                    if u in colors:
                        forbidden.add(colors[u])
                n >>= 1
                idx += 1
            
            c = 1
            while c in forbidden:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        if max_color + len(R) <= best_size:
            return
            
        # Branching
        # Sort by color descending
        sorted_vertices = sorted(colors.items(), key=lambda item: item[1], reverse=True)
        
        for c, v in sorted_vertices:
            if time.time() > deadline:
                return
                
            # Check if v is still in P
            if not (P & (1 << (v - 1))):
                continue
                
            new_R = R + [v]
            new_P = (P & compat[v]) & ~(1 << (v - 1))
            
            tomita(new_P, new_R)
            
            P = P & ~(1 << (v - 1))
            
            if popcount(P) + len(R) <= best_size:
                break

    tomita(all_mask, [])
    
    if not best_clique:
        return [1]
        
    return best_clique
