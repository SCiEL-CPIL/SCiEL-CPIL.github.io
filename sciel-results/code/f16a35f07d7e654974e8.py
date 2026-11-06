import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 512
    
    # Precompute LCMs and compatibility
    adj = [0] * n
    
    for i in range(n):
        num_i = i + 1
        bits = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= n:
                bits |= (1 << j)
        adj[i] = bits
    
    best_size = 22
    best_set = list(range(1, 23))
    
    def popcount(x):
        return x.bit_count()
    
    # Greedy clique for initial lower bound
    def greedy_clique():
        P = (1 << n) - 1
        clique = []
        while P:
            best_v = -1
            best_deg = -1
            tmp = P
            while tmp:
                lsb = tmp & -tmp
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & P)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                tmp ^= lsb
            clique.append(best_v)
            P = P & adj[best_v]
        return clique
    
    g_clique = greedy_clique()
    if len(g_clique) > best_size:
        best_size = len(g_clique)
        best_set = [v + 1 for v in g_clique]
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Tomita-style coloring bound
    # Returns list of (vertex, color) sorted by color descending, then by vertex
    def color_bound(P):
        # Get vertices in P
        verts = []
        tmp = P
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            verts.append(v)
            tmp ^= lsb
        
        if not verts:
            return []
        
        # Sort by degree in P descending for better coloring
        verts.sort(key=lambda x: popcount(adj[x] & P), reverse=True)
        
        # Greedy coloring
        col = {}
        for v in verts:
            used = set()
            for u in verts:
                if u == v:
                    continue
                if u in col and (adj[v] >> u) & 1:
                    used.add(col[u])
            c = 1
            while c in used:
                c += 1
            col[v] = c
        
        # Build list sorted by color descending
        colored = [(v, col[v]) for v in verts]
        colored.sort(key=lambda x: (-x[1], -popcount(adj[x[0]] & P)))
        
        return colored
    
    def search(P, C, clique):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if C > best_size:
                best_size = C
                best_set = clique[:]
            return
        
        # Get colored vertices
        colored = color_bound(P)
        
        if not colored:
            return
        
        # Max color gives upper bound on additional vertices
        max_color = colored[0][1]
        if C + max_color <= best_size:
            return
        
        # Process vertices in reverse color order (highest color first)
        # This is the key to Tomita's algorithm
        for v, c in colored:
            if time.time() > deadline:
                return
            
            # Prune: if C + c <= best_size, we can't improve
            if C + c <= best_size:
                break
            
            new_P = P & adj[v]
            search(new_P, C + 1, clique + [v])
            
            # Remove v from P
            P = P & ~(1 << v)
            
            # Update bound: remaining vertices have color <= c
            # Actually, we need to recompute or track, but for now just check simple bound
            if C + popcount(P) <= best_size:
                break
    
    initial_P = (1 << n) - 1
    search(initial_P, 0, [])
    
    return best_set
