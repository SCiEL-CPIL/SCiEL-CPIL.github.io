import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n_base = 32
    adj = [0] * n_base
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = base + i, base + j
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_base.append([u, v])
    
    def has_c5_with_new_vertex(adj, new_vertex, new_neighbors):
        """Check if adding new_vertex with given neighbors creates a C5 in the base graph"""
        n = len(adj)
        nb_set = set(new_neighbors)
        # A C5 through new_vertex: new_vertex - a - b - c - d - new_vertex
        # where a,d in new_neighbors, and a-b-c-d is a path of length 3 in base
        # So we need: a,d in nb_set, a~b, b~c, c~d in base, a!=c, b!=d, etc.
        for a in new_neighbors:
            for d in new_neighbors:
                if a == d:
                    continue
                # Find common neighbors of a and d at distance 2 via b,c
                # a-b-c-d: b is neighbor of a, c is neighbor of d, b~c
                for b in range(n):
                    if b == a or b == d or b in nb_set:
                        continue
                    if not (adj[a] >> b) & 1:
                        continue
                    # b is neighbor of a, now find c neighbor of d and b
                    common_bd = adj[b] & adj[d]
                    if common_bd:
                        # c is in common_bd, c != a, c != b, c != d
                        c_bits = common_bd & ~(1 << a) & ~(1 << b) & ~(1 << d)
                        if c_bits:
                            return True
        return False
    
    def non_neighbors_alpha(adj, candidates, target):
        """Check if induced subgraph on candidates has alpha >= target"""
        if len(candidates) < target:
            return False
        if not candidates:
            return False
        
        def bt(cands, size):
            if size >= target:
                return True
            if not cands:
                return False
            if size + len(cands) < target:
                return False
            v = cands[0]
            rest = cands[1:]
            if bt([x for x in rest if not (adj[v] >> x) & 1], size + 1):
                return True
            if bt(rest, size):
                return True
            return False
        
        return bt(candidates, 0)
    
    best_n = n_base
    best_edges = edges_base[:]
    
    found = False
    from itertools import combinations
    
    # Systematic search over neighbor sets for vertex 32
    for k in range(4, 13):
        if time.time() > deadline:
            break
        if found:
            break
        
        if k <= 6:
            subsets = list(combinations(range(32), k))
            random.shuffle(subsets)
        else:
            subsets = [tuple(random.sample(range(32), k)) for _ in range(10000)]
        
        for subset in subsets:
            if time.time() > deadline:
                break
            if found:
                break
            
            neighbor_set = set(subset)
            
            # Check C5: only need to check C5s involving vertex 32
            if has_c5_with_new_vertex(adj, 32, list(subset)):
                continue
            
            # Check alpha: non-neighbors must not have independent set of size 8
            non_neighbors = [v for v in range(32) if v not in neighbor_set]
            if non_neighbors_alpha(adj, non_neighbors, 8):
                continue
            
            # Valid! Build the graph
            n = 33
            adj2 = adj[:] + [0]
            for v in subset:
                adj2[32] |= (1 << v)
                adj2[v] |= (1 << 32)
            
            best_n = n
            best_edges = []
            for i in range(n):
                bits = adj2[i]
                while bits:
                    bit = bits & -bits
                    j = bit.bit_length() - 1
                    bits ^= bit
                    if i < j:
                        best_edges.append([i, j])
            found = True
            break
    
    return {'n': best_n, 'edges': best_edges}
