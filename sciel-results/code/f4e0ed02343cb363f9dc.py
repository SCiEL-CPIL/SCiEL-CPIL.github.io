import time
import random as _random

_T0 = time.time()
_DEADLINE = _T0 + 115.0

def _random_graph(n, p, seed):
    rng = _random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _solve_instance(edges, n, time_budget):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def _greedy_init(seed):
        rng = _random.Random(seed)
        remaining = set((u, v) for u, v in edges)
        cliques = []
        
        while remaining:
            # Pick a random edge
            u, v = rng.choice(list(remaining))
            
            # Find maximal clique containing u, v in remaining graph
            C = [u, v]
            mask = (1 << u) | (1 << v)
            cand = adj[u] & adj[v]
            cand &= ~mask
            
            # Greedily add vertices
            while cand:
                # Pick vertex with max degree in cand
                best_w = -1
                max_d = -1
                temp = cand
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    # Check if w is connected to all in C
                    # Since C is clique, we just need w connected to all in C.
                    # cand is intersection of neighbors of C, so any w in cand is connected to all in C.
                    # So we can just pick one.
                    # To be safe, check:
                    if (adj[w] & mask) == mask:
                        d = bin(adj[w] & cand).count('1')
                        if d > max_d:
                            max_d = d
                            best_w = w
                    temp ^= lsb
                    
                if best_w == -1:
                    break
                
                C.append(best_w)
                mask |= (1 << best_w)
                cand &= adj[best_w]
                
            cliques.append(C)
            
            # Remove edges in C
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    a, b = C[i], C[j]
                    remaining.discard((a, b))
                    remaining.discard((b, a))
                    
        return cliques

    def _local_search(cliques, time_limit):
        if not cliques:
            return cliques
            
        cliques = [list(c) for c in cliques]
        masks = [0] * len(cliques)
        for i, c in enumerate(cliques):
            m = 0
            for v in c:
                m |= (1 << v)
            masks[i] = m
            
        rng = _random.Random(int(time.time()) + 999)
        start = time.time()
        
        while time.time() - start < time_limit:
            n_cliques = len(cliques)
            if n_cliques < 2:
                break
                
            i = rng.randrange(n_cliques)
            if not cliques[i]:
                continue
            v = rng.choice(cliques[i])
            j = rng.randrange(n_cliques)
            if i == j or not cliques[j]:
                continue
                
            if (adj[v] & masks[j]) == masks[j]:
                cliques[i].remove(v)
                masks[i] &= ~(1 << v)
                cliques[j].append(v)
                masks[j] |= (1 << v)
                
                if not cliques[i]:
                    cliques.pop(i)
                    masks.pop(i)
                    continue
            
            # Try swap
            if rng.random() < 0.2:
                i = rng.randrange(len(cliques))
                j = rng.randrange(len(cliques))
                if i == j or i >= len(cliques) or j >= len(cliques):
                    continue
                if not cliques[i] or not cliques[j]:
                    continue
                v = rng.choice(cliques[i])
                w = rng.choice(cliques[j])
                if v == w:
                    continue
                mask_i_wo_v = masks[i] ^ (1 << v)
                mask_j_wo_w = masks[j] ^ (1 << w)
                if (adj[w] & mask_i_wo_v) == mask_i_wo_v and \
                   (adj[v] & mask_j_wo_w) == mask_j_wo_w:
                    cliques[i].remove(v)
                    cliques[i].append(w)
                    cliques[j].remove(w)
                    cliques[j].append(v)
                    masks[i] = mask_i_wo_v | (1 << w)
                    masks[j] = mask_j_wo_w | (1 << v)
                    
        return [c for c in cliques if c]

    # Run a few greedy inits and keep best
    best_cliques = None
    best_count = float('inf')
    
    for s in range(10):
        c = _greedy_init(s)
        if len(c) < best_count:
            best_count = len(c)
            best_cliques = c
            
    # Local search on best
    improved = _local_search(best_cliques, time_budget - 5)
    if len(improved) < best_count:
        best_cliques = improved
        
    return best_cliques

def construct():
    edges1 = _random_graph(32, 0.6, 601)
    edges2 = _random_graph(64, 0.55, 602)
    
    res1 = _solve_instance(edges1, 32, 40.0)
    res2 = _solve_instance(edges2, 64, 60.0)
    
    return [res1, res2]
