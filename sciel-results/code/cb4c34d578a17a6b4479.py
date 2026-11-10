# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed, deadline):
        edges = instance(n, p, seed)
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        uncovered = set(edges)
        cliques = []
        
        while uncovered:
            if time.time() > deadline:
                # Fallback: add remaining edges as single cliques to ensure validity
                for e in uncovered:
                    cliques.append(list(e))
                break
                
            # Heuristic: Pick an edge that is part of a dense subgraph
            # Sample a few edges to find one with high common neighborhood
            if len(uncovered) > 10:
                # Deterministic sampling based on sorted order to avoid random module overhead/variability
                sample = sorted(uncovered)[:10]
            else:
                sample = sorted(uncovered)
                
            best_edge = None
            best_common_count = -1
            for e in sample:
                u, v = e
                common = adj[u] & adj[v]
                # Use bit_count if available, else bin count
                try:
                    count = common.bit_count()
                except AttributeError:
                    count = bin(common).count('1')
                if count > best_common_count:
                    best_common_count = count
                    best_edge = e
                    
            if not best_edge:
                best_edge = next(iter(uncovered))
                
            u, v = best_edge
            clique = [u, v]
            candidates = adj[u] & adj[v]
            
            while candidates:
                if time.time() > deadline:
                    break
                
                best_w = -1
                best_score = -1
                c = candidates
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    # Score: how many other candidates are connected to w?
                    inter = candidates & adj[w]
                    try:
                        score = inter.bit_count()
                    except AttributeError:
                        score = bin(inter).count('1')
                    if score > best_score:
                        best_score = score
                        best_w = w
                    c &= c - 1
                    
                if best_w == -1:
                    break
                    
                clique.append(best_w)
                candidates &= adj[best_w]
                
            cliques.append(clique)
            
            # Remove edges in this clique from uncovered
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    edge = (a, b) if a < b else (b, a)
                    if edge in uncovered:
                        uncovered.remove(edge)
                        
        return cliques

    results = []
    # Allocate time roughly equally, with a bit more for the larger graph if needed, 
    # but since n=64 is only twice n=32, and complexity is similar, split 50/50.
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    half_time = remaining / 2
    
    deadline1 = start_time + half_time
    res1 = solve_instance(32, 0.6, 601, deadline1)
    results.append(res1)
    
    deadline2 = start_time + time_limit
    res2 = solve_instance(64, 0.55, 602, deadline2)
    results.append(res2)
    
    return results
# EVOLVE-BLOCK-END
