import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s kill
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges
    
    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        cliques = []
        
        # remaining_adj[i] = bitmask of neighbors j > i that still have edge (i,j) remaining
        # This is asymmetric: remaining_adj[i] only tracks edges (i,j) where j > i
        # For edge (u,v) with u < v: it's in remaining_adj[u] at bit v
        remaining_adj = [0] * n
        for u, v in edges:
            if u < v:
                remaining_adj[u] |= (1 << v)
            else:
                remaining_adj[v] |= (1 << u)
        
        # Count remaining edges
        def count_remaining():
            cnt = 0
            for i in range(n):
                mask = remaining_adj[i]
                while mask:
                    cnt += 1
                    mask &= mask - 1
            return cnt
        
        def edge_exists(u, v):
            if u > v:
                u, v = v, u
            return bool(remaining_adj[u] & (1 << v))
        
        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            remaining_adj[u] &= ~(1 << v)
        
        def find_clique_from_start(start_vertices, adj_orig, n):
            """Try to build a large clique starting from given vertices."""
            # C is the current clique
            C = list(start_vertices)
            if len(C) < 2:
                return C
            
            # Candidates: vertices connected to ALL in C via remaining edges
            def get_candidates(current_clique):
                cands = []
                for w in range(n):
                    if w in current_clique:
                        continue
                    ok = True
                    for x in current_clique:
                        if not edge_exists(w, x):
                            ok = False
                            break
                    if ok:
                        cands.append(w)
                return cands
            
            candidates = get_candidates(C)
            
            while candidates:
                # Score each candidate by number of connections to other candidates
                best_c = None
                best_score = -1
                for c in candidates:
                    # Count edges from c to other candidates
                    score = 0
                    for c2 in candidates:
                        if c != c2 and edge_exists(c, c2):
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c = c
                
                if best_c is None:
                    break
                
                C.append(best_c)
                # Update candidates: must be connected to best_c
                new_cands = []
                for c in candidates:
                    if c != best_c and edge_exists(c, best_c):
                        new_cands.append(c)
                candidates = new_cands
            
            return C
        
        def find_best_clique(n, adj_orig):
            """Find the best clique using multiple strategies."""
            if count_remaining() == 0:
                return []
            
            best_clique = []
            
            # Strategy 1: Start from each vertex, try to build clique
            # For efficiency, only try a subset of vertices
            tried = set()
            
            # Get vertices with high remaining degree
            deg = [0] * n
            for i in range(n):
                mask = remaining_adj[i]
                while mask:
                    deg[i] += 1
                    mask &= mask - 1
                # Also count edges from lower-indexed vertices to i
                for j in range(i):
                    if remaining_adj[j] & (1 << i):
                        deg[i] += 1
            
            # Sort vertices by degree descending
            verts_by_deg = sorted(range(n), key=lambda x: -deg[x])
            
            # Try top 10 vertices as starting points
            for start in verts_by_deg[:10]:
                if deg[start] < 1:
                    continue
                
                # Get neighbors of start in remaining graph
                neighbors = []
                for w in range(n):
                    if w != start and edge_exists(start, w):
                        neighbors.append(w)
                
                if not neighbors:
                    continue
                
                # Try greedy clique from {start} + best neighbor
                for nb in neighbors[:5]:
                    C = find_clique_from_start([start, nb], adj_orig, n)
                    if len(C) > len(best_clique):
                        best_clique = C
            
            # Strategy 2: Pick random edges and try to extend
            if count_remaining() > 0:
                # Sample some edges
                all_edges = []
                for i in range(n):
                    mask = remaining_adj[i]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        all_edges.append((i, v))
                        mask &= mask - 1
                
                if all_edges:
                    # Try up to 20 random edges
                    sample_size = min(20, len(all_edges))
                    sample = random.sample(all_edges, sample_size)
                    for u, v in sample:
                        C = find_clique_from_start([u, v], adj_orig, n)
                        if len(C) > len(best_clique):
                            best_clique = C
            
            return best_clique
        
        while count_remaining() > 0:
            if time.time() - start_time > timeout:
                # Fallback: add all remaining edges as 2-cliques
                for i in range(n):
                    mask = remaining_adj[i]
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        cliques.append([i, v])
                        mask &= mask - 1
                break
            
            C = find_best_clique(n, adj)
            
            if not C or len(C) < 2:
                # Shouldn't happen if edges remain, but safety check
                # Find any remaining edge
                for i in range(n):
                    mask = remaining_adj[i]
                    if mask:
                        v = (mask & -mask).bit_length() - 1
                        C = [i, v]
                        break
                if not C or len(C) < 2:
                    break
            
            # Remove all edges in C from remaining
            for a, b in combinations(C, 2):
                if a < b:
                    remaining_adj[a] &= ~(1 << b)
                else:
                    remaining_adj[b] &= ~(1 << a)
            
            cliques.append(C)
        
        return cliques
    
    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
