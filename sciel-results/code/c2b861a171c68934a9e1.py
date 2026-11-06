import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 115.0
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        # adj[i] is a bitmask of neighbors of i
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
        # remaining mask: for each vertex, which neighbors are still "active"
        # Actually, track remaining edges as a set of (u,v) pairs
        # But for speed, use a 2D boolean or bitmask per vertex
        # remaining_adj[i] = bitmask of j > i such that (i,j) is still remaining
        # This is asymmetric but that's fine for our purposes
        
        # Better: use a set of frozensets or just a set of (min,max) tuples
        # For n=64, max edges ~2016, so a set is fine
        
        remaining = set(edges)
        cliques = []
        
        # Precompute: for each edge, how many common neighbors does it have?
        # This tells us the max clique size containing that edge
        
        while remaining:
            if time.time() - start_time > timeout:
                # Fallback: add all remaining edges as 2-cliques
                for u, v in remaining:
                    cliques.append([u, v])
                break
            
            # Find the most constrained edge: one with fewest common neighbors in remaining
            # common neighbors of (u,v) = adj[u] & adj[v] & remaining_mask
            # But we need to check which of those neighbors form edges with both u and v in remaining
            
            # For speed, compute for each edge the number of vertices w such that
            # (u,w) and (v,w) are both in remaining
            # This is the number of triangles containing (u,v) in the remaining graph
            
            best_edge = None
            best_common = -1  # We want the MINIMUM common neighbors (most constrained)
            # But actually, we want to maximize the clique we can extract
            # So maybe we want the edge with the LARGEST common neighborhood?
            # No - we want to handle constrained edges first to avoid being stuck with them later.
            
            # Let's try: pick the edge with the smallest number of common neighbors
            # among remaining edges, to handle "hard" edges first
            
            # For efficiency, sample a subset of edges if there are many
            if len(remaining) > 100:
                import random as _r
                sample = _r.sample(list(remaining), 100)
            else:
                sample = list(remaining)
            
            for u, v in sample:
                # Count common neighbors in remaining
                common = 0
                # Check all w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    e1 = (min(u, w), max(u, w))
                    e2 = (min(v, w), max(v, w))
                    if e1 in remaining and e2 in remaining:
                        common += 1
                if best_edge is None or common < best_common:
                    best_common = common
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            # Find maximum clique containing (u,v) in remaining graph
            # Candidates: vertices w such that (u,w) and (v,w) are in remaining
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                e1 = (min(u, w), max(u, w))
                e2 = (min(v, w), max(v, w))
                if e1 in remaining and e2 in remaining:
                    candidates.append(w)
            
            # Now find a maximal clique in {u,v} ∪ candidates where all edges exist in remaining
            # Use greedy: start with {u,v}, repeatedly add the candidate with most connections to current clique
            C = [u, v]
            cand_set = set(candidates)
            
            while cand_set:
                best_c = None
                best_score = -1
                for c in cand_set:
                    # Check c connects to all in C via remaining
                    ok = True
                    for x in C:
                        e = (min(c, x), max(c, x))
                        if e not in remaining:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Score: connections to other candidates
                    score = 0
                    for c2 in cand_set:
                        if c2 != c:
                            e2 = (min(c, c2), max(c, c2))
                            if e2 in remaining:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c is None:
                    break
                C.append(best_c)
                # Update candidates
                new_cands = set()
                for c in cand_set:
                    if c != best_c:
                        e = (min(best_c, c), max(best_c, c))
                        if e in remaining:
                            new_cands.add(c)
                cand_set = new_cands
            
            # Remove edges in C from remaining
            for a, b in combinations(C, 2):
                e = (min(a, b), max(a, b))
                if e in remaining:
                    remaining.remove(e)
            
            cliques.append(C)
        
        return cliques
    
    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
