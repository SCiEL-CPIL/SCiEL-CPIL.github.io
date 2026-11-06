"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start = time.time()
    deadline = start + 140
    n = 16
    
    # Q3: 8 vertices, 12 edges
    # Vertices are 3-bit numbers: 0..7
    # Edges connect vertices differing in exactly one bit
    q3_vertices = list(range(8))
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            diff = u ^ v
            if diff & (diff - 1) == 0:  # power of 2
                q3_edges.append((u, v))
    assert len(q3_edges) == 12
    
    q3_adj = {v: [] for v in range(8)}
    for a, b in q3_edges:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
    
    # All possible edges in K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)  # 120
    
    # Precompute: for each pair of edges in K16, check if they can appear in the same Q3 embedding
    # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if there exists a Q3 embedding where
    # both edges are present (as edges of the Q3 subgraph)
    
    # For efficiency: precompute for each Q3 edge position (a,b) in Q3,
    # which pairs of K16 edges can be assigned to positions (a,b) and (c,d) in Q3
    # such that the mapping is a valid partial isomorphism.
    
    # Actually, let's think differently. Two K16 edges conflict if there exists
    # a set of 8 distinct vertices and a bijection to Q3 vertices such that
    # both K16 edges map to Q3 edges.
    
    # More efficiently: for each pair of K16 edges (e1, e2), check if they can
    # be part of the same Q3. This means: there exist Q3 edge positions (a,b) and (c,d)
    # (possibly same) such that we can assign e1's endpoints to {a,b} and e2's endpoints
    # to {c,d} with all 8 vertices distinct, and for every Q3 edge (x,y), the corresponding
    # K16 edge exists.
    
    # This is expensive to precompute for all 120*119/2 pairs. Let's use a different approach.
    
    # Alternative: Use the fact that Q3 is bipartite with parts of size 4.
    # In Q3, vertices 0,3,5,6 are one part (even parity) and 1,2,4,7 are the other (odd parity).
    # Each vertex has degree 3.
    
    # Strategy: Greedy construction with efficient Q3 checking.
    # Start with a good initial graph and try to add edges.
    
    def has_q3(edge_set, new_edge):
        """Check if adding new_edge to edge_set creates a Q3 subgraph."""
        u, v = new_edge
        u, v = min(u,v), max(u,v)
        
        # Try all ways to embed Q3 such that (u,v) is one of the Q3 edges
        for a, b in q3_edges:
            # Try both orientations
            for ua, va in [(u, v), (v, u)]:
                assign = {a: ua, b: va}
                used = {ua, va}
                
                # Order remaining Q3 vertices by some heuristic
                remaining = [x for x in range(8) if x not in (a, b)]
                
                # Try to complete the embedding via backtracking
                def bt(rem):
                    if not rem:
                        # Check all Q3 edges are in edge_set or are new_edge
                        for ea, eb in q3_edges:
                            x, y = assign[ea], assign[eb]
                            if (min(x,y), max(x,y)) == new_edge:
                                continue
                            if (min(x,y), max(x,y)) not in edge_set:
                                return False
                        return True
                    
                    # Pick the remaining Q3 vertex with most constraints
                    best_qv = None
                    best_cands = None
                    for qv in rem:
                        cands = []
                        for cand in range(n):
                            if cand in used:
                                continue
                            ok = True
                            for nb in q3_adj[qv]:
                                if nb in assign:
                                    x = assign[nb]
                                    if (min(cand, x), max(cand, x)) == new_edge:
                                        continue
                                    if (min(cand, x), max(cand, x)) not in edge_set:
                                        ok = False
                                        break
                            if ok:
                                cands.append(cand)
                        if best_cands is None or len(cands) < len(best_cands):
                            best_qv = qv
                            best_cands = cands
                        if len(cands) == 0:
                            return False
                    
                    for cand in best_cands:
                        assign[best_qv] = cand
                        used.add(cand)
                        new_rem = [x for x in rem if x != best_qv]
                        if bt(new_rem):
                            return True
                        del assign[best_qv]
                        used.remove(cand)
                    return False
                
                if bt(remaining):
                    return True
        return False
    
    # Initial graph: K_{3,13}
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    # Try to add edges greedily with multiple random orderings
    best_edges = set(edges)
    best_count = len(edges)
    
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                edges.add((a, b))
        
        missing = list(set(all_edges) - edges)
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 5:
                break
            if has_q3(edges, e):
                continue
            edges.add(e)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
    
    # Try local improvement: remove one edge, try to add two
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        edges = set(best_edges)
        for e in list(edges):
            if time.time() > deadline - 5:
                break
            edges.remove(e)
            # Try to add edges that were previously rejected
            missing = list(set(all_edges) - edges)
            random.shuffle(missing)
            added = 0
            for ne in missing:
                if time.time() > deadline - 3:
                    break
                if has_q3(edges, ne):
                    continue
                edges.add(ne)
                added += 1
                if added >= 2:
                    break
            if len(edges) > len(best_edges):
                best_edges = set(edges)
                best_count = len(edges)
                improved = True
    
    return [[u, v] for u, v in best_edges]
# EVOLVE-BLOCK-END
