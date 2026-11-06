"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448
    
    # Build list of all edges as (u, v) with u < v
    all_edges = []
    for x in range(nverts):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # For each edge, find the other 3 edges that would complete a C4 with it
    # An edge (u,v) in direction i: C4 is completed if edges (u,u'), (v,v'), (u',v') exist
    # where u' = u^(1<<j), v' = v^(1<<j) for some j != i
    # Actually, for a C4 in the hypercube: vertices u, v, w, z where u~v, v~w, w~z, z~u
    # If we have edge (u,v) along dimension i, then a C4 is formed if there exists
    # dimension j != i such that u^j, v^j are vertices and edges (u,u^j), (v,v^j), (u^j,v^j) exist
    
    # For each edge, precompute the set of "conflict" edge pairs
    # Edge (u,v) along dim i conflicts with edge (a,b) if together they can form a C4
    
    # Better approach: for each edge, store the list of edges that would complete a C4
    # Edge (u,v) along dim i: for each j != i, the C4 would be u-v-v^j-u^j-u
    # So we need edges (u,u^j), (v,v^j), (u^j,v^j)
    # Edge (u,v) is blocked if for some j != i, all three of those edges are present
    
    # For each edge, precompute the "witness" edges
    edge_witnesses = {}
    for (u, v) in all_edges:
        # Determine which dimension this edge is along
        diff = u ^ v
        i = diff.bit_length() - 1
        witnesses = []
        for j in range(d):
            if j != i:
                uj = u ^ (1 << j)
                vj = v ^ (1 << j)
                # The three other edges of the C4
                e1 = (min(u, uj), max(u, uj))
                e2 = (min(v, vj), max(v, vj))
                e3 = (min(uj, vj), max(uj, vj))
                witnesses.append((e1, e2, e3))
        edge_witnesses[(u, v)] = witnesses
    
    # Greedy with randomization: try multiple random orderings, keep best
    best_count = 0
    best_edges = []
    
    np.random.seed(42)
    random.seed(42)
    
    while time.time() < deadline:
        # Random permutation of edges
        order = list(range(len(all_edges)))
        np.random.shuffle(order)
        
        chosen = set()
        chosen_count = 0
        
        for idx in order:
            u, v = all_edges[idx]
            edge = (u, v)
            
            # Check if adding this edge creates a C4
            blocked = False
            for (e1, e2, e3) in edge_witnesses[edge]:
                if e1 in chosen and e2 in chosen and e3 in chosen:
                    blocked = True
                    break
            
            if not blocked:
                chosen.add(edge)
                chosen_count += 1
        
        if chosen_count > best_count:
            best_count = chosen_count
            best_edges = [list(e) for e in chosen]
        
        # Early termination if we hit the known upper bound
        if best_count >= 308:
            break
    
    return best_edges
# EVOLVE-BLOCK-END
