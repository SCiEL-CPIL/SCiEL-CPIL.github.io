import time
import itertools
import sys

def construct():
    start_time = time.time()
    n = 16
    vertices = list(range(n))
    
    # Precompute all edges and assign IDs
    all_edges = []
    edge_to_id = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_to_id[(u, v)] = len(all_edges)
            all_edges.append((u, v))
            
    m = len(all_edges) # 120
    
    # Initialize state
    # adj_mask[i] is a 16-bit integer where bit j is 1 if (i,j) is an edge
    adj_mask = [0] * n
    # current_edges is a set of edge IDs
    current_edges = set()
    
    # Initial state: K_{3,13}
    # Vertices 0,1,2 connected to 3..15
    for a in range(3):
        for b in range(3, 16):
            u, v = a, b
            if u > v: u, v = v, u
            eid = edge_to_id[(u, v)]
            current_edges.add(eid)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
    best_count = len(current_edges)
    best_edges = set(current_edges)
    
    # Candidates: all edges not in current_edges
    candidates = [eid for eid in range(m) if eid not in current_edges]
    
    # Heuristic: Sort candidates.
    # Edges between vertices 3..15 are "internal" to the large part.
    # Edges between 0..2 and 3..15 are "cross".
    # Edges between 0..2 are "small part internal".
    # Adding cross edges might create Q3s more easily?
    # Let's prioritize edges that are "less dangerous".
    # Or just prioritize to find a large solution quickly.
    # Let's try sorting by endpoint degrees in the current graph? 
    # Since we start with K_{3,13}, degrees are fixed initially.
    # deg(0..2) = 13, deg(3..15) = 3.
    
    def edge_key(eid):
        u, v = all_edges[eid]
        # Prefer edges within the large set (3..15) first?
        # If we add many edges within 3..15, we might create Q3s with the 0..2 vertices.
        # A Q3 requires 4 vertices in each bipartition.
        # If we have K_{3,13}, we have 3 in one part. We need 1 more in that part to make 4.
        # So if we add an edge (0,1), we still have only 3 in that part? No, 0 and 1 are in the same part of K_{3,13}.
        # In K_{3,13}, the bipartition is {0,1,2} and {3..15}.
        # If we add edge (0,1), the graph is no longer bipartite with that partition.
        # But does it contain Q3?
        # Q3 is bipartite. If the graph contains a Q3, that Q3 must be bipartite.
        # The vertices of the Q3 must be 2-colorable.
        
        # Let's just use a simple heuristic: 
        # Edges with lower "potential to form Q3" first.
        # Approximation: sum of degrees of endpoints.
        du = bin(adj_mask[u]).count('1')
        dv = bin(adj_mask[v]).count('1')
        return du + dv
        
    # Static sort might not be best as degrees change.
    # But re-sorting every step is expensive.
    # Let's stick to a static order based on initial state or just index order.
    # Actually, let's try to add edges that *don't* create Q3s first?
    # We can't know without checking.
    
    # Let's use the index order but maybe reverse it?
    # Or sort by (u,v) where u,v are in 3..15 first.
    def static_key(eid):
        u, v = all_edges[eid]
        if u >= 3 and v >= 3:
            return 0
        elif u < 3 and v < 3:
            return 2
        else:
            return 1
            
    candidates.sort(key=static_key)
    
    def creates_q3(u, v):
        """
        Check if adding edge (u,v) creates a Q3.
        u, v are not currently connected.
        """
        # In Q3, u and v are adjacent.
        # u has neighbors x3, x4 in Q3.
        # v has neighbors x2, x5 in Q3.
        # x2, x3, x4, x5 are distinct and not u, v.
        # Edges: (u,x3), (u,x4), (v,x2), (v,x5), (x2,x3), (x4,x5).
        # Plus the rest of the cube.
        
        n_u = adj_mask[u]
        n_v = adj_mask[v]
        
        # Iterate x3 in N(u)
        mask_u = n_u
        while mask_u:
            lsb_u = mask_u & -mask_u
            x3 = lsb_u.bit_length() - 1
            mask_u ^= lsb_u
            
            # x3 must be in N(u)
            # x3 != v (v is not in N(u) yet)
            
            # Iterate x4 in N(u), x4 != x3
            mask_u2 = n_u & ~(1 << x3)
            while mask_u2:
                lsb_u2 = mask_u2 & -mask_u2
                x4 = lsb_u2.bit_length() - 1
                mask_u2 ^= lsb_u2
                
                # Iterate x2 in N(v)
                mask_v = n_v
                while mask_v:
                    lsb_v = mask_v & -mask_v
                    x2 = lsb_v.bit_length() - 1
                    mask_v ^= lsb_v
                    
                    if x2 == x3 or x2 == x4: continue
                    
                    # Check edge (x2, x3)
                    if not (adj_mask[x2] & (1 << x3)): continue
                    
                    # Iterate x5 in N(v), x5 != x2, x5 != x3, x5 != x4
                    mask_v2 = n_v & ~(1 << x2)
                    while mask_v2:
                        lsb_v2 = mask_v2 & -mask_v2
                        x5 = lsb_v2.bit_length() - 1
                        mask_v2 ^= lsb_v2
                        
                        if x5 == x3 or x5 == x4: continue
                        
                        # Check edge (x4, x5)
                        if not (adj_mask[x4] & (1 << x5)): continue
                        
                        # Distinctness check for x2,x3,x4,x5
                        # We already checked x2!=x3, x2!=x4, x5!=x2, x5!=x3, x5!=x4
                        # x3!=x4 is ensured by loop.
                        
                        # Now find x6, x7
                        # x6 in N(x2) ∩ N(x5)
                        # x7 in N(x3) ∩ N(x4)
                        # Edge (x6, x7) must exist.
                        
                        common_25 = adj_mask[x2] & adj_mask[x5]
                        common_34 = adj_mask[x3] & adj_mask[x4]
                        
                        # Exclude u, v, x2, x3, x4, x5 from common sets?
                        # x6 must be distinct from u,v,x2,x3,x4,x5.
                        # Since x6 is neighbor of x2, and x2 is neighbor of v...
                        # x6 could be v? If (x2,v) is edge, yes.
                        # But x6 must be distinct from v.
                        # So we should mask out u, v, x2, x3, x4, x5.
                        
                        forbidden_mask = (1 << u) | (1 << v) | (1 << x2) | (1 << x3) | (1 << x4) | (1 << x5)
                        
                        c25 = common_25 & ~forbidden_mask
                        c34 = common_34 & ~forbidden_mask
                        
                        # Check if there exists x6 in c25, x7 in c34 such that edge (x6,x7) exists.
                        # Iterate x6 in c25
                        m25 = c25
                        while m25:
                            lsb_25 = m25 & -m25
                            x6 = lsb_25.bit_length() - 1
                            m25 ^= lsb_25
                            
                            # x7 must be in c34 AND neighbor of x6
                            # AND x7 != x6
                            m34 = c34 & adj_mask[x6]
                            if m34 & ~(1 << x6): # If any bit set
                                return True
        return False

    def search(current_edges, adj_mask, candidates, count, depth):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > 115:
            return
            
        # Upper bound
        if count + len(candidates) <= best_count:
            return
            
        if not candidates:
            if count > best_count:
                best_count = count
                best_edges = set(current_edges)
            return
            
        # Pick candidate
        # Heuristic: Pick the one with highest "conflict potential"? 
        # Or just first in sorted list.
        eid = candidates[0]
        rest = candidates[1:]
        u, v = all_edges[eid]
        
        # Try adding eid
        if not creates_q3(u, v):
            # Add edge
            new_adj_mask = adj_mask[:]
            new_adj_mask[u] |= (1 << v)
            new_adj_mask[v] |= (1 << u)
            new_edges = current_edges | {eid}
            
            search(new_edges, new_adj_mask, rest, count + 1, depth + 1)
            
        # Try skipping eid
        search(current_edges, adj_mask, rest, count, depth + 1)
        
    search(current_edges, adj_mask, candidates, best_count, 0)
    
    return [list(all_edges[e]) for e in best_edges]
