"""Simulated annealing approach for Q3-free dense graph on 16 vertices."""
import time
import random
import math

def construct():
    deadline = time.time() + 118  # safety margin before 150s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set(all_edges)

    # Precompute: for each vertex, list of potential neighbors
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph.
        adj[i] is a bitmask of neighbors of i (u,v NOT yet included in each other's adj).
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)

        # We need to find 6 more vertices a2..a7 such that:
        # u-v, u-a2, u-a4, v-a3, v-a5, a2-a3, a2-a4, a4-a5, a3-a5, a2-a6, a4-a6, a3-a7, a5-a7, a6-a7
        # Actually Q3 has 8 vertices and 12 edges. Let me re-derive.
        # Q3 vertices: 000,001,010,011,100,101,110,111
        # Edges: differ in exactly one bit.
        # Label: a=000, b=001, c=010, d=011, e=100, f=101, g=110, h=111
        # Edges: a-b,a-c,a-e, b-d,b-f, c-d,c-g, e-f,e-g, d-h, f-h, g-h
        
        # So if we have edge (u,v), in Q3 it could be any of the 12 edges.
        # Let's say u plays role 'a' and v plays role 'b' (edge a-b).
        # Then we need: c (neighbor of a, not b), d (neighbor of b, not a, neighbor of c),
        #               e (neighbor of a, not b, not c), f (neighbor of b, not a, not d, neighbor of e),
        #               g (neighbor of c, not a, not d, not e, neighbor of e), 
        #               h (neighbor of d, not a, not b, not f, not g, neighbor of f, neighbor of g)
        
        # Actually let me use a cleaner labeling. In Q3, each vertex has degree 3.
        # Let u and v be connected. In Q3, u has 2 other neighbors (besides v), v has 2 other neighbors (besides u).
        # Let u's other neighbors be p, q. Let v's other neighbors be r, s.
        # Q3 edges: u-v, u-p, u-q, v-r, v-s, p-r, q-s (or p-s, q-r), and then the remaining 4 edges form a 4-cycle among {p,q,r,s} plus connections to a new vertex... 
        
        # Let me just enumerate. Q3 = hypercube. Vertices are 3-bit strings.
        # If u=000, v=001 (differ in bit 0):
        # u's neighbors: 001(v), 010, 100
        # v's neighbors: 000(u), 011, 101
        # So p=010, q=100, r=011, s=101
        # Remaining edges: 010-011(p-r), 010-110, 100-101(q-s), 100-110, 011-111, 101-111
        # Wait: 010-011 yes, 010-110 yes, 100-101 yes, 100-110 yes, 011-111 yes, 101-111 yes.
        # So we need a vertex t=110 that is neighbor of p=010 and q=100, and a vertex w=111 neighbor of r=011 and s=101, with edge t-w.
        
        # Generalizing: given edge (u,v):
        # p, q are the other 2 neighbors of u (in Q3)
        # r, s are the other 2 neighbors of v (in Q3)
        # Need: p-r, q-s (or p-s, q-r)
        # Need: t = common neighbor of p and q (not u,v,r,s)
        # Need: w = common neighbor of r and s (not u,v,p,q,t)
        # Need: edge t-w
        
        # But we don't know which pairing. Let's check both.
        
        # Get other neighbors
        p_candidates = neigh_u  # bits set in neigh_u
        r_candidates = neigh_v  # bits set in neigh_v
        
        # p and q are two distinct vertices from p_candidates
        # r and s are two distinct vertices from r_candidates
        # We need p-r and q-s (or p-s and q-r)
        # t: common neighbor of p,q not in {u,v,r,s}
        # w: common neighbor of r,s not in {u,v,p,q,t}
        # edge t-w
        
        p_bits = p_candidates
        while p_bits:
            p_bit = p_bits & (-p_bits)
            p_bits -= p_bit
            p_idx = p_bit.bit_length() - 1
            
            q_bits = p_candidates & ~(1 << p_idx)
            while q_bits:
                q_bit = q_bits & (-q_bits)
                q_bits -= q_bit
                q_idx = q_bit.bit_length() - 1
                
                # r must be neighbor of v AND neighbor of p (for pairing p-r, q-s)
                r_cand1 = r_candidates & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                # s must be neighbor of v AND neighbor of q
                s_cand1 = r_candidates & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                
                if r_cand1 and s_cand1:
                    r_bits = r_cand1
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        
                        s_cand = s_cand1 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            # t: common neighbor of p and q
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            # w: common neighbor of r and s
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True
                            
                # Also check pairing p-s, q-r
                r_cand2 = r_candidates & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand2 = r_candidates & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                
                if r_cand2 and s_cand2:
                    r_bits = r_cand2
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        
                        s_cand = s_cand2 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return True
        return False

    def count_q3_edges(adj):
        """For each edge, check if removing it would help. Count edges that are 'critical' (part of some Q3)."""
        # We'll use this to guide removal
        return 0

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += bin(adj[u]).count('1')
        return c // 2

    def is_q3_free(adj):
        """Verify the graph is Q3-free by checking all edges."""
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    # Temporarily remove edge to check
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_q3(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return False
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                mask -= bit
        return True

    def greedy_add_back(adj, edge_order, deadline):
        """Try to add edges in order, skipping those that create Q3."""
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def find_q3_involving_edge(adj, u, v):
        """Find a Q3 that uses edge (u,v). Returns the 8 vertices or None."""
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        p_bits = neigh_u
        while p_bits:
            p_bit = p_bits & (-p_bits)
            p_bits -= p_bit
            p_idx = p_bit.bit_length() - 1
            q_bits = neigh_u & ~(1 << p_idx)
            while q_bits:
                q_bit = q_bits & (-q_bits)
                q_bits -= q_bit
                q_idx = q_bit.bit_length() - 1
                
                r_cand1 = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand1 = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand1 and s_cand1:
                    r_bits = r_cand1
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand = s_cand1 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return [u, v, p_idx, q_idx, r_idx, s_idx, t_idx]
                
                r_cand2 = neigh_v & adj[q_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                s_cand2 = neigh_v & adj[p_idx] & ~(1 << p_idx) & ~(1 << q_idx)
                if r_cand2 and s_cand2:
                    r_bits = r_cand2
                    while r_bits:
                        r_bit = r_bits & (-r_bits)
                        r_bits -= r_bit
                        r_idx = r_bit.bit_length() - 1
                        s_cand = s_cand2 & ~(1 << r_idx)
                        if not s_cand:
                            continue
                        s_bits = s_cand
                        while s_bits:
                            s_bit = s_bits & (-s_bits)
                            s_bits -= s_bit
                            s_idx = s_bit.bit_length() - 1
                            used = (1 << u) | (1 << v) | (1 << p_idx) | (1 << q_idx) | (1 << r_idx) | (1 << s_idx)
                            t_cand = (adj[p_idx] & adj[q_idx]) & ~used
                            if not t_cand:
                                continue
                            w_cand = (adj[r_idx] & adj[s_idx]) & ~used
                            if not w_cand:
                                continue
                            t_bits = t_cand
                            while t_bits:
                                t_bit = t_bits & (-t_bits)
                                t_bits -= t_bit
                                t_idx = t_bit.bit_length() - 1
                                w_c = w_cand & ~(1 << t_idx)
                                if w_c and (adj[t_idx] & w_c):
                                    return [u, v, p_idx, q_idx, r_idx, s_idx, t_idx]
        return None

    random.seed(42)
    best_adj = [0] * n
    best_count = 0

    # Strategy: Start from complete graph, remove edges to eliminate Q3s, then add back greedily.
    # Use simulated annealing: at each step, pick an edge involved in a Q3, remove it.
    # Occasionally accept a "worse" move (remove an edge not in any Q3) to escape local minima.
    
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Start from complete graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to make Q3-free
        # Find all edges that are part of at least one Q3
        # Iteratively remove them
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Find an edge that's part of a Q3
            found = False
            # Randomly sample edges
            sample_size = min(20, len(all_edges))
            sample = random.sample(all_edges, sample_size)
            random.shuffle(sample)
            
            for u, v in sample:
                if not (adj[u] & (1 << v)):
                    continue
                # Check if this edge is in a Q3
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                in_q3 = creates_q3(adj, u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if in_q3:
                    # Remove this edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    found = True
                    break
            
            if not found:
                # Check if truly Q3-free
                # Do a full check
                truly_free = True
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        bit = mask & (-mask)
                        v = bit.bit_length() - 1
                        if v > u:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            if creates_q3(adj, u, v):
                                truly_free = False
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                break
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        mask -= bit
                    if not truly_free:
                        break
                
                if truly_free:
                    break
                else:
                    # We need to find and remove more edges
                    # Do a more thorough search
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            bit = mask & (-mask)
                            v = bit.bit_length() - 1
                            if v > u:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                if creates_q3(adj, u, v):
                                    # Remove edge
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                    break
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                            mask -= bit
                        else:
                            continue
                        break
        
        # Phase 2: Greedily add back edges
        remaining = [(u, v) for u in range(n) for v in range(u + 1, n) 
                     if not (adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add_back(adj, remaining, deadline)
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Phase 3: Local search - try swapping edges
        if time.time() < deadline - 15:
            for ls_iter in range(200):
                if time.time() > deadline - 10:
                    break
                # Try to add a random missing edge
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) 
                          if not (adj[u] & (1 << v))]
                if not missing:
                    break
                u, v = random.choice(missing)
                if not creates_q3(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Try removing a random existing edge and adding this one
                    existing = [(a, b) for a in range(n) for b in range(a + 1, n) 
                               if adj[a] & (1 << b)]
                    if existing:
                        a, b = random.choice(existing)
                        # Remove (a,b), try adding (u,v)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        if not creates_q3(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
            
            cnt = count_edges(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

    # Final: try to add any remaining edges to best
    if time.time() < deadline - 3:
        remaining = [(u, v) for u in range(n) for v in range(u + 1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add_back(best_adj, remaining, deadline)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
