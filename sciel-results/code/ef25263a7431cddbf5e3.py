"""Dense Q3-free graph on 16 vertices via bitmask precomputation + local search."""
import time
import random
import math

def construct():
    deadline = time.time() + 118
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    # Precompute all Q3 embeddings in K16
    # A Q3 has 8 vertices and 12 edges
    # We enumerate all ways to pick 8 vertices and check if they can form a Q3
    # But that's C(16,8) * (8!/48) which is too much
    # Instead, we'll use a different approach: precompute for each edge, which Q3s it belongs to
    
    # Actually, let's precompute all Q3s in K16 more efficiently
    # A Q3 is determined by choosing 3 "directions" and a base vertex
    # In K16, we can label vertices 0-15 as 4-bit numbers
    # A Q3 can be embedded by choosing 3 of the 4 bit positions and a base
    
    # For each choice of 3 bit positions (out of 4) and a base vertex (with the 4th bit fixed),
    # we get a Q3
    
    q3_edges = []  # list of sets of 12 edges
    q3_vertices = []  # list of sets of 8 vertices
    
    # Choose 3 bit positions out of 4
    for bits in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
        # For each base (the remaining bit can be 0 or 1)
        for base_bit in [0, 1]:
            # Generate all 8 vertices of this Q3
            vertices = []
            for i in range(8):
                b0 = (i >> 0) & 1
                b1 = (i >> 1) & 1
                b2 = (i >> 2) & 1
                # The remaining bit is base_bit
                if bits[0] == 0:
                    v = (b0 << 0) | (b1 << 1) | (b2 << 2) | (base_bit << 3)
                elif bits[0] == 1:
                    v = (base_bit << 0) | (b0 << 1) | (b1 << 2) | (b2 << 3)
                elif bits[0] == 2:
                    v = (base_bit << 0) | (b1 << 1) | (b0 << 2) | (b2 << 3)
                else:
                    v = (b0 << 0) | (b1 << 1) | (b2 << 2) | (base_bit << 3)
                vertices.append(v)
            
            # Generate edges: vertices that differ in exactly one of the chosen bits
            edges = set()
            for i in range(8):
                for j in range(i+1, 8):
                    # Check if i and j differ in exactly one bit position
                    diff = i ^ j
                    if diff in [1, 2, 4]:  # power of 2
                        u, v = vertices[i], vertices[j]
                        edges.add((min(u,v), max(u,v)))
            
            q3_edges.append(frozenset(edges))
            q3_vertices.append(frozenset(vertices))
    
    # For each edge, precompute which Q3s contain it
    edge_to_q3s = {e: [] for e in all_edges}
    for i, q3e in enumerate(q3_edges):
        for e in q3e:
            if e in edge_to_q3s:
                edge_to_q3s[e].append(i)
    
    # Bitmask representation of edges
    # Map each edge to an index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = all_edges  # index -> edge
    
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph.
        adj[i] is a bitmask of neighbors of i (u,v NOT yet included).
        """
        e = (min(u,v), max(u,v))
        if e not in edge_to_q3s:
            return False
        
        for q3_idx in edge_to_q3s[e]:
            q3_e = q3_edges[q3_idx]
            # Check if all other 11 edges of this Q3 are present
            ok = True
            for qe in q3_e:
                if qe == e:
                    continue
                qu, qv = qe
                if not (adj[qu] & (1 << qv)):
                    ok = False
                    break
            if ok:
                return True
        return False

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

    random.seed(42)
    best_adj = [0] * n
    best_count = 0

    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Start from complete graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 1: Remove edges to make Q3-free
        max_iter = 300
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Find an edge that's part of a Q3
            found = False
            sample_size = min(30, len(all_edges))
            sample = random.sample(all_edges, sample_size)
            random.shuffle(sample)
            
            for u, v in sample:
                if not (adj[u] & (1 << v)):
                    continue
                e = (min(u,v), max(u,v))
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
                    # More thorough search
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            bit = mask & (-mask)
                            v = bit.bit_length() - 1
                            if v > u:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                if creates_q3(adj, u, v):
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
        
        # Phase 3: Local search
        if time.time() < deadline - 15:
            for ls_iter in range(500):
                if time.time() > deadline - 10:
                    break
                missing = [(u, v) for u in range(n) for v in range(u + 1, n) 
                          if not (adj[u] & (1 << v))]
                if not missing:
                    break
                u, v = random.choice(missing)
                if not creates_q3(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    existing = [(a, b) for a in range(n) for b in range(a + 1, n) 
                               if adj[a] & (1 << b)]
                    if existing:
                        a, b = random.choice(existing)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        if not creates_q3(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
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
