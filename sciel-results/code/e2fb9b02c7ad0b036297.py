"""Dense Q3-free graph on 16 vertices via complement removal + intensive local search."""
import time
import random

def construct():
    deadline = time.time() + 140  # Safety margin
    n = 16
    full_mask = (1 << n) - 1

    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def find_q3_in_graph(adj):
        """
        Find a Q3 in the current graph adj.
        Returns a tuple of 8 vertices that form a Q3, or None.
        
        A Q3 is a bipartite graph with 8 vertices and 12 edges.
        We search for it by looking for 4-cycles that can be extended.
        
        Strategy: For each pair of non-adjacent vertices (u,v), check if they can be opposite vertices of a Q3.
        In Q3, opposite vertices have no common neighbors.
        """
        # For each pair (u, v) that are NOT adjacent
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue  # They are adjacent, can't be opposite in Q3
                
                # In Q3, opposite vertices have no common neighbors
                common_uv = adj[u] & adj[v]
                if common_uv:
                    continue  # Can't be opposite in Q3
                
                # Now look for the 6 other vertices that would form the Q3 with u and v as opposites
                # The 6 vertices split into two sets of 3: N(u) and N(v) should each have size 3
                # and they should be disjoint from {u, v} and from each other's sets in the right way.
                
                nu = adj[u]  # neighbors of u
                nv = adj[v]  # neighbors of v
                
                # We need |nu| >= 3 and |nv| >= 3
                if bin(nu).count('1') < 3 or bin(nv).count('1') < 3:
                    continue
                
                # Pick 3 neighbors of u and 3 neighbors of v
                # The 3 neighbors of u should each be connected to exactly 2 of the 3 neighbors of v (the ones that form the "middle layer")
                # Actually, let's think about the Q3 structure more carefully.
                
                # Q3 vertices can be labeled 0..7 (binary 000..111).
                # u = 000, v = 111 (opposite).
                # N(u) = {001, 010, 100}
                # N(v) = {110, 101, 011}
                # The middle vertices are: 001, 010, 100, 110, 101, 011
                # Edges: 001-111, 010-111, 100-111, 110-000, 101-000, 011-000
                # Plus: 001-011, 001-101, 010-011, 010-110, 100-101, 100-110
                # Wait, let me reconsider. 
                # 000 connected to: 001, 010, 100
                # 111 connected to: 110, 101, 011
                # 001 connected to: 000, 111, 011, 101
                # 010 connected to: 000, 111, 011, 110
                # 100 connected to: 000, 111, 101, 110
                # 110 connected to: 111, 010, 100, 011... wait no.
                # 110 = 6. Connected to 111(7), 010(2), 100(4), 111... 
                # Let me just enumerate: 
                # 000: 001, 010, 100
                # 001: 000, 011, 101, 111
                # 010: 000, 011, 110, 111
                # 100: 000, 101, 110, 111
                # 011: 001, 010, 111, 110... wait.
                # 011 = 3. Connected to 001(1), 010(2), 111(7), 011... 
                # 011 to 011? No. 011 to 111? Yes (distance 1). 011 to 001? Yes. 011 to 010? Yes. 011 to 101? No (dist 2). 011 to 110? No (dist 2).
                # So 011: 001, 010, 111. And also 011 to... let me check all.
                # 011 (3): differs from 001(1) in bit 1, from 010(2) in bit 0, from 111(7) in bit 2. 
                # Also 011 to 011? No. To 101(5)? dist 2. To 110(6)? dist 2. To 000(0)? dist 2.
                # So 011 connects to: 001, 010, 111.
                # Similarly 101(5): connects to 001, 100, 111.
                # 110(6): connects to 010, 100, 111.
                
                # So the structure is:
                # u=000 connects to A={001,010,100}
                # v=111 connects to B={110,101,011}
                # 001 connects to 011, 101 (and u, v)
                # 010 connects to 011, 110 (and u, v)
                # 100 connects to 101, 110 (and u, v)
                # 011 connects to 001, 010 (and v)
                # 101 connects to 001, 100 (and v)
                # 110 connects to 010, 100 (and v)
                
                # So we need:
                # - A = {a1, a2, a3} subset of N(u)
                # - B = {b1, b2, b3} subset of N(v)
                # - a1 connected to b1, b2
                # - a2 connected to b1, b3
                # - a3 connected to b2, b3
                # - No other edges between A and B (in the Q3, each a_i connects to exactly 2 b_j)
                # - A and B disjoint, and A union B has no self-loops obviously.
                
                nu_list = []
                nm = nu
                while nm:
                    bit = nm & (-nm)
                    idx = bit.bit_length() - 1
                    if idx != u and idx != v:
                        nu_list.append(idx)
                    nm -= bit
                
                nv_list = []
                nm = nv
                while nm:
                    bit = nm & (-nm)
                    idx = bit.bit_length() - 1
                    if idx != u and idx != v:
                        nv_list.append(idx)
                    nm -= bit
                
                if len(nu_list) < 3 or len(nv_list) < 3:
                    continue
                
                # Try all combinations of 3 from nu_list and 3 from nv_list
                from itertools import combinations
                for A in combinations(nu_list, 3):
                    A_set = set(A)
                    # B must be disjoint from A
                    nv_candidate = [x for x in nv_list if x not in A_set]
                    if len(nv_candidate) < 3:
                        continue
                    for B in combinations(nv_candidate, 3):
                        B_set = set(B)
                        # Check the edge pattern
                        # a1 should connect to exactly 2 of B
                        # a2 should connect to exactly 2 of B
                        # a3 should connect to exactly 2 of B
                        # And the pattern should match: 
                        # a1-b1, a1-b2; a2-b1, a2-b3; a3-b2, a3-b3
                        # This is a specific pattern. Let's just check if the induced subgraph on A union B 
                        # has exactly the right edges.
                        # In the Q3, the edges between A and B are: 
                        # a1-b1, a1-b2, a2-b1, a2-b3, a3-b2, a3-b3
                        # That's 6 edges.
                        # But we don't know the labeling. So let's just check if the bipartite graph 
                        # between A and B has exactly 6 edges and each vertex has degree 2.
                        
                        edge_count = 0
                        deg_a = {a: 0 for a in A}
                        deg_b = {b: 0 for b in B}
                        for a in A:
                            for b in B:
                                if adj[a] & (1 << b):
                                    edge_count += 1
                                    deg_a[a] += 1
                                    deg_b[b] += 1
                        
                        if edge_count != 6:
                            continue
                        if any(d != 2 for d in deg_a.values()):
                            continue
                        if any(d != 2 for d in deg_b.values()):
                            continue
                        
                        # Also check that there are no edges within A or within B that would be extra
                        # In Q3, there are no edges within A (they are all at distance 2 from each other)
                        # and no edges within B.
                        for i in range(3):
                            for j in range(i+1, 3):
                                if adj[A[i]] & (1 << A[j]):
                                    # This would be an extra edge, but it's fine if it exists in the graph,
                                    # as long as the 8-vertex subgraph contains a Q3 as a subgraph (not necessarily induced).
                                    # The task says "subgraph isomorphic to Q3", so extra edges are OK.
                                    pass
                        
                        # Found a Q3!
                        return (u, v) + A + B
        
        return None

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

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

    def get_missing_edges(adj):
        return [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]

    def get_existing_edges(adj):
        res = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    res.append((u, v))
                mask -= bit
        return res

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_iters=500):
        """
        Local search: try to add edges. If stuck, try targeted swaps with simulated annealing.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        temperature = 1.0
        cooling = 0.995
        
        # Phase 1: Greedy add missing edges
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        greedy_add(adj, missing)
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Phase 2: Simulated annealing with swaps
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random missing edge
            missing = get_missing_edges(adj)
            found_add = False
            if missing:
                for _ in range(15):
                    if not missing:
                        break
                    idx = random.randint(0, len(missing) - 1)
                    u, v = missing[idx]
                    if not creates_q3_fast(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        missing.pop(idx)
                        current_count += 1
                        found_add = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                        break
            
            if found_add:
                temperature = max(0.1, temperature * cooling)
                continue
            
            # No edge could be added. Perform a swap: remove one edge, try to add others.
            existing = get_existing_edges(adj)
            if not existing:
                break
            
            # Pick a random edge to remove
            u_rem, v_rem = random.choice(existing)
            
            trial = adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            
            # Try to add edges to trial
            missing_trial = get_missing_edges(trial)
            random.shuffle(missing_trial)
            greedy_add(trial, missing_trial)
            
            new_count = count_edges(trial)
            delta = new_count - current_count
            
            if delta > 0 or random.random() < (1.0 / (1.0 + __import__('math').exp(-delta / max(temperature, 0.01)))):
                adj = trial[:]
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            
            temperature = max(0.1, temperature * cooling)
        
        return best_adj, best_count

    def complement_removal():
        """
        Start from K16 and remove edges to break all Q3s.
        """
        adj = [full_mask ^ (1 << u) for u in range(n)]
        
        max_prune_iters = 500
        for _ in range(max_prune_iters):
            if time.time() > deadline - 20:
                break
            
            q3 = find_q3_in_graph(adj)
            if not q3:
                break
            
            # Remove a random edge from the Q3
            verts = q3
            # Q3 edges: we need to figure out which pairs are edges in the Q3
            # For simplicity, just pick a random pair of vertices from the 8 and remove their edge
            # if it exists.
            edge_candidates = []
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = verts[i], verts[j]
                    if u > v:
                        u, v = v, u
                    if adj[u] & (1 << v):
                        edge_candidates.append((u, v))
            
            if edge_candidates:
                u, v = random.choice(edge_candidates)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return adj

    random.seed(42)
    best_adj = None
    best_count = -1
    
    # Strategy 1: Complement removal
    if time.time() < deadline - 30:
        adj = complement_removal()
        cnt = count_edges(adj)
        if best_adj is None or cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        if time.time() < deadline - 15:
            adj2, cnt2 = local_search(adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2[:]
    
    # Strategy 2: Random greedy from empty + local search
    for i in range(30):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        greedy_add(adj, all_edges)
        
        adj3, cnt3 = local_search(adj)
        
        if cnt3 > best_count:
            best_count = cnt3
            best_adj = adj3[:]
    
    # Final polish
    if best_adj is not None and time.time() < deadline - 2:
        remaining = get_missing_edges(best_adj)
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    if best_adj is None:
        # Fallback: empty graph
        return []
    
    return edges_from_adj(best_adj)
