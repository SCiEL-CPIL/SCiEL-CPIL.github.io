import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    n = 16
    
    # Precompute bit masks for vertices
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def creates_q3(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        This is the critical bottleneck. We optimize it heavily.
        
        In Q3, vertices are 3-bit vectors. u and v are adjacent, so they differ in 1 bit.
        Let d = u xor v.
        The other 6 vertices in the Q3 are determined by u, v, and the other 2 dimensions.
        Let the dimensions be 0, 1, 2. Assume d has bit 0 set.
        Then the Q3 vertices are:
        u, v
        u^2, v^2  (diff in bit 1)
        u^4, v^4  (diff in bit 2)
        u^6, v^6  (diff in bits 1 and 2)
        
        Wait, the vertices are:
        000 (u)
        100 (v)
        001 (u^1)
        010 (u^2)
        101 (v^1)
        110 (v^2)
        011 (u^3)
        111 (v^3)
        
        Here I used bit values 1, 2, 4.
        If d=1 (bit 0), the other bits are 1 and 2.
        Vertices:
        u
        v = u ^ 1
        u ^ 2
        u ^ 4
        v ^ 2 = u ^ 3
        v ^ 4 = u ^ 5
        u ^ 6
        v ^ 6 = u ^ 7
        
        So the 8 vertices are u, u^1, u^2, u^4, u^3, u^5, u^6, u^7.
        Actually, it's easier to iterate over the 3 possible "other" dimensions.
        Since n=16, vertices are 4-bit vectors (0..15).
        Q3 is a subgraph on 8 vertices.
        Any Q3 in a 4-bit space corresponds to fixing 1 bit and varying the other 3?
        No, Q3 is 3-dimensional. In a 4-dimensional space (16 vertices), a Q3 is defined by
        choosing 3 dimensions to vary and 1 dimension to fix.
        
        So, for a fixed edge (u, v), let d = u xor v. d must be a power of 2.
        Let dim_v = index of the set bit in d.
        The other 3 dimensions are the remaining 3 bits in the 4-bit representation.
        Let these be dims d1, d2, d3.
        The 8 vertices of the potential Q3 are:
        u, v
        u ^ (1<<d1), v ^ (1<<d1)
        u ^ (1<<d2), v ^ (1<<d2)
        u ^ (1<<d3), v ^ (1<<d3)
        u ^ (1<<d1) ^ (1<<d2), v ^ (1<<d1) ^ (1<<d2)
        u ^ (1<<d1) ^ (1<<d3), v ^ (1<<d1) ^ (1<<d3)
        u ^ (1<<d2) ^ (1<<d3), v ^ (1<<d2) ^ (1<<d3)
        u ^ (1<<d1) ^ (1<<d2) ^ (1<<d3), v ^ (1<<d1) ^ (1<<d2) ^ (1<<d3)
        
        We need to check if all 12 edges of this Q3 exist in G (including the new edge u-v).
        Since we are checking if adding (u,v) *creates* a Q3, we assume (u,v) is the new edge.
        We need to check if the other 11 edges exist in the current graph.
        
        There are only 3 choices for the set of 3 dimensions (since one is fixed by u,v).
        Actually, the dimension of u and v is fixed. The other 3 are fixed too.
        So there is exactly ONE candidate Q3 containing edge (u,v) for each choice of the "plane"?
        No. The edge (u,v) fixes the direction. The Q3 must contain this edge.
        The other 2 dimensions of the Q3 must be chosen from the 3 available dimensions in the 4-bit space?
        Wait. The vertices are 0..15 (4 bits).
        A Q3 is a 3-cube. It lives in a 3-dimensional subspace.
        If u and v differ in bit k, then the Q3 must vary in bit k and two other bits, say i and j.
        The fourth bit l is fixed.
        So for a given edge (u,v), there are C(3,2) = 3 possible Q3s that could contain this edge?
        No. The bit k is the one that differs. The other 3 bits are candidates.
        We must choose 2 of the remaining 3 bits to be the other axes of the cube.
        So there are 3 possible Q3 structures to check for each edge (u,v).
        
        Let's verify.
        u=0 (0000), v=1 (0001). Diff bit 0.
        Remaining bits: 1, 2, 3.
        Choice 1: Vary bits 1, 2. Fix bit 3.
        Vertices: 0, 1, 2, 3, 4, 5, 6, 7. (Bit 3 is 0).
        Choice 2: Vary bits 1, 3. Fix bit 2.
        Vertices: 0, 1, 2, 3, 8, 9, 10, 11. (Bit 2 is 0).
        Choice 3: Vary bits 2, 3. Fix bit 1.
        Vertices: 0, 1, 4, 5, 8, 9, 12, 13. (Bit 1 is 0).
        
        Yes, exactly 3 candidates.
        
        For each candidate, we check if all 12 edges exist.
        Edges in Q3:
        (u, v) - new edge, assumed present.
        (u^i, v^i), (u^j, v^j)
        (u^i, u^ij), (v^i, v^ij)
        (u^j, u^ij), (v^j, v^ij)
        (u^i, u^j) ? No, u^i and u^j differ in 2 bits, not adjacent in Q3.
        Wait, in Q3, edges connect vertices differing in 1 bit.
        So edges are:
        u-v
        u^i - v^i
        u^j - v^j
        u^i - u^ij  (diff in j)
        v^i - v^ij  (diff in j)
        u^j - u^ij  (diff in i)
        v^j - v^ij  (diff in i)
        u^ij - u^ijk? No, only 3 dims.
        Let dims be k (fixed diff), i, j.
        Vertices:
        0: u
        1: v (u^k)
        2: u^i
        3: v^i (u^k^i)
        4: u^j
        5: v^j (u^k^j)
        6: u^ij
        7: v^ij (u^k^i^j)
        
        Edges:
        0-1 (u-v)
        0-2 (u-u^i)
        0-4 (u-u^j)
        1-3 (v-v^i)
        1-5 (v-v^j)
        2-3 (u^i-v^i)
        2-6 (u^i-u^ij)
        3-7 (v^i-v^ij)
        4-5 (u^j-v^j)
        4-6 (u^j-u^ij)
        5-7 (v^j-v^ij)
        6-7 (u^ij-v^ij)
        
        Total 12 edges.
        We need to check if these 11 edges (excluding u-v) are in `adj`.
        """
        d = u ^ v
        if d == 0 or (d & (d - 1)) != 0:
            return False # Not an edge in hypercube sense, but in general graph?
            # Wait, u and v are just vertices. They don't have to be adjacent in the hypercube sense.
            # The Q3 is an abstract graph. It is isomorphic to the 3-cube.
            # My previous logic assumed the vertices of the Q3 in G correspond to vertices in the 4-cube Q4.
            # Is every Q3 in a 4-cube graph isomorphic to a subgraph of Q4?
            # Yes, if we embed Q3 into Q4.
            # BUT, the graph G is arbitrary on 16 vertices. It is not necessarily a subgraph of Q4.
            # The vertices 0..15 are just labels.
            # So I CANNOT assume the Q3 vertices are related by XOR.
            
            # This invalidates the XOR-based optimization!
            # I must use the generic neighbor intersection method, but optimized.
            return None # Fallback to generic?
        
        # RE-EVALUATION:
        # The task is "simple graph on vertices 0..15".
        # "containing no subgraph isomorphic to the 3-cube Q3".
        # This is a general graph. The vertices are not hypercube coordinates.
        # So I MUST use the generic check.
        
        # However, I can optimize the generic check.
        # The current program's check is:
        # Iterate a2, a4 in N(u). Iterate a3, a5 in N(v).
        # This is O(D^4).
        
        # Can I do better?
        # A Q3 has 8 vertices.
        # Let the vertices be x0..x7.
        # x0-x1 is an edge.
        # N(x0) in Q3 = {x1, x2, x3}.
        # N(x1) in Q3 = {x0, x4, x5}.
        # x2, x3 are not connected.
        # x4, x5 are not connected.
        # x2-x4, x2-x5? No.
        # Let's trace Q3 edges again.
        # 000-001, 000-010, 000-100
        # 001-011, 001-101
        # 010-011, 010-110
        # 100-101, 100-110
        # 011-111
        # 101-111
        # 110-111
        
        # Let u=000, v=100.
        # N(u) = {001, 010, 100}. In G, N(u) includes v and others.
        # Let a, b be two neighbors of u (other than v).
        # Let c, d be two neighbors of v (other than u).
        # In Q3, a=001, b=010. c=101, d=110.
        # Edges: a-c (001-101), b-d (010-110).
        # Also a-b is NOT an edge. c-d is NOT an edge.
        # Then there are two more vertices e, f.
        # e=011, f=111.
        # e is connected to a, b, f.
        # f is connected to c, d, e.
        
        # So, to check if (u,v) creates a Q3:
        # 1. Pick a, b from N(u) \ {v}.
        # 2. Pick c, d from N(v) \ {u}.
        # 3. Check if (a,c) in E and (b,d) in E. (Or (a,d) and (b,c)).
        # 4. Check if there exists e in N(a) ∩ N(b) \ {u, a, b, c, d, v}
        #    and f in N(c) ∩ N(d) \ {u, v, a, b, c, d, e}
        #    such that (e,f) in E.
        
        # This is what the current program does.
        # How to speed it up?
        # Use bitsets.
        
        # Let's implement this bitset version.
        
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # Iterate pairs a, b from neigh_u
        # We can iterate a, then b.
        # To avoid duplicates, assume a < b? No, just iterate all.
        
        # Optimization: Precompute common neighbors for pairs?
        # Too much memory.
        
        # Let's just write the tight loop.
        
        a_bits = neigh_u
        while a_bits:
            a_bit = a_bits & (-a_bits)
            a = a_bit.bit_length() - 1
            a_bits -= a_bit
            
            b_bits = neigh_u & ~(1 << a)
            while b_bits:
                b_bit = b_bits & (-b_bits)
                b = b_bit.bit_length() - 1
                b_bits -= b_bit
                
                # Candidates for c: neighbors of v that are also neighbors of a
                c_cand = neigh_v & adj[a]
                # Candidates for d: neighbors of v that are also neighbors of b
                d_cand = neigh_v & adj[b]
                
                # c and d must be distinct from a, b, u, v
                c_cand &= ~(1 << a) & ~(1 << b) & ~(1 << u) & ~(1 << v)
                d_cand &= ~(1 << a) & ~(1 << b) & ~(1 << u) & ~(1 << v)
                
                if not c_cand or not d_cand:
                    continue
                
                # We need to check two cases:
                # Case 1: a connects to c, b connects to d.
                # Case 2: a connects to d, b connects to c.
                
                # Case 1:
                # c is in c_cand, d is in d_cand.
                # Also c != d.
                # And e in N(a) ∩ N(b) \ {u,v,a,b,c,d}
                # And f in N(c) ∩ N(d) \ {u,v,a,b,c,d,e}
                # And (e,f) in E.
                
                # Let's iterate c from c_cand.
                c_bits = c_cand
                while c_bits:
                    c_bit = c_bits & (-c_bits)
                    c = c_bit.bit_length() - 1
                    c_bits -= c_bit
                    
                    # d must be in d_cand, distinct from c
                    d_cand_specific = d_cand & ~(1 << c)
                    if not d_cand_specific:
                        continue
                    
                    d_bits = d_cand_specific
                    while d_bits:
                        d_bit = d_bits & (-d_bits)
                        d = d_bit.bit_length() - 1
                        d_bits -= d_bit
                        
                        # Check if (a,c) and (b,d) are edges?
                        # We picked c from N(v) ∩ N(a), so (a,c) is edge.
                        # We picked d from N(v) ∩ N(b), so (b,d) is edge.
                        # So the cross edges are satisfied.
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        
                        e_cand = (adj[a] & adj[b]) & ~used_mask
                        if not e_cand:
                            continue
                        
                        f_cand_base = (adj[c] & adj[d]) & ~used_mask
                        
                        e_bits = e_cand
                        while e_bits:
                            e_bit = e_bits & (-e_bits)
                            e = e_bit.bit_length() - 1
                            e_bits -= e_bit
                            
                            f_cand = f_cand_base & ~(1 << e)
                            if not f_cand:
                                continue
                            
                            # Check if any f in f_cand is connected to e
                            if adj[e] & f_cand:
                                return True
                                
                # Case 2: a connects to d, b connects to c.
                # This is symmetric.
                # We can just swap the roles of (a,c) and (b,d) in the logic?
                # Actually, the loop structure above checks (a,c) and (b,d).
                # We need to check (a,d) and (b,c).
                
                # Let's add this.
                c_bits = c_cand
                while c_bits:
                    c_bit = c_bits & (-c_bits)
                    c = c_bit.bit_length() - 1
                    c_bits -= c_bit
                    
                    # d must be in d_cand, distinct from c
                    # But now we require (a,d) and (b,c).
                    # So d must be in N(a) ∩ N(v) ? No.
                    # d must be in N(v).
                    # And (a,d) must be edge.
                    # So d must be in N(a) ∩ N(v) = c_cand.
                    # And (b,c) must be edge.
                    # So c must be in N(b) ∩ N(v) = d_cand.
                    
                    # So we iterate c from d_cand (since c must be connected to b).
                    # And d from c_cand (since d must be connected to a).
                    
                    # Wait, I already have c_cand and d_cand.
                    # Let's rename to avoid confusion.
                    # N_a_v = neigh_v & adj[a]
                    # N_b_v = neigh_v & adj[b]
                    
                    # Case 2:
                    # c is in N_b_v (so b-c is edge)
                    # d is in N_a_v (so a-d is edge)
                    # c != d.
                    
                    # Let's re-iterate.
                    c2_bits = d_cand # c must be in N(b)
                    while c2_bits:
                        c2_bit = c2_bits & (-c2_bits)
                        c2 = c2_bit.bit_length() - 1
                        c2_bits -= c2_bit
                        
                        d2_cand = c_cand & ~(1 << c2) # d must be in N(a), distinct from c2
                        if not d2_cand:
                            continue
                        
                        d2_bits = d2_cand
                        while d2_bits:
                            d2_bit = d2_bits & (-d2_bits)
                            d2 = d2_bit.bit_length() - 1
                            d2_bits -= d2_bit
                            
                            used_mask = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c2) | (1 << d2)
                            
                            e_cand = (adj[a] & adj[b]) & ~used_mask
                            if not e_cand:
                                continue
                            
                            f_cand_base = (adj[c2] & adj[d2]) & ~used_mask
                            
                            e_bits = e_cand
                            while e_bits:
                                e_bit = e_bits & (-e_bits)
                                e = e_bit.bit_length() - 1
                                e_bits -= e_bit
                                
                                f_cand = f_cand_base & ~(1 << e)
                                if not f_cand:
                                    continue
                                
                                if adj[e] & f_cand:
                                    return True
                                
        return False

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

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

    # --- Strategy: Greedy Construction + Local Search ---
    
    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_iters=10000):
        """
        1. Try to add any missing edge. If it doesn't create Q3, add it.
        2. If no edge can be added, try to remove an edge.
           If removing an edge allows adding more edges, do it.
           This is a 2-opt style move.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            added_any = False
            
            # Phase 1: Add any valid edge
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                if adj[u] & (1 << v):
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    added_any = True
            
            if added_any:
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                continue
            
            # Phase 2: No edge could be added. Try removing one.
            # Remove an edge that is "most constraining".
            # Heuristic: Remove an edge (u,v) such that many missing edges become valid.
            # Or just random removal to escape local minima.
            
            # Let's try removing a random edge and see if we can add more.
            existing = [(u, v) for u, v in all_edges if adj[u] & (1 << v)]
            if not existing:
                break
                
            u_rem, v_rem = random.choice(existing)
            
            trial = adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            trial_count = current_count - 1
            
            # Try to add edges to trial
            added_in_trial = 0
            for u_add, v_add in all_edges:
                if trial[u_add] & (1 << v_add):
                    continue
                if (u_add, v_add) == (u_rem, v_rem):
                    continue
                if not creates_q3(u_add, v_add, trial):
                    trial[u_add] |= (1 << v_add)
                    trial[v_add] |= (1 << u_add)
                    trial_count += 1
                    added_in_trial += 1
            
            # Accept if we added more than 1 edge (net gain) or equal (to explore)
            if trial_count >= current_count:
                adj = trial
                current_count = trial_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                pass
                
        return best_adj, best_count

    # --- Main ---
    
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple restarts with different initial structures
    
    # 1. Empty graph, greedy add
    adj = [0] * n
    random.shuffle(all_edges)
    greedy_add(adj, all_edges)
    adj, cnt = local_search(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj[:]
        
    # 2. Complete graph, remove edges to break Q3s?
    # This is harder. Let's stick to adding.
    
    # 3. Random initial graph
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        adj = [0] * n
        # Random subset of edges
        p = 0.5
        for u, v in all_edges:
            if random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Clean up: remove edges that create Q3?
        # This is expensive.
        # Instead, run local search which handles it?
        # Local search only adds. It doesn't remove Q3s.
        # If the initial graph has Q3s, it's invalid.
        
        # So we must ensure initial graph is Q3-free.
        # Greedy add from empty is safe.
        # Let's just do more greedy adds with different orders.
        
        random.shuffle(all_edges)
        greedy_add(adj, all_edges)
        adj, cnt = local_search(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Final Polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
