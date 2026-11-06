<<<<<<< SEARCH
    # Try to construct n=33 graph
    def try_n33(time_budget):
        start_time = time.time()
        deadline = start_time + time_budget
        
        n_base, adj_base = get_base_32()
        
        # Try many random configurations for vertex 32
        # Vertex 32 connects to some subset of 0..31
        # To avoid C5: 32's neighbors must form an independent set in the base graph
        # (otherwise two neighbors of 32 that are adjacent create a triangle, 
        #  but that's fine - C5 needs a 5-cycle. Actually, if 32 connects to 
        #  adjacent vertices u,v in a K4, then 32-u-v-32 is a triangle, not C5.
        #  C5 would need: 32-a-b-c-d-32 where all are distinct and edges exist.
        #  Since base has no edges between K4s, any path from 32 goes into one K4,
        #  stays in it (K4 is complete, so max path length in K4 is 3: a-b-c),
        #  then back to 32. So 32-a-b-c-32 is only 4 vertices + 32 = 5? 
        #  No: 32-a, a-b, b-c, c-32. That's a 4-cycle if c is also neighbor of 32.
        #  For C5: 32-a-b-c-d-32. Need d neighbor of 32 and c. 
        #  In a K4: a,b,c,d all mutually adjacent. So 32-a-b-c-d-32 IS a C5
        #  if 32 connects to all 4 vertices of a K4!
        #  So: 32 must NOT connect to all 4 vertices of any K4.
        #  If 32 connects to 3 or fewer vertices in each K4, can we still get C5?
        #  32-a-b-c-d-32: need 4 distinct neighbors of 32 in the same K4.
        #  So if 32 connects to at most 3 per K4, no C5 through that K4.
        #  Can C5 go through multiple K4s? No edges between K4s, so no.
        #  Therefore: 32 must connect to at most 3 vertices in each K4 block.
        
        # For alpha: base has alpha=8 (one per K4). 
        # Adding 32: if 32 is isolated, alpha=9. Bad.
        # 32 must connect to at least one vertex in each "group" to reduce alpha.
        # Actually: alpha of new graph. If 32 connects to vertex in block i, 
        # then in an IS, we can't have both 32 and that vertex.
        # Max IS: either include 32 (then can't include its neighbors, 
        #  can include 1 from each other block, and from blocks where 32 has neighbors,
        #  can include 1 from non-neighbor vertices in that block) 
        #  or exclude 32 (then alpha = 8 from base, but wait, base alpha is 8,
        #  and we might be able to do better... no, base IS is 8, one per K4).
        # 
        # Let's think: if 32 connects to vertices in blocks S (subset of {0..7}),
        # then:
        # - If we include 32: for blocks in S, we can pick 1 from non-neighbors of 32 
        #   in that block (up to 4 - deg_32_in_block vertices available, but still at most 1 
        #   from the K4 minus neighbors... actually in K4, if 32 connects to d vertices,
        #   the remaining 4-d vertices form an independent set in the K4? No! In K4 all 
        #   pairs are adjacent. So the remaining 4-d vertices are still mutually adjacent.
        #   We can pick at most 1 from them.
        #   For blocks not in S: pick 1 from the 4 vertices.
        #   Total: 1 (for 32) + |S| (one from each block in S, from non-neighbors) 
        #           + (8-|S|) (one from each block not in S) = 1 + 8 = 9. 
        #   Wait, that's always 9? No: for blocks in S, if 32 connects to all 4, 
        #   we can pick 0 from that block. But we said at most 3 per block.
        #   If 32 connects to d vertices in block i (d <= 3), we can still pick 1 
        #   from the remaining 4-d >= 1 vertices. So we get 1 from that block.
        #   So including 32 gives: 1 + sum over blocks of 1 = 1 + 8 = 9. Always 9!
        #   That means if we include 32, alpha >= 9 always (as long as each block 
        #   has at least 1 non-neighbor of 32, which is true since d<=3).
        #
        # - If we exclude 32: alpha = alpha(base) = 8.
        #
        # So alpha(new graph) = max(9, 8) = 9 if 32 has any configuration with d<=3 per block.
        # This means n=33 with this structure ALWAYS has alpha >= 9!
        # 
        # Unless... we also add edges between blocks? But that might create C5.
        # Or modify the base structure.
        
        # Alternative: modify the base. Instead of 8 K4s, use a different structure.
        # What if we use 7 K4s + 1 K5? No, K5 has C5.
        # What about Turan graph T(32,4)? T(n,r) is complete r-partite with balanced parts.
        # T(32,4): 4 parts of 8, all edges between parts. Alpha = 8 (one part).
        # C5-free? T(n,r) is C5-free iff... actually complete multipartite graphs 
        # contain C5 if there are at least 5 parts or if a part has size >= 2 and 
        # there are at least 3 parts. Wait: in T(n,4), take one vertex from each of 
        # 4 parts: that's K4, no C5. To get C5, need 5 vertices. But only 4 parts.
        # A C5 in complete multipartite: vertices must be in distinct parts (since 
        # same part = no edge). So need 5 distinct parts for C5. With 4 parts, no C5!
        # So T(32,4) is C5-free and alpha=8. Same as 8 K4s essentially.
        
        # For n=33: T(33,4) has parts of sizes 9,8,8,8. Alpha=9. Bad.
        # T(33,5): parts ~7 each. Alpha=7. C5-free? Need 5 parts for C5, have 5.
        #   Can we form C5? Pick one from each of 5 parts: that's K5, which contains C5!
        #   So T(33,5) has C5. Bad.
        
        # What about a different approach for n=33?
        # Start with T(32,4) and add vertex 33 with careful connections.
        # Or: use the fact that we need alpha < 9, so alpha <= 8.
        # By Turan's theorem, if alpha <= 8, then the complement has no K9.
        # The complement of our graph must be K9-free.
        
        # Let me try a completely different strategy:
        # Use simulated annealing on n=33 with a smarter initialization.
        # Start from T(32,4) + vertex 32, and optimize.
        
        # Actually, let me just try random n=33 graphs with SA, but with 
        # much faster cost evaluation.
        
        def cost_fast(n, adj_mask):
            """Return 0 if valid, else penalty."""
            if has_c5_fast(n, adj_mask):
                return 100
            if has_is_k_fast(n, adj_mask, 9):
                return 10
            return 0
        
        # Initialize from T(32,4) + random vertex 32
        n = 33
        adj_mask = [0] * n
        
        # T(32,4): 4 parts of 8 vertices each
        part_sizes = [8, 8, 8, 8]
        vertex = 0
        parts = []
        for size in part_sizes:
            part = list(range(vertex, vertex + size))
            parts.append(part)
            vertex += size
        
        # Add all edges between different parts
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
        
        # Vertex 32: connect to random subset, at most 3 per part to avoid C5
        for part in parts:
            # Choose 0-3 random vertices from this part to connect to
            k = random.randint(0, 3)
            chosen = random.sample(part, k)
            for v in chosen:
                adj_mask[32] |= (1 << v)
                adj_mask[v] |= (1 << 32)
        
        # Simulated Annealing with smarter moves
        T = 1.0
        cooling = 0.9995
        best_adj = adj_mask[:]
        best_cost = cost_fast(n, best_adj)
        current_cost = best_cost
        
        while time.time() < deadline:
            # Smarter move: try to fix violations
            if current_cost > 0:
                # If C5 exists, try removing an edge from a detected C5
                # If IS exists, try adding an edge within the IS
                if current_cost >= 100:
                    # C5 violation: find a C5 and remove a random edge from it
                    # For simplicity, just flip a random edge but bias toward removal
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    mask_u = 1 << u
                    mask_v = 1 << v
                    if adj_mask[u] & mask_v:
                        # Remove edge
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        # Add edge (less likely when C5 exists)
                        if random.random() < 0.3:
                            adj_mask[u] |= mask_v
                            adj_mask[v] |= mask_u
                        else:
                            # Try another edge to remove
                            u2 = random.randint(0, n - 1)
                            v2 = random.randint(0, n - 1)
                            if u2 == v2:
                                continue
                            mu2 = 1 << u2
                            mv2 = 1 << v2
                            if adj_mask[u2] & mv2:
                                adj_mask[u2] &= ~mv2
                                adj_mask[v2] &= ~mu2
                else:
                    # IS violation: try adding an edge
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                    mask_u = 1 << u
                    mask_v = 1 << v
                    if not (adj_mask[u] & mask_v):
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
                    else:
                        # Remove edge
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
            else:
                # Valid: explore neighborhood
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                mask_u = 1 << u
                mask_v = 1 << v
                if adj_mask[u] & mask_v:
                    adj_mask[u] &= ~mask_v
                    adj_mask[v] &= ~mask_u
                else:
                    adj_mask[u] |= mask_v
                    adj_mask[v] |= mask_u
            
            new_cost = cost_fast(n, adj_mask)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj_mask[:]
            else:
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    # Revert
                    if adj_mask[u] & mask_v:
                        adj_mask[u] &= ~mask_v
                        adj_mask[v] &= ~mask_u
                    else:
                        adj_mask[u] |= mask_v
                        adj_mask[v] |= mask_u
            
            T *= cooling
            
            if best_cost == 0:
                return best_adj
        
        if best_cost == 0:
            return best_adj
        return None
=======
    def try_n33(time_budget):
        """Try to find n=33 graph using T(32,4) base + vertex 32.
        
        Key insight: For T(32,4) with parts P0,P1,P2,P3 of size 8:
        - Base is C5-free (only 4 parts, C5 needs 5 distinct parts in complete multipartite)
        - alpha(base) = 8 (one entire part)
        - Adding vertex 32: to keep alpha <= 8, vertex 32 must connect to at least
          one vertex in EACH part. Otherwise, pick 32 + one vertex from each part 
          not connected to 32 + remaining parts = could get 9.
          Actually: if 32 doesn't connect to part Pi, then IS can include 32 + 
          all of Pi (8 vertices) = 9. Bad.
          If 32 connects to at least 1 in each part: IS including 32 can include 
          at most 7 from each part (excluding neighbors of 32 in that part). 
          But wait, parts are independent sets, so from part Pi we can include 
          all non-neighbors of 32 in Pi. If 32 connects to d_i in Pi, we can 
          include 8 - d_i from Pi. Total IS with 32: 1 + sum(8 - d_i).
          For this to be <= 8: sum(8 - d_i) <= 7, i.e., sum(d_i) >= 25.
          With 4 parts of 8, sum(d_i) >= 25 means avg >= 6.25.
          So 32 must connect to many vertices: e.g., 7,7,7,4 or 8,7,7,3 etc.
        
        C5 constraint with vertex 32: 
        A C5 through 32: 32-a-b-c-d-32 where a,d in N(32), and a-b-c-d is a path.
        In T(32,4), edges only exist between different parts.
        So a,b,c,d must be in parts forming a path: a in Pi, b in Pj (j!=i), 
        c in Pk (k!=j), d in Pl (l!=k), and also d-a edge (l!=i).
        For C5: need 4 distinct vertices in 4 distinct parts (or some in same part? 
        No, same part = no edge). So a,b,c,d in 4 distinct parts.
        Parts: Pi, Pj, Pk, Pl all distinct. But there are only 4 parts!
        So a,b,c,d are one from each part. And edges: a-b (Pi-Pj), b-c (Pj-Pk), 
        c-d (Pk-Pl), d-a (Pl-Pi). This is a 4-cycle in the complete 4-partite graph, 
        plus edges to 32. So 32-a-b-c-d-32 is a C5 iff 32 connects to both a and d,
        and a,b,c,d are in 4 distinct parts with the cycle edges.
        
        So to avoid C5: for any 4-tuple (a,b,c,d) with a in Pi, b in Pj, c in Pk, 
        d in Pl (all distinct parts), if 32 connects to both a and d, then we need 
        to ensure NOT all of a-b, b-c, c-d exist. But in T(32,4), ALL inter-part 
        edges exist! So a-b, b-c, c-d always exist.
        
        Therefore: if 32 connects to vertices in all 4 parts, there IS a C5!
        Specifically, pick any a in N(32) ∩ Pi and d in N(32) ∩ Pl (i != l), 
        and any b in Pj, c in Pk (j,k != i,l, j != k). Then 32-a-b-c-d-32 is a C5.
        
        This means: with T(32,4) base, it's IMPOSSIBLE to add vertex 32 connecting 
        to all 4 parts without creating a C5. And connecting to < 4 parts gives alpha >= 9.
        
        So T(32,4) + vertex 32 cannot work. Need a different base or different approach.
        
        Alternative: modify the base. What if we remove some inter-part edges to 
        prevent the C5, while keeping alpha <= 8?
        
        Or: use a different base entirely. What about a graph that's not complete 
        multipartite?
        
        Let me try: start with T(32,4) but remove some edges to allow vertex 32 
        to connect to all 4 parts without C5.
        
        Actually, let me try a different structural approach:
        Use 4 groups of 8 vertices. Within each group, make it a clique (K8). 
        Between groups, add edges carefully.
        
        Wait, that's the complement of T(32,4). The complement of T(32,4) is 4 disjoint K8s.
        That has alpha = 32 (any vertex from different groups is non-adjacent... no wait,
        in 4 disjoint K8s, alpha = 4 (one from each K8). And it's C5-free (each component 
        is a clique, no cycles of length > 2 within a component, no inter-component edges).
        But alpha = 4 < 9, and n = 32. Same as before.
        
        For n=33: 4 K8s + 1 vertex. The extra vertex connects to some of the 32.
        If it connects to at least 1 in each K8, then IS with extra vertex: 
        1 + (1 from each K8 minus neighbors) = 1 + 4 = 5 (if connects to 1 per K8, 
        can still pick 1 from each K8's remaining 7). So alpha <= 5. Good.
        C5: extra vertex + 4 others. In K8, max path is 2 (a-b-c, all in K8). 
        So extra-a-b-c-extra is a 4-cycle (if extra connects to a and c, and a-b, b-c in K8).
        For C5: extra-a-b-c-d-extra. Need d in same K8 as a (for d-a edge) or different.
        If a,d in same K8: a-b-c-d is a path in K8 (all edges exist), so extra-a-b-c-d-extra 
        is a C5 if extra connects to a and d. So if extra connects to 2+ vertices in any K8, 
        we get a C5 through that K8!
        If extra connects to exactly 1 per K8: no C5 through any K8 (need 2 in same K8).
        Can C5 go through multiple K8s? No inter-K8 edges. So no.
        So: 4 K8s + vertex connecting to exactly 1 per K8 = C5-free, alpha <= 5. n=33!
        
        Let me verify: 4 K8s = 32 vertices. Add vertex 32 connecting to 1 vertex in each K8.
        Total n = 33.
        C5-free: As argued, yes.
        Alpha: Max IS. If include vertex 32: can't include its 4 neighbors. 
        From each K8, can include at most 1 (since K8 is a clique). 
        So IS with 32: 1 + 4 = 5 (one from each K8, not the neighbor of 32).
        If exclude 32: IS = 4 (one from each K8).
        So alpha = 5 < 9. 
        """
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Construct: 4 K8s + vertex 32 connecting to 1 per K8
        n = 33
        adj_mask = [0] * n
        
        # 4 K8s: vertices 0-7, 8-15, 16-23, 24-31
        for g in range(4):
            base = g * 8
            for i in range(8):
                for j in range(i + 1, 8):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        
        # Vertex 32 connects to 1 vertex in each K8
        # Try different choices
        for trial in range(200):
            if time.time() > deadline:
                break
            # Random choice of which vertex in each K8
            for g in range(4):
                v = random.randint(0, 7)
                adj_mask[32] |= (1 << (g * 8 + v))
                adj_mask[g * 8 + v] |= (1 << 32)
            
            if is_valid(n, adj_mask):
                return adj_mask[:]
        
        # If random didn't work, try systematic
        # Actually, by the argument above, ANY choice of 1 per K8 should work.
        # Let me just use a fixed choice.
        adj_mask = [0] * n
        for g in range(4):
            base = g * 8
            for i in range(8):
                for j in range(i + 1, 8):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        for g in range(4):
            v = g * 8  # Connect to first vertex of each K8
            adj_mask[32] |= (1 << v)
            adj_mask[v] |= (1 << 32)
        
        if is_valid(n, adj_mask):
            return adj_mask[:]
        
        return None
>>>>>>> REPLACE

