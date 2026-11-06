import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # 35 second margin
    
    best_edges = None
    best_count = 0
    
    # Precompute all triples for K3,3 checking
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Run multiple restarts with different strategies
    for seed in range(500):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Try different initial densities
        p = 0.7 + 0.2 * (seed % 5) / 4.0  # 0.7, 0.75, 0.8, 0.85, 0.9
        
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Precompute neighbor bitsets
        neigh = [0]*n
        for v in range(n):
            mask = 0
            for u in range(n):
                if adj[v][u]:
                    mask |= (1 << u)
            neigh[v] = mask
        
        def remove_edge(u, v):
            adj[u][v] = 0
            adj[v][u] = 0
            neigh[u] &= ~(1 << v)
            neigh[v] &= ~(1 << u)

        def add_edge(u, v):
            adj[u][v] = 1
            adj[v][u] = 1
            neigh[u] |= (1 << v)
            neigh[v] |= (1 << u)
        
        # Fast K3,3 check: count violations per edge using bitsets
        def compute_edge_badness():
            """Compute how many K3,3 violations each edge participates in."""
            badness = [[0]*n for _ in range(n)]
            found = False
            for a1, a2, a3 in all_triples:
                common = neigh[a1] & neigh[a2] & neigh[a3]
                count = common.bit_count()
                if count >= 3:
                    found = True
                    # Each edge (a, c) where a in {a1,a2,a3} and c in common
                    # participates in C(count-1, 2) K3,3s for this triple
                    add_val = (count - 1) * (count - 2) // 2
                    for a in (a1, a2, a3):
                        c_mask = common
                        while c_mask:
                            lsb = c_mask & -c_mask
                            c = lsb.bit_length() - 1
                            badness[a][c] += add_val
                            badness[c][a] += add_val
                            c_mask ^= lsb
            return badness, found
        
        # Greedy removal of K3,3 violations
        while time.time() < deadline:
            badness, found = compute_edge_badness()
            if not found:
                break
                
            # Find edge with max badness
            max_bad = -1
            max_u, max_v = -1, -1
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] and badness[i][j] > max_bad:
                        max_bad = badness[i][j]
                        max_u, max_v = i, j
                        
            if max_u == -1:
                break
                
            remove_edge(max_u, max_v)
        
        # Fast edge addition: check if adding (u,v) creates K3,3
        # When adding (u,v), new K3,3s can only involve u and v together.
        # A K3,3 has u in one part, v in the other.
        # Part A = {u, a1, a2}, Part B = {v, b1, b2}
        # a1,a2 in N(v), b1,b2 in N(u)
        # Need a_i connected to b_j for all i,j
        def is_add_valid(u, v):
            """Check if adding edge (u,v) creates a K3,3."""
            Nv_mask = neigh[v]
            Nu_mask = neigh[u]
            
            # Need at least 2 neighbors in each
            if Nv_mask.bit_count() < 2 or Nu_mask.bit_count() < 2:
                return True
            
            # Extract neighbor lists
            Nv = []
            m = Nv_mask
            while m:
                lsb = m & -m
                Nv.append(lsb.bit_length() - 1)
                m ^= lsb
            
            Nu = []
            m = Nu_mask
            while m:
                lsb = m & -m
                Nu.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # For each pair (a1, a2) in Nv and (b1, b2) in Nu,
            # check if they form a K2,2 (all 4 edges present)
            # and all 4 vertices are distinct
            for i in range(len(Nv)):
                for j in range(i+1, len(Nv)):
                    a1, a2 = Nv[i], Nv[j]
                    # Check edges from a1 and a2 to all of Nu
                    # a1 must connect to at least 2 in Nu, a2 must connect to at least 2 in Nu
                    # and the sets must overlap in at least 2
                    a1_to_Nu = 0
                    a2_to_Nu = 0
                    for b in Nu:
                        if adj[a1][b]:
                            a1_to_Nu |= (1 << b)
                        if adj[a2][b]:
                            a2_to_Nu |= (1 << b)
                    common_Nu = a1_to_Nu & a2_to_Nu
                    if common_Nu.bit_count() >= 2:
                        # Need to check distinctness: a1, a2 not in Nu, b1, b2 not in Nv
                        # Since a1, a2 are in Nv and b1, b2 are in Nu,
                        # and u,v are not in their own neighbor sets,
                        # we just need a1 != b1, a1 != b2, a2 != b1, a2 != b2
                        # But a1, a2 are neighbors of v, and b1, b2 are neighbors of u.
                        # Could a1 == b1? Yes, if a1 is neighbor of both u and v.
                        # But if a1 == b1, then we have only 3 distinct vertices in the 4 positions,
                        # which doesn't form K3,3. So we need 4 distinct vertices.
                        # Let's just check the 4 edges and distinctness properly.
                        # Get the common neighbors
                        common_list = []
                        m2 = common_Nu
                        while m2:
                            lsb2 = m2 & -m2
                            common_list.append(lsb2.bit_length() - 1)
                            m2 ^= lsb2
                        # Pick any 2 from common_list as b1, b2
                        if len(common_list) >= 2:
                            b1, b2 = common_list[0], common_list[1]
                            # Check distinctness
                            if a1 != b1 and a1 != b2 and a2 != b1 and a2 != b2:
                                # All 4 edges exist (by construction)
                                return False
            return True
        
        # Greedy addition of edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v]:
                        continue
                    if is_add_valid(u, v):
                        add_edge(u, v)
                        changed = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
        
        # Count edges
        count = 0
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    count += 1
                    edges.append([i, j])
        
        if count > best_count:
            best_count = count
            best_edges = edges

    return best_edges if best_edges else []
