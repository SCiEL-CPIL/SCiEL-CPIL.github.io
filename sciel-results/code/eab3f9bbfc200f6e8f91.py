import random
import time

def construct():
    n = 20
    # Hard wall-clock guard: evaluator kills at 150s. Return best found by 135s.
    deadline = time.time() + 135
    rng = random.Random(42)

    # Bitmask representation for speed
    # adj[i] is an int bitmask of neighbors of i
    # common[i][j] is the count of common neighbors of i and j (popcount(adj[i] & adj[j]))
    
    def common_neighbors(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')
    
    def init_common(adj):
        common = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                common[i][j] = bin(adj[i] & adj[j]).count('1')
                common[j][i] = common[i][j]
        return common

    def is_valid(adj):
        for i in range(n):
            for j in range(i+1, n):
                if bin(adj[i] & adj[j]).count('1') >= 3:
                    return False
        return True

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    # 1. Construct PG(2,4) polarity graph (sparse) and its complement (dense)
    # GF(4) tables
    gf4_add = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    gf4_mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf4_add_fn(a, b):
        return gf4_add[a][b]
        
    def gf4_mul_fn(a, b):
        return gf4_mul[a][b]

    # Points of PG(2,4)
    points = []
    for y in range(4):
        for z in range(4):
            points.append((1, y, z))
    for z in range(4):
        points.append((0, 1, z))
    points.append((0, 0, 1))
        
    # Polarity matrix M (symplectic in char 2)
    M = [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
         
    def dot(p, q):
        # p^T M q
        mq0 = gf4_add_fn(gf4_mul_fn(M[0][0], q[0]), gf4_add_fn(gf4_mul_fn(M[0][1], q[1]), gf4_mul_fn(M[0][2], q[2])))
        mq1 = gf4_add_fn(gf4_mul_fn(M[1][0], q[0]), gf4_add_fn(gf4_mul_fn(M[1][1], q[1]), gf4_mul_fn(M[1][2], q[2])))
        mq2 = gf4_add_fn(gf4_mul_fn(M[2][0], q[0]), gf4_add_fn(gf4_mul_fn(M[2][1], q[1]), gf4_mul_fn(M[2][2], q[2])))
        res = gf4_add_fn(gf4_mul_fn(p[0], mq0), gf4_add_fn(gf4_mul_fn(p[1], mq1), gf4_mul_fn(p[2], mq2)))
        return res
        
    # Build 21x21 adjacency
    A21 = [[0]*21 for _ in range(21)]
    for i in range(21):
        for j in range(21):
            if i != j:
                if dot(points[i], points[j]) == 0:
                    A21[i][j] = 1
                    A21[j][i] = 1
                    
    # Complement
    B21 = [[0]*21 for _ in range(21)]
    for i in range(21):
        for j in range(21):
            if i != j:
                B21[i][j] = 1 - A21[i][j]
                
    # Find best 20-vertex subgraph of B21
    best_seed_adj = None
    best_seed_edges = -1
    
    for del_idx in range(21):
        adj = [0]*20
        idx_map = []
        k = 0
        for i in range(21):
            if i != del_idx:
                idx_map.append(i)
                k += 1
                
        for i in range(20):
            for j in range(20):
                if B21[idx_map[i]][idx_map[j]]:
                    adj[i] |= (1 << idx_map[j])
                    
        if is_valid(adj):
            ec = edge_count(adj)
            if ec > best_seed_edges:
                best_seed_edges = ec
                best_seed_adj = adj[:]
                
    if best_seed_adj is None:
        # Fallback: Random dense graph
        adj = [0]*20
        for i in range(20):
            for j in range(i+1, 20):
                if rng.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Repair
        common = init_common(adj)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(20):
                for j in range(i+1, 20):
                    if common[i][j] >= 3:
                        # Remove edge from a common neighbor
                        # Find common neighbors
                        mask = adj[i] & adj[j]
                        # Pick one
                        lsb = mask & -mask
                        z = lsb.bit_length() - 1
                        # Remove i-z or j-z?
                        # Removing i-z reduces common[i][j] by 1?
                        # common[i][j] counts k s.t. i-k and j-k.
                        # If we remove i-z, then z is no longer a neighbor of i.
                        # So z is no longer a common neighbor of i and j.
                        # common[i][j] decreases by 1.
                        # common[i][k] for k in N(z) decreases by 1.
                        # common[j][k] for k in N(z) decreases by 1.
                        adj[i] ^= (1 << z)
                        adj[z] ^= (1 << i)
                        
                        # Update common
                        # common[i][j] -= 1
                        common[i][j] -= 1
                        common[j][i] -= 1
                        # For each k in N(z) (excluding i, j):
                        # common[i][k] -= 1 (because i-z removed, z was neighbor of i and k)
                        # common[j][k] -= 1 (because i-z removed? No. j-z is still there. 
                        # Wait. common[j][k] counts m s.t. j-m and k-m.
                        # Removing i-z affects common[i][*] and common[z][*].
                        # It does NOT affect common[j][k] directly unless j==i or j==z.
                        # So we only update common[i][k] and common[z][k].
                        
                        # Neighbors of z:
                        m = adj[z]
                        while m:
                            lsb_k = m & -m
                            k = lsb_k.bit_length() - 1
                            m -= lsb_k
                            if k == i: continue
                            common[i][k] -= 1
                            common[k][i] -= 1
                            common[z][k] -= 1
                            common[k][z] -= 1
                            
                        changed = True
                        break
                if changed: break
        best_seed_adj = adj
        best_seed_edges = edge_count(adj)

    best_adj = best_seed_adj[:]
    best_edges = best_seed_edges
    best_common = init_common(best_adj)
    
    # 2. Simulated Annealing
    # Objective: Maximize edges. Constraint: No common neighbors >= 3.
    # Moves:
    # 1. Add edge (i,j) if valid.
    # 2. Remove edge (i,j) if it exists.
    # 3. Swap: Remove (i,j), Add (k,l).
    
    # We will primarily use "Add if valid" and "Remove if invalid" and "Swap".
    # To escape local optima, we allow invalid states with a penalty?
    # Or we use "Remove one, Add two" moves?
    # Let's stick to valid states only, but use "2-opt" like swaps.
    # Actually, the finding said "Simulated annealing with edge removal/re-addition cycles".
    # This implies we might go through invalid states? Or just random removals followed by re-adds?
    # If we stay in valid states, we can't "remove" an edge that isn't causing a violation unless we want to make space for others.
    # Let's allow invalid states with a cost.
    # Cost = -edges + penalty * violations.
    # If penalty is high, we stay valid. If we lower temp, we accept worse.
    
    # Better: Use valid-only search but with "perturb" (remove random edges) then "repair" (add valid edges).
    # This is effectively SA if we control the number of removals.
    
    # Let's implement a fast local search first to polish the seed.
    def local_search(adj, common, max_iters=10000):
        # Greedy add
        changed = True
        iters = 0
        while changed and iters < max_iters and time.time() < deadline:
            changed = False
            iters += 1
            # Try to add an edge
            # Randomize order to avoid bias
            edges_to_try = [(i, j) for i in range(20) for j in range(i+1, 20) if not (adj[i] & (1 << j))]
            rng.shuffle(edges_to_try)
            for i, j in edges_to_try:
                if time.time() > deadline: return adj, common
                # Check validity of adding (i,j)
                # Need: for all k in N(i) \ {j}, common[k][j] < 2
                # and for all l in N(j) \ {i}, common[i][l] < 2
                
                # Optimize: 
                # If common[i][j] is already 2, adding (i,j) is fine for (i,j) pair?
                # No, adding (i,j) increases common[k][l] for k in N(i), l in N(j).
                # Specifically, for any k in N(i) and l in N(j), common[k][l] increases by 1.
                # We need common[k][l] < 2 for all such pairs.
                
                # Quick check:
                # If there exists k in N(i) such that common[k][j] >= 2, we can't add.
                # Because if common[k][j] >= 2, let l1, l2 be common neighbors of k and j.
                # Then adding (i,j) makes l1, l2, and ... wait.
                # If we add (i,j), then i and j become neighbors.
                # This increases common count for pairs (i, x) where x in N(j) and (j, y) where y in N(i).
                # It also increases common count for (x, y) where x in N(i) and y in N(j) ?
                # No. common[x][y] counts neighbors of x and y.
                # Adding (i,j) means i is now neighbor of j, and j is now neighbor of i.
                # So for any x in N(i), x is neighbor of i. i is neighbor of j.
                # This doesn't directly change common[x][y] unless x or y is i or j.
                # Wait.
                # common[x][y] = |N(x) ∩ N(y)|.
                # If we add edge (i,j):
                # i is added to N(j). j is added to N(i).
                # For pair (i, y) where y in N(j):
                # N(i) gains j. N(y) contains i? No, y is in N(j), so j is in N(y).
                # i is in N(y) iff y is in N(i).
                # Let's just check the condition:
                # Adding (i,j) is valid iff:
                # 1. common[i][j] <= 1 (so that i and j don't have 3 common neighbors after? No, common[i][j] is count BEFORE.
                #    If common[i][j] == 2, then i and j have 2 common neighbors.
                #    Adding (i,j) doesn't change the set of common neighbors of i and j.
                #    So common[i][j] remains 2. This is valid (<3).
                #    So common[i][j] can be anything < 3.
                # 2. For all x in N(i) \ {j}, common[x][j] < 2.
                #    Why? If common[x][j] == 2, let u, v be common neighbors of x and j.
                #    Then x, j, u, v form a K_{2,3}? No.
                #    If we add (i,j), then i is a new neighbor of j.
                #    Does this create a K_{2,3}?
                #    Consider vertices x and j.
                #    N(x) contains i (since i in N(x)? No, x in N(i) implies i in N(x)).
                #    N(j) contains i (new edge).
                #    So i is a common neighbor of x and j.
                #    Previously, u and v were common neighbors of x and j.
                #    So now x and j have common neighbors {i, u, v}.
                #    That's 3 common neighbors! Violation.
                #    So we MUST have common[x][j] < 2 for all x in N(i).
                # 3. Similarly, for all y in N(j) \ {i}, common[i][y] < 2.
                
                # Check 2:
                ok = True
                m = adj[i]
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    m -= lsb
                    if x == j: continue
                    if common[x][j] >= 2:
                        ok = False
                        break
                if ok:
                    m = adj[j]
                    while m:
                        lsb = m & -m
                        y = lsb.bit_length() - 1
                        m -= lsb
                        if y == i: continue
                        if common[i][y] >= 2:
                            ok = False
                            break
                
                if ok:
                    # Add edge
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
                    # Update common
                    # common[i][j] unchanged (intersection of N(i) and N(j) doesn't change?
                    # N(i) gains j. N(j) gains i.
                    # N(i) ∩ N(j) = (N(i) \ {j}) ∩ (N(j) \ {i})?
                    # j is in N(j) but not in N(i) (before). i is in N(i) but not in N(j).
                    # So j is not in N(i) ∩ N(j). i is not in N(i) ∩ N(j).
                    # So common[i][j] is unchanged.
                    
                    # For x in N(i) \ {j}:
                    # common[x][j] increases by 1 (because i is now in N(j) and i is in N(x)).
                    # common[x][i] increases by 1 (because j is now in N(i) and j is in N(x)? No.
                    # common[x][i] = |N(x) ∩ N(i)|.
                    # N(i) gains j.
                    # Is j in N(x)? x in N(i) means i in N(x).
                    # j in N(x) means x in N(j).
                    # So if x is also in N(j), then j is in N(x).
                    # So common[x][i] increases by 1 IF x is in N(j).
                    # i.e., if x is a common neighbor of i and j.
                    
                    # Let's update systematically.
                    # New edge (i,j).
                    # For all x in N(i) \ {j}:
                    #   common[x][j] += 1
                    #   if x in N(j): common[x][i] += 1  (since j in N(x) and j in N(i))
                    #   else: common[x][i] unchanged?
                    #   Wait. common[x][i] = |N(x) ∩ N(i)|.
                    #   N(i) gains j.
                    #   So if j in N(x), then j is added to the intersection.
                    #   j in N(x) <=> x in N(j).
                    #   So yes, if x in N(j), common[x][i] += 1.
                    
                    # For all y in N(j) \ {i}:
                    #   common[i][y] += 1
                    #   if y in N(i): common[y][j] += 1
                    #   (Symmetric to above)
                    
                    # For x in N(i) \ {j} and y in N(j) \ {i}:
                    #   Do common[x][y] change?
                    #   N(x) and N(y) don't change directly.
                    #   But wait.
                    #   common[x][y] = |N(x) ∩ N(y)|.
                    #   N(x) contains i. N(y) contains j.
                    #   i and j are now connected.
                    #   Does this add a common neighbor?
                    #   No, i is not in N(y) (unless y=i). j is not in N(x) (unless x=j).
                    #   So common[x][y] is unchanged.
                    
                    # So we only need to update:
                    # 1. common[x][j] += 1 for x in N(i) \ {j}
                    # 2. common[i][y] += 1 for y in N(j) \ {i}
                    # 3. common[x][i] += 1 for x in N(i) ∩ N(j)
                    # 4. common[y][j] += 1 for y in N(i) ∩ N(j)
                    
                    # Note: 3 and 4 are for the intersection.
                    # Let I = N(i) ∩ N(j).
                    # For x in I:
                    #   common[x][i] += 1
                    #   common[x][j] += 1
                    # For x in N(i) \ {j} \ I:
                    #   common[x][j] += 1
                    # For y in N(j) \ {i} \ I:
                    #   common[i][y] += 1
                    
                    # Let's implement this.
                    I = adj[i] & adj[j]
                    # x in N(i) \ {j}
                    m = adj[i] & ~(1 << j)
                    while m:
                        lsb = m & -m
                        x = lsb.bit_length() - 1
                        m -= lsb
                        common[x][j] += 1
                        common[j][x] += 1
                        if (I & (1 << x)):
                            common[x][i] += 1
                            common[i][x] += 1
                    # y in N(j) \ {i}
                    m = adj[j] & ~(1 << i)
                    while m:
                        lsb = m & -m
                        y = lsb.bit_length() - 1
                        m -= lsb
                        common[i][y] += 1
                        common[y][i] += 1
                        # Note: if y in I, we already updated common[y][j]?
                        # No, the first loop updated common[x][j] for x in N(i).
                        # y is in N(j). If y is in I, then y is in N(i).
                        # So common[y][j] was updated in the first loop?
                        # First loop: x in N(i). Updates common[x][j].
                        # So if y in I, common[y][j] was incremented.
                        # We don't need to do it again.
                        # But we DO need to update common[y][j] for y in N(j) \ I?
                        # No, common[y][j] is updated in the first loop ONLY if y is in N(i).
                        # If y is NOT in N(i), then y is not in I.
                        # Does common[y][j] change?
                        # common[y][j] = |N(y) ∩ N(j)|.
                        # N(j) gains i.
                        # Is i in N(y)?
                        # i in N(y) <=> y in N(i).
                        # If y NOT in N(i), then i NOT in N(y).
                        # So i is not added to the intersection.
                        # So common[y][j] is UNCHANGED for y not in N(i).
                        
                    changed = True
                    break # Restart loop to check other edges with updated common
        
        return adj, common

    # Run initial local search
    best_adj, best_common = local_search(best_adj, best_common, max_iters=1000)
    best_edges = edge_count(best_adj)
    
    # Simulated Annealing
    # We will use a "remove and re-add" strategy.
    # Remove k random edges, then run local search to re-add as many as possible.
    # If the result is better, keep it.
    
    current_adj = best_adj[:]
    current_common = best_common
    current_edges = best_edges
    
    T = 50.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline:
        # Perturb
        perturb_adj = current_adj[:]
        # Remove 2-5 random edges
        num_removals = rng.randint(2, 5)
        for _ in range(num_removals):
            # Pick random existing edge
            # Get list of edges
            edges_list = []
            for i in range(20):
                m = perturb_adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    m -= lsb
                    if i < j:
                        edges_list.append((i, j))
            if not edges_list: break
            i, j = rng.choice(edges_list)
            perturb_adj[i] ^= (1 << j)
            perturb_adj[j] ^= (1 << i)
            
        # Recompute common for perturbed graph
        perturb_common = init_common(perturb_adj)
        
        # Local search
        perturb_adj, perturb_common = local_search(perturb_adj, perturb_common, max_iters=500)
        perturb_edges = edge_count(perturb_adj)
        
        # Acceptance
        delta = perturb_edges - current_edges
        if delta > 0:
            current_adj = perturb_adj
            current_common = perturb_common
            current_edges = perturb_edges
            if current_edges > best_edges:
                best_adj = current_adj[:]
                best_common = current_common
                best_edges = current_edges
        else:
            # Accept with probability exp(delta / T)
            if rng.random() < pow(2.71828, delta / T):
                current_adj = perturb_adj
                current_common = perturb_common
                current_edges = perturb_edges
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Reset current to best? Or just keep going?
            # Let's reset to best to explore new directions
            current_adj = best_adj[:]
            current_common = best_common
            current_edges = best_edges
            
    # Final polish
    best_adj, best_common = local_search(best_adj, best_common, max_iters=10000)
    best_edges = edge_count(best_adj)
    
    # Convert to edge list
    edges = []
    for i in range(20):
        for j in range(i+1, 20):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges

