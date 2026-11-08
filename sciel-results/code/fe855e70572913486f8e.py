import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks = len(all_blocks)
    
    # Build conflict graph
    # Two blocks conflict if they share any (i,j,k)
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Degree for ordering
    degrees = [len(conflicts[i]) for i in range(num_blocks)]
    
    # Branch and Bound for Maximum Independent Set
    # We want the set of blocks with no conflicts between them.
    # This is equivalent to finding a clique in the complement graph,
    # but working with independent set in the conflict graph is direct.
    
    best_set = []
    best_size = 0
    
    # Order nodes by degree descending (more constrained first) for pruning
    # Actually, for MIS, high degree nodes are "bad" to include if they block many others?
    # Standard heuristic: sort by degree.
    
    def backtrack(current_set, available_nodes, current_size):
        nonlocal best_set, best_size
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if not available_nodes:
            return
            
        # Pruning: if current_size + len(available_nodes) <= best_size, return
        if current_size + len(available_nodes) <= best_size:
            return
            
        # Pick a node from available_nodes
        # Heuristic: pick node with highest degree in the induced subgraph of available_nodes?
        # Or just the first one.
        # Let's pick the one with highest degree in available_nodes to prune faster?
        # Actually, picking a node that conflicts with many available nodes is good for pruning the "exclude" branch?
        
        # Simple selection: first node
        u = next(iter(available_nodes))
        
        # Branch 1: Include u
        # New available nodes are those in available_nodes that do NOT conflict with u
        new_available = set()
        for v in available_nodes:
            if v not in conflicts[u]:
                new_available.add(v)
                
        backtrack(current_set + [u], new_available, current_size + 1)
        
        # Branch 2: Exclude u
        # Available nodes remain the same, but we remove u from consideration
        # We must remove u from the set to avoid infinite loops
        available_nodes.remove(u)
        backtrack(current_set, available_nodes, current_size)
        
    # Initial call
    # To speed up, we can use a greedy coloring bound or just simple count.
    # For small graphs (num_blocks ~ 3*10*21 = 630? No.
    # i pairs: 3. j pairs: 10. k pairs: 21. Total 3*10*21 = 630 blocks.
    # 630 nodes is too large for simple backtracking without strong bounds.
    
    # Let's use a better heuristic or a different approach.
    # The structure is very regular.
    # We identified that we need to pack squares in 5x7 grid for each i.
    # Let's use a specialized search for the 3 i-pairs.
    
    # Generate all possible 2x2 squares in 5x7 grid
    squares_5x7 = []
    for j1 in range(m - 1):
        for j2 in range(j1 + 1, m):
            if j2 != j1 + 1: continue # Only adjacent? No, any 2.
            # Actually, Strassen block requires specific indices.
            # The block uses j1, j2. They don't have to be adjacent.
            # But the "square" in the j,k plane is defined by {j1,j2} x {k1,k2}.
            # For packing disjointness, we just need the sets to be disjoint.
            
    squares = []
    for j1 in range(m - 1):
        for j2 in range(j1 + 1, m):
            for k1 in range(p - 1):
                for k2 in range(k1 + 1, p):
                    squares.append(((j1, j2, k1, k2), frozenset([(j1,k1), (j1,k2), (j2,k1), (j2,k2)])))
    
    num_squares = len(squares)
    
    # Conflict graph for squares in 5x7
    sq_conflicts = [set() for _ in range(num_squares)]
    for i in range(num_squares):
        for j in range(i + 1, num_squares):
            if squares[i][1] & squares[j][1]:
                sq_conflicts[i].add(j)
                sq_conflicts[j].add(i)
                
    # We need to select sets S_01, S_02, S_12 of squares.
    # S_01 and S_02 must be disjoint (in terms of square indices? No, in terms of j,k footprint).
    # If square A and square B have disjoint j,k footprints, they can coexist in R_0.
    # So S_01 union S_02 must be an independent set in the square conflict graph.
    # Similarly S_01 union S_12 must be independent.
    # S_02 union S_12 must be independent.
    
    # This is still hard.
    # Let's go back to the 630 node graph but use a better algorithm.
    # Tomita's algorithm for Maximum Clique is very fast.
    # Max Independent Set in G is Max Clique in Complement G.
    # Building complement of 630 nodes is 630^2 ~ 400k edges. Manageable.
    
    # Let's implement Tomita's MCQ algorithm for Maximum Clique on the Complement Graph.
    # Nodes: 0..629
    # Edge in Complement G if NOT edge in Conflict G (and i != j).
    
    # Actually, let's just use the greedy with many restarts but smarter.
    # Or use `networkx` if available? No, standard lib.
    
    # Let's try a simple Backtracking with strong pruning on the 630 node graph.
    # 630 is small enough for a good implementation.
    
    # Re-initialize for Backtracking
    best_set = []
    best_size = 0
    nodes = list(range(num_blocks))
    
    # Sort nodes by degree descending
    nodes.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map old index to new index? No, just use the sorted list.
    # We need to check conflicts using original indices.
    
    def backtrack_mis(start_idx, current_set, available_mask):
        # available_mask is a set of original indices available
        nonlocal best_set, best_size
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if not available_mask:
            return
            
        if current_size + len(available_mask) <= best_size:
            return
            
        # Pick a node
        # Heuristic: pick node with highest degree in available_mask
        # To save time, just pick the first one from a pre-sorted list?
        # Let's pick the one with highest degree in the induced subgraph.
        # This is expensive. Let's just pick the first one from the sorted list that is in available_mask.
        
        u = None
        for node in nodes:
            if node in available_mask:
                u = node
                break
                
        if u is None:
            return
            
        # Branch 1: Include u
        new_available = set()
        for v in available_mask:
            if v != u and v not in conflicts[u]:
                new_available.add(v)
                
        backtrack_mis(start_idx, current_set + [u], new_available)
        
        # Branch 2: Exclude u
        available_mask.remove(u)
        backtrack_mis(start_idx, current_set, available_mask)
        
    # Start with all nodes
    initial_available = set(range(num_blocks))
    
    # Run with time limit
    # To avoid deep recursion issues, we rely on the pruning.
    # 630 nodes might be too slow for pure Python backtracking if not pruned well.
    # Let's add a time check inside.
    
    global time_check
    time_check = start_time
    
    def check_time():
        return time.time() > deadline - 5.0
        
    # Modify backtrack to check time
    # Since we can't easily modify the recursive call signature without overhead, 
    # we'll wrap it.
    
    # Alternative: Use a greedy approach but with "swap" improvements.
    # Start with a good greedy packing.
    # Then try to swap one block for two blocks.
    
    # Let's try the Swap approach, it's often more effective for this size.
    
    def greedy_packing():
        used = set()
        current = []
        # Sort by degree ascending? (Less constrained first)
        # Or random?
        # Let's try random restarts.
        for _ in range(1000):
            if time.time() > deadline - 10.0:
                break
            random.shuffle(nodes)
            used = set()
            current = []
            for node in nodes:
                if not (all_blocks[node][1] & used):
                    current.append(node)
                    used |= all_blocks[node][1]
            if len(current) > len(best_set):
                best_set = current[:]
                best_size = len(current)
                
    greedy_packing()
    
    # Local Search: Try to improve
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_packing = best_set[:]
        used = set()
        for node in current_packing:
            used |= all_blocks[node][1]
            
        # Try removing one block and adding two
        for i in range(len(current_packing)):
            if time.time() > deadline - 5.0:
                break
            removed = current_packing[i]
            new_packing = current_packing[:i] + current_packing[i+1:]
            new_used = set()
            for node in new_packing:
                new_used |= all_blocks[node][1]
            
            # Try to add two blocks that are disjoint from new_used and each other
            # Find all blocks disjoint from new_used
            candidates = [b for b in range(num_blocks) if not (all_blocks[b][1] & new_used)]
            
            # Find max independent set of size 2 in candidates
            # Just check all pairs
            added = []
            for c1 in candidates:
                if len(added) == 2: break
                for c2 in candidates:
                    if c1 == c2: continue
                    if c2 in added: continue
                    if c1 in added: continue
                    if not (all_blocks[c1][1] & all_blocks[c2][1]):
                        added.append(c1)
                        added.append(c2)
                        break
                if len(added) == 2:
                    break
                    
            if len(added) == 2:
                new_packing.extend(added)
                if len(new_packing) > len(best_set):
                    best_set = new_packing
                    best_size = len(new_packing)
                    improved = True
                    break # Restart loop with new best
                    
    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for idx in best_set:
        params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
