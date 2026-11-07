import time
import random
import copy

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix for O(1) checks
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    all_edges = list(edge_set)
    num_edges = len(all_edges)
    
    # Map edge to index for faster set operations
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Initialize with a greedy solution
    # Each edge is its own biclique: [[u], [v]]
    # This is a valid partition but has N bicliques.
    # We will try to merge and move to reduce count.
    
    # Representation: list of bicliques. Each biclique is (L_set, R_set, edge_indices_set)
    # To ensure validity, we maintain that edge_indices_set exactly matches { (u,v) for u in L_set, v in R_set }
    
    # Greedy initialization: 
    # Start with empty partition, add edges one by one, expanding bicliques.
    # Or simpler: start with each edge as a biclique, then try to merge.
    # Merging is harder to validate quickly if we don't track edges.
    
    # Let's use a representation: 
    # partition = list of [L_list, R_list]
    # We need to check validity: 
    # 1. Disjointness of cross-edge sets.
    # 2. Coverage of all edges.
    # 3. Biclique property (all cross pairs are edges).
    
    # Strategy: 
    # 1. Generate initial solution using greedy maximal biclique expansion (like current but faster).
    # 2. Use Simulated Annealing with moves:
    #    - Move 1: Pick a random edge from a biclique, try to move it to another biclique (if it fits).
    #      This might split a biclique or require creating a new one? 
    #      Actually, moving an edge (u,v) from B1 to B2 is only valid if (u,v) can be added to B2 without violating biclique property, AND removing it from B1 doesn't break B1's biclique property (it won't, as biclique property is monotone with respect to removal of edges? No, removing an edge from a biclique leaves a valid biclique).
    #      Wait, if we move edge (u,v) from B1 to B2:
    #      - B1 loses edge (u,v). B1 remains a valid biclique.
    #      - B2 gains edge (u,v). B2 must remain a valid biclique. This means u and v must be compatible with existing L/R of B2.
    #      - But we also need to ensure that the edge (u,v) is NOT already covered by another biclique (it isn't, as partition is disjoint).
    #      - So the move is: Remove (u,v) from B1. Add (u,v) to B2.
    #      - Check if B2 + {(u,v)} is a valid biclique.
    #      - If yes, accept with probability.
    #      - If B1 becomes empty, remove B1.
    #      - This reduces the number of bicliques if B1 was empty, or keeps it same. It doesn't directly reduce count unless we merge empty ones.
    #      - Actually, this move doesn't change the number of bicliques unless B1 becomes empty.
    #      - To reduce count, we need to MERGE bicliques.
    
    #    - Move 2: Merge two bicliques B1, B2 into B3.
    #      - Valid if L1 U L2 and R1 U R2 form a biclique.
    #      - i.e., all pairs in (L1 U L2) x (R1 U R2) are edges.
    #      - This reduces count by 1.
    
    #    - Move 3: Split a biclique? Probably not useful for minimization.
    
    # Let's implement Greedy Init + Local Search (Merge & Move).
    
    def is_biclique(L, R):
        if not L or not R:
            return False
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    def get_edges(L, R):
        return set((min(u,v), max(u,v)) for u in L for v in R)

    # Greedy Init
    remaining_edges = set(all_edges)
    partition = []
    
    while remaining_edges:
        if time.time() > deadline:
            break
        # Pick random edge
        e = random.choice(list(remaining_edges))
        u, v = e
        L = {u}
        R = {v}
        # Expand
        changed = True
        while changed:
            changed = False
            # Try add to L
            for x in range(n):
                if x in L: continue
                if all(adj[x][y] for y in R):
                    # Check if all (x,y) in remaining_edges? 
                    # No, we just need the biclique to be valid in the GRAPH.
                    # The edges it covers must be a subset of remaining_edges.
                    # So we only add x if all (x,y) for y in R are in remaining_edges.
                    if all((min(x,y), max(x,y)) in remaining_edges for y in R):
                        L.add(x)
                        changed = True
            # Try add to R
            for y in range(n):
                if y in R: continue
                if all(adj[x][y] for x in L):
                    if all((min(x,y), max(x,y)) in remaining_edges for x in L):
                        R.add(y)
                        changed = True
        
        # Remove covered edges
        covered = get_edges(L, R)
        remaining_edges -= covered
        partition.append([list(L), list(R)])
        
    # If we timed out during init, finish quickly
    if remaining_edges:
        for u, v in remaining_edges:
            partition.append([[u], [v]])
            
    best_partition = copy.deepcopy(partition)
    best_count = len(best_partition)
    
    # Simulated Annealing / Local Search
    T = 100.0
    T_min = 0.01
    cooling = 0.99
    
    # Precompute edge sets for each biclique for fast update?
    # Let's represent partition as list of (L_set, R_set)
    # And maintain a global edge_to_biclique map?
    # Since we need to check validity of moves, we can just recompute affected parts.
    
    # To speed up, let's keep partition as list of [L_list, R_list]
    # And a helper to get edge set of a biclique.
    
    # Move: Merge B_i, B_j
    def try_merge(i, j):
        L1, R1 = partition[i]
        L2, R2 = partition[j]
        L_new = set(L1) | set(L2)
        R_new = set(R1) | set(R2)
        if is_biclique(L_new, R_new):
            # Valid merge
            new_b = [list(L_new), list(R_new)]
            # Remove i, j, add new_b
            # Order doesn't matter for validity, but for index stability, let's just rebuild list
            new_partition = []
            for k in range(len(partition)):
                if k != i and k != j:
                    new_partition.append(partition[k])
            new_partition.append(new_b)
            return new_partition
        return None

    # Move: Move edge from B_i to B_j
    # This is complex because we need to remove edge from B_i (might break biclique if it was "holding" vertices? No, removing edges doesn't break biclique property of remaining vertices).
    # But we need to ensure the edge can be added to B_j.
    # If we move edge (u,v) from B_i to B_j:
    # 1. Check if (u,v) can be added to B_j. 
    #    This means u and v must be compatible with L_j, R_j.
    #    Specifically, if u is not in L_j U R_j, we can add it to L_j or R_j?
    #    Actually, the biclique is defined by the sets. If we add an edge, we might need to expand L_j or R_j.
    #    This is effectively "expanding" B_j to cover (u,v).
    #    If B_j expands, it might cover OTHER edges that were in B_i or other bicliques.
    #    This violates the "disjoint cross-edge sets" constraint.
    #    So "Move" is tricky.
    
    # Better Move: "Swap" or "Reassign" is hard.
    # Let's stick to Merge and "Split" (reverse of merge) and "Relocate Vertex".
    
    # Alternative Move: Pick a vertex v in B_i. Try to move it to B_j.
    # If v is in L_i, moving it to L_j means:
    # - All edges (v, r) for r in R_i are removed from B_i.
    # - All edges (v, r) for r in R_j are added to B_j.
    # - This requires that these edges are NOT covered by other bicliques.
    # - And that B_i \ {v} is still a valid biclique (it is).
    # - And that B_j U {v} is a valid biclique.
    # - AND that the edges moved are actually present in the graph and not covered elsewhere.
    
    # This is getting complicated to track disjointness efficiently.
    # Given the small size (30 nodes, ~200 edges), we can afford to be slower.
    
    # Let's implement a simpler local search:
    # 1. Try all pairs to merge.
    # 2. If no merge possible, try to "improve" by splitting a large biclique and re-merging parts?
    
    # Actually, the "Merge" move is the most powerful for reducing count.
    # If we get stuck, we can try "Perturbation": Randomly split a biclique into two.
    
    current_partition = copy.deepcopy(best_partition)
    current_count = len(current_partition)
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        if T < T_min:
            break
            
        # Try Merge
        merged = False
        if len(current_partition) > 1:
            i, j = random.sample(range(len(current_partition)), 2)
            new_p = try_merge(i, j)
            if new_p:
                # Merge is always an improvement (reduces count)
                current_partition = new_p
                current_count = len(current_partition)
                if current_count < best_count:
                    best_count = current_count
                    best_partition = copy.deepcopy(current_partition)
                T *= cooling
                continue
                
        # If no merge, try Perturbation: Split a random biclique
        # Pick a biclique with size > 2
        candidates = [k for k in range(len(current_partition)) if len(current_partition[k][0]) + len(current_partition[k][1]) > 2]
        if candidates:
            k = random.choice(candidates)
            L, R = current_partition[k]
            # Pick a random vertex to move? Or just split randomly?
            # Let's pick a random vertex u from L (if L not empty) and move it to a new biclique?
            # Or split L into L1, L2.
            # Let's just create a new biclique with a random subset of edges from this one.
            # This is risky for validity.
            
            # Instead, let's try "Relocate Vertex" if we can track it.
            # For now, let's just accept that we might get stuck and rely on multiple restarts.
            
        # Cooling
        T *= cooling
        
    # Final check and format
    # Ensure we return the best found
    final_bicliques = []
    for L, R in best_partition:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_bicliques.append([final_L, final_R])
        
    return final_bicliques

# To improve, we should add a "Restart" mechanism if we get stuck.
# But the above is a single run. Let's wrap it in a restart loop.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
    all_edges = list(edge_set)
    
    best_partition = None
    best_count = float('inf')
    
    def is_biclique(L, R):
        if not L or not R:
            return False
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    def get_edges(L, R):
        return set((min(u,v), max(u,v)) for u in L for v in R)

    def greedy_init():
        remaining_edges = set(all_edges)
        partition = []
        while remaining_edges:
            if time.time() > deadline:
                # Fallback
                for u, v in remaining_edges:
                    partition.append([[u], [v]])
                return partition
            e = random.choice(list(remaining_edges))
            u, v = e
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if all(adj[x][y] for y in R):
                        if all((min(x,y), max(x,y)) in remaining_edges for y in R):
                            L.add(x)
                            changed = True
                for y in range(n):
                    if y in R: continue
                    if all(adj[x][y] for x in L):
                        if all((min(x,y), max(x,y)) in remaining_edges for x in L):
                            R.add(y)
                            changed = True
            covered = get_edges(L, R)
            remaining_edges -= covered
            partition.append([list(L), list(R)])
        return partition

    def try_merge(partition, i, j):
        L1, R1 = partition[i]
        L2, R2 = partition[j]
        L_new = set(L1) | set(L2)
        R_new = set(R1) | set(R2)
        if is_biclique(L_new, R_new):
            new_b = [list(L_new), list(R_new)]
            new_partition = []
            for k in range(len(partition)):
                if k != i and k != j:
                    new_partition.append(partition[k])
            new_partition.append(new_b)
            return new_partition
        return None

    # Run multiple restarts
    while time.time() < deadline - 2:
        partition = greedy_init()
        count = len(partition)
        
        # Local Search
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        
        for step in range(1000):
            if time.time() > deadline - 2:
                break
            if T < T_min:
                break
                
            improved = False
            if len(partition) > 1:
                # Try random merge
                i, j = random.sample(range(len(partition)), 2)
                new_p = try_merge(partition, i, j)
                if new_p:
                    partition = new_p
                    count = len(partition)
                    improved = True
            
            if not improved:
                # Perturbation: Randomly split a biclique to escape local optimum
                # Pick a biclique, pick a random vertex, move it to a new biclique?
                # This is hard to do validly without checking global constraints.
                # Simple perturbation: Shuffle the vertices in a random biclique?
                # No, that breaks it.
                
                # Let's just cool down.
                pass
                
            T *= cooling
            
        if count < best_count:
            best_count = count
            best_partition = copy.deepcopy(partition)
            
    if best_partition is None:
        # Should not happen, but fallback
        best_partition = [[ [u], [v] ] for u, v in all_edges]
        
    final_bicliques = []
    for L, R in best_partition:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_bicliques.append([final_L, final_R])
        
    return final_bicliques

# Add a better perturbation: "Edge Swap"
# If we can't merge, we might be stuck.
# A better perturbation: Pick two bicliques, and try to swap a vertex between them if valid.
# Or, simply restart with a different random seed/order. The loop above does restarts.

# Let's refine the perturbation to be more effective.
# If we get stuck (no merges for X steps), we can try to "break" a biclique by moving a vertex to a new biclique.
# This increases count, but might allow future merges.

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
    all_edges = list(edge_set)
    
    best_partition = None
    best_count = float('inf')
    
    def is_biclique(L, R):
        if not L or not R:
            return False
        for u in L:
            for v in R:
                if not adj[u][v]:
                    return False
        return True

    def get_edges(L, R):
        return set((min(u,v), max(u,v)) for u in L for v in R)

    def greedy_init():
        remaining_edges = set(all_edges)
        partition = []
        while remaining_edges:
            if time.time() > deadline:
                for u, v in remaining_edges:
                    partition.append([[u], [v]])
                return partition
            e = random.choice(list(remaining_edges))
            u, v = e
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if all(adj[x][y] for y in R):
                        if all((min(x,y), max(x,y)) in remaining_edges for y in R):
                            L.add(x)
                            changed = True
                for y in range(n):
                    if y in R: continue
                    if all(adj[x][y] for x in L):
                        if all((min(x,y), max(x,y)) in remaining_edges for x in L):
                            R.add(y)
                            changed = True
            covered = get_edges(L, R)
            remaining_edges -= covered
            partition.append([list(L), list(R)])
        return partition

    def try_merge(partition, i, j):
        L1, R1 = partition[i]
        L2, R2 = partition[j]
        L_new = set(L1) | set(L2)
        R_new = set(R1) | set(R2)
        if is_biclique(L_new, R_new):
            new_b = [list(L_new), list(R_new)]
            new_partition = []
            for k in range(len(partition)):
                if k != i and k != j:
                    new_partition.append(partition[k])
            new_partition.append(new_b)
            return new_partition
        return None

    while time.time() < deadline - 2:
        partition = greedy_init()
        count = len(partition)
        
        T = 10.0
        T_min = 0.1
        cooling = 0.95
        no_improve_steps = 0
        
        for step in range(2000):
            if time.time() > deadline - 2:
                break
            if T < T_min:
                break
                
            improved = False
            if len(partition) > 1:
                i, j = random.sample(range(len(partition)), 2)
                new_p = try_merge(partition, i, j)
                if new_p:
                    partition = new_p
                    count = len(partition)
                    improved = True
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
            if not improved and no_improve_steps > 50:
                # Perturbation: Split a random biclique
                # Pick a biclique with > 1 edge
                candidates = [k for k in range(len(partition)) if len(partition[k][0]) > 1 or len(partition[k][1]) > 1]
                if candidates:
                    k = random.choice(candidates)
                    L, R = partition[k]
                    # Pick a random vertex from L (if L has > 1) or R (if R has > 1)
                    # Move it to a new biclique?
                    # Let's just create a new biclique with a random subset of edges.
                    # This is complex. Let's just restart the local search with a perturbed partition.
                    # Simplest perturbation: Shuffle the order of edges in greedy init?
                    # Or, take one biclique and split it into two by picking a random vertex and moving it.
                    # Let's try: Pick vertex v in L. Create B_new with L_new={v}, R_new=R.
                    # This is valid if (v, r) for all r in R are edges (they are).
                    # But we must remove v from L in the original biclique.
                    # Original L becomes L \ {v}.
                    # This splits B into B1 (L\{v}, R) and B2 ({v}, R).
                    # This increases count by 1.
                    # It might allow B1 or B2 to merge with others later.
                    
                    if L and len(L) > 1:
                        v = random.choice(L)
                        L_old = set(L)
                        R_set = set(R)
                        L_new1 = list(L_old - {v})
                        R_new1 = list(R_set)
                        L_new2 = [v]
                        R_new2 = list(R_set)
                        
                        # Replace partition[k] with two bicliques
                        new_partition = []
                        for idx, b in enumerate(partition):
                            if idx == k:
                                new_partition.append([L_new1, R_new1])
                                new_partition.append([L_new2, R_new2])
                            else:
                                new_partition.append(b)
                        partition = new_partition
                        count = len(partition)
                        no_improve_steps = 0
                    elif R and len(R) > 1:
                        v = random.choice(R)
                        L_set = set(L)
                        R_old = set(R)
                        L_new1 = list(L_set)
                        R_new1 = list(R_old - {v})
                        L_new2 = list(L_set)
                        R_new2 = [v]
                        
                        new_partition = []
                        for idx, b in enumerate(partition):
                            if idx == k:
                                new_partition.append([L_new1, R_new1])
                                new_partition.append([L_new2, R_new2])
                            else:
                                new_partition.append(b)
                        partition = new_partition
                        count = len(partition)
                        no_improve_steps = 0

            T *= cooling
            
        if count < best_count:
            best_count = count
            best_partition = copy.deepcopy(partition)
            
    if best_partition is None:
        best_partition = [[ [u], [v] ] for u, v in all_edges]
        
    final_bicliques = []
    for L, R in best_partition:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_bicliques.append([final_L, final_R])
        
    return final_bicliques

