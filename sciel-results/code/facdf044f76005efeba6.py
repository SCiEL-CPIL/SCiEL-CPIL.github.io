"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Baseline: weight-2 vectors
    base_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
    
    base_set = np.array(base_vectors, dtype=np.int64)
    best_set = base_set
    best_score = len(base_set)
    
    # Strategy: Maximize set of weight-4 vectors (norm 2).
    # Constraint: pairwise dot <= 2.
    # Candidates: All vectors with 4 non-zero entries from {1, -1}.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280.
    
    # Generate candidates
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    
    # Build Conflict Graph
    # Two vectors conflict if their dot product > 2
    # Dot product of two weight-4 vectors with entries +/-1:
    # Let k be number of common supports with same sign.
    # Let l be number of common supports with opposite sign.
    # dot = k - l.
    # We need dot <= 2.
    # If they share 4 supports:
    #   same signs: 4, opp: 0 -> dot 4 (conflict)
    #   3 same, 1 opp -> dot 2 (ok)
    #   2 same, 2 opp -> dot 0 (ok)
    #   1 same, 3 opp -> dot -2 (ok)
    #   0 same, 4 opp -> dot -4 (ok)
    # If they share 3 supports:
    #   max dot 3 (if all same sign on shared, and one unique doesn't matter)
    #   Wait, dot is sum over all 11 dims.
    #   If supports differ, they are orthogonal on non-shared dims.
    #   So dot is just sum over shared dims.
    #   If 3 shared, max dot is 3 (if all same sign). 3 > 2, so conflict.
    #   So if they share 3 or 4 supports, they might conflict.
    #   Specifically:
    #   Share 4: Conflict if >= 3 same signs.
    #   Share 3: Conflict if 3 same signs.
    #   Share 2: Max dot 2. Never conflict.
    #   Share 1: Max dot 1. Never conflict.
    #   Share 0: Dot 0. Never conflict.
    
    # Precompute neighbors for each candidate
    # For speed, we can represent vectors as integers or use the array.
    # n_cands = 5280.
    # Adjacency list.
    
    neighbors = [set() for _ in range(n_cands)]
    
    # Optimization: Group candidates by support (4-subset).
    # There are C(11,4) = 330 supports.
    # Each support has 16 sign vectors.
    # Conflicts only happen between:
    # 1. Vectors with the same support (16 vectors per support).
    # 2. Vectors with supports that share 3 or 4 elements.
    
    # Let's just compute the full adjacency matrix or list once.
    # 5280 * 5280 / 2 = ~14 million checks.
    # Each check is a dot product or logic.
    # Using numpy for all pairs at once might be memory heavy (5280x5280 int8 = 28MB).
    
    # Let's do it in batches or use the logic.
    # Actually, let's just build the adjacency list by iterating pairs.
    # To speed up, we can pre-compute dot products for all pairs?
    # 5280^2 = 27.8M dot products of length 11.
    # This is fast in numpy if done in chunks.
    
    # Create adjacency matrix (boolean)
    # To save memory, we can just store the neighbor indices.
    
    # Let's use a smarter approach:
    # For each vector i, find j > i such that dot(v_i, v_j) > 2.
    
    # We can vectorize this.
    # For i in range(n_cands):
    #   dots = cand_arr[i] @ cand_arr[i+1:].T
    #   conflicts = np.where(dots > 2)[0] + i + 1
    #   neighbors[i].update(conflicts)
    #   for c in conflicts:
    #       neighbors[c].add(i)
    
    # This is O(N^2 * D) operations. 5280^2 * 11 ~ 300M ops.
    # In Python loop, this is too slow.
    # We need vectorization.
    
    # Let's compute the full Gram matrix G = cand_arr @ cand_arr.T
    # G is 5280x5280.
    # 5280^2 * 8 bytes (float64) = 223 MB.
    # 5280^2 * 4 bytes (float32) = 111 MB.
    # 5280^2 * 1 byte (int8) = 28 MB.
    # Dot products are integers in [-4, 4]. int8 is enough.
    
    # Compute G
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Create adjacency list
    neighbors = [set() for _ in range(n_cands)]
    # Find indices where G > 2
    # We only need upper triangle to avoid duplicates
    # But np.where on full matrix is fine.
    
    # To be memory efficient and fast, let's just use the matrix G for the greedy algorithm.
    # We don't even need the adjacency list if we use G directly in the greedy step,
    # but checking "is v_i adjacent to any selected?" is O(k) per step.
    # With adjacency list, it's O(degree).
    
    # Let's build adjacency lists from G.
    # This takes time but is a one-time cost.
    
    # We can use scipy.sparse if available, but let's stick to numpy.
    # Iterating over 5280 rows and finding non-zeroes.
    
    # Actually, for the "Minimum Degree" greedy, we need to update degrees dynamically.
    # This requires knowing the neighbors.
    
    # Let's build the neighbor sets.
    for i in range(n_cands):
        # Get indices where G[i] > 2
        # Exclude i itself (G[i,i] = 4 > 2, but it's not a neighbor)
        conflicts = np.where(G[i] > 2)[0]
        # Remove self
        conflicts = conflicts[conflicts != i]
        neighbors[i] = set(conflicts.tolist())
        
    # Now run Minimum Degree Greedy with Randomization
    rng = np.random.default_rng(42)
    
    max_restarts = 50 # Fewer restarts needed if heuristic is strong
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomize degrees slightly to break ties
        # Or just randomize the order of tie-breaking
        # Let's use a priority queue logic or just scan.
        
        # To make it faster, we can use a list of active nodes.
        # But updating degrees is tricky.
        
        # Simple approach:
        # active = set(range(n_cands))
        # selected = []
        # while active:
        #   find v in active with min (len(neighbors[v] & active))
        #   this is O(N^2) per selection. Too slow.
        
        # Better:
        # Maintain current degree for each active node.
        # When we pick v, we remove v and its neighbors.
        # For each neighbor u of v that is still active, decrement degree[u].
        
        # This is O(N * E) total?
        # Each edge is processed at most twice (once for each endpoint removal).
        # So total work is O(E).
        # E is ~ 5280 * 57 / 2 ~ 150k edges.
        # This is very fast!
        
        active = [True] * n_cands
        # Current degree of each node in the active subgraph
        deg = [len(neighbors[i]) for i in range(n_cands)]
        
        # We need to efficiently find the node with min degree among active.
        # Using a heap is good, but degrees change.
        # Since N=5280, a linear scan O(N) per selection is O(N^2) total ~ 28M ops.
        # This is acceptable in Python if optimized?
        # 28M simple ops might take 1-2 seconds.
        # 50 restarts -> 50-100 seconds. Might be tight.
        
        # Let's try to optimize the scan.
        # Or use a bucket queue? Degrees are small (< 60).
        
        # Let's just do the linear scan but check time.
        
        selected_indices = []
        
        # To add randomness, we can add random noise to degrees initially
        # or pick randomly among min-degree nodes.
        
        for _ in range(n_cands):
            if time.time() > deadline - 5:
                break
                
            if not any(active):
                break
                
            # Find active node with min degree
            min_deg = float('inf')
            min_node = -1
            # Random tie breaking: add small random noise to deg for comparison?
            # Or just pick first min.
            
            # To speed up, we can maintain a list of candidates with low degree?
            # For now, simple scan.
            
            for i in range(n_cands):
                if active[i]:
                    if deg[i] < min_deg:
                        min_deg = deg[i]
                        min_node = i
            
            if min_node == -1:
                break
                
            # Select min_node
            selected_indices.append(min_node)
            active[min_node] = False
            
            # Update neighbors
            for nb in neighbors[min_node]:
                if active[nb]:
                    deg[nb] -= 1
                    # Note: We don't remove nb from active here, 
                    # because nb might still be selectable if it has no other active neighbors?
                    # No, if nb is a neighbor of selected node, it CANNOT be selected.
                    # So we MUST remove it from active.
                    active[nb] = False
                    # Wait, if we remove nb, do we need to update its neighbors' degrees?
                    # Yes, because nb is no longer in the graph.
                    # But we are iterating over neighbors of min_node.
                    # We need to propagate the removal.
                    
            # This logic is flawed. Removing a node affects its neighbors' degrees.
            # If we remove nb, we must decrement degree of all neighbors of nb.
            # This becomes a BFS/DFS propagation.
            
            # Let's restart the logic for Minimum Degree Greedy.
            # Standard algorithm:
            # 1. While vertices remain:
            # 2. Pick v with min degree.
            # 3. Add v to Independent Set.
            # 4. Remove v and all its neighbors from the graph.
            # 5. Update degrees of remaining vertices.
            
            # Step 4 & 5 combined:
            # When we remove a set R of vertices, for each u not in R,
            # deg[u] -= |N(u) ∩ R|.
            
            # In the loop above, R = {min_node} ∪ N(min_node).
            # So for each u active, if u is neighbor of min_node, it is removed.
            # For each u active (not removed), if u is neighbor of any removed node, deg[u] decreases.
            
            # Let's implement it correctly.
            
            # The previous code block was inside the restart loop, I will rewrite that section properly in the final edit.
            
            pass # Placeholder for actual implementation below

    # The above logic inside the loop was getting complex. Let's write a clean helper or inline it properly.
    
    # Re-implementing the core loop cleanly:
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected_indices = []
        
        while np.any(active):
            if time.time() > deadline - 5:
                break
                
            # Find min degree among active
            # Mask inactive with inf
            current_deg = np.where(active, deg, np.inf)
            min_idx = np.argmin(current_deg)
            
            if not active[min_idx]:
                break
                
            # Select min_idx
            selected_indices.append(min_idx)
            active[min_idx] = False
            
            # Get neighbors to remove
            nbs = neighbors[min_idx]
            
            # Create a set of nodes to remove: min_idx and its active neighbors
            # We need to update degrees of all active nodes that are neighbors of removed nodes.
            
            # Efficient update:
            # For each active node u, deg[u] -= number of neighbors in removed_set.
            
            # Since N is small, we can just iterate through all active nodes?
            # Or iterate through neighbors of removed nodes.
            
            removed_set = {min_idx}
            for nb in nbs:
                if active[nb]:
                    removed_set.add(nb)
                    active[nb] = False
            
            # Now update degrees for all remaining active nodes.
            # A node u's degree decreases by the number of its neighbors in removed_set.
            
            # We can do this by iterating through removed_set and for each v in removed_set,
            # iterate through its neighbors and decrement their degree if they are still active.
            
            for v in removed_set:
                for nb in neighbors[v]:
                    if active[nb]:
                        deg[nb] -= 1
            
        if len(selected_indices) > best_score:
            best_score = len(selected_indices)
            best_set = cand_arr[selected_indices].copy()
            
    return best_set

# EVOLVE-BLOCK-END
