import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy:
    # The optimal kissing configuration in D=11 (size 224) is contained within
    # the set of integer vectors with squared norm 2 (type A) and squared norm 4 (type B).
    # Type A: 2 non-zero entries of magnitude 1. Count: C(11,2)*4 = 220.
    # Type B: 1 non-zero entry of magnitude 2. Count: 11*2 = 22.
    # Total pool: 242 vectors.
    # We construct the conflict graph where an edge exists if dist^2 < max_norm_sq.
    # Note: max_norm_sq depends on the subset selected.
    # However, for the optimal set of size 224, it is known to consist of a mix.
    # If we select any Type B vector, max_norm_sq becomes 4.
    # If we select only Type A vectors, max_norm_sq is 2.
    # 
    # Let's analyze the two cases:
    # Case 1: Only Type A vectors.
    # Max norm sq = 2.
    # Condition: dist^2 >= 2.
    # We established that all 220 Type A vectors satisfy this.
    # So we can get 220 vectors.
    #
    # Case 2: Mix of Type A and Type B.
    # Max norm sq = 4.
    # Condition: dist^2 >= 4 for ALL pairs.
    # This is a stricter condition.
    # We need to find the maximum independent set in the graph defined by:
    # Node i, j are connected (conflict) if dist^2(i,j) < 4.
    # If we find an independent set of size > 220 in this graph, it must include Type B vectors
    # (since there are only 220 Type A vectors total, and they are not all mutually compatible
    # under the dist^2 >= 4 constraint? Wait.
    # Let's check if Type A vectors conflict with each other under dist^2 >= 4.
    # Dist^2 between two Type A vectors can be 2 (e.g., (1,1,0..) and (1,0,1..)).
    # 2 < 4. So they CONFLICT if we require dist^2 >= 4.
    # So we cannot take all 220 Type A vectors if we include any Type B vector.
    # We must prune the Type A set to be compatible with the chosen Type B vectors AND
    # with each other under the stricter dist^2 >= 4 rule.
    
    # Therefore, the problem reduces to:
    # Find the maximum independent set in the graph G where:
    # V = {Type A vectors} U {Type B vectors}
    # Edge (i,j) exists if dist^2(i,j) < 4.
    # Note: This graph is static. We don't need to worry about "max norm sq" changing
    # because if we pick ANY Type B vector, the max norm sq is 4.
    # If we pick NO Type B vectors, we are in Case 1 (size 220).
    # If we pick SOME Type B vectors, we are in Case 2.
    # The MIS of G will give the best solution for Case 2.
    # We compare size(MIS(G)) with 220.
    
    # Generate Type A vectors (norm sq 2)
    type_a = []
    indices_a = []
    for comb in itertools.combinations(range(D), 2):
        i, j = comb
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[i] = signs[0]
            v[j] = signs[1]
            type_a.append(tuple(v))
            indices_a.append(len(type_a) - 1) # just for tracking, not strictly needed
    
    # Generate Type B vectors (norm sq 4)
    type_b = []
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            type_b.append(tuple(v))
    
    all_vectors = type_a + type_b
    n = len(all_vectors)
    
    # Convert to numpy array for fast distance calculation
    V = np.array(all_vectors, dtype=np.int64)
    
    # Compute pairwise squared distances
    # dist^2 = ||v||^2 + ||w||^2 - 2 v.w
    norms_sq = np.sum(V**2, axis=1)
    Dots = V @ V.T
    DistSq = norms_sq[:, None] + norms_sq[None, :] - 2 * Dots
    
    # Conflict matrix for Case 2 (dist^2 < 4)
    # We treat diagonal as conflict (self) but MIS algorithms handle that or we ignore it.
    # Standard MIS: find largest subset with no edges between them.
    # Edge if DistSq[i,j] < 4.
    Conflict = DistSq < 4
    np.fill_diagonal(Conflict, True) # Self-conflict to prevent picking same node twice in some algos, though usually handled by logic
    
    # Tomita Algorithm for Maximum Independent Set
    # We implement a simplified version suitable for N=242.
    # The graph is relatively dense.
    
    best_set = []
    best_size = 0
    
    # We will use a recursive branch and bound.
    # To make it fast, we use the "coloring" bound or simple degree bound.
    # Given N=242, even a slightly inefficient B&B might work, but let's be smart.
    # We can use the fact that the graph is defined by geometry.
    
    # However, implementing Tomita from scratch in Python with full optimization is risky.
    # A simpler, effective approach for N=242 is a greedy with local search (2-swap) or
    # a simple backtracking with strong pruning.
    
    # Let's try a randomized greedy with local improvement (Simulated Annealing or Hill Climbing)
    # on the conflict graph.
    
    # Initialize with the known 220 Type A set (indices 0 to 219)
    # Check if this is independent in G (dist^2 >= 4).
    # We know it's not, because some Type A pairs have dist^2=2.
    # So we need to start from scratch or a valid small set.
    
    # Let's use a greedy MIS on the Conflict graph.
    # Order vertices by degree (ascending) is a good heuristic.
    
    # Compute degrees
    degrees = np.sum(Conflict, axis=1)
    
    # We'll run multiple random restarts of a greedy + local search.
    
    def greedy_mis(order):
        selected = set()
        available = np.ones(n, dtype=bool)
        for idx in order:
            if not available[idx]:
                continue
            # Check if idx can be added (no conflict with selected)
            # Since we maintain 'available' as nodes compatible with ALL previously selected,
            # if available[idx] is True, it is compatible.
            selected.add(idx)
            # Remove conflicts of idx from available
            available = available & ~Conflict[idx]
            available[idx] = False
        return selected

    def local_search(selected):
        # Try to add a vertex if possible
        # Try to swap a vertex if it allows adding more later?
        # Simple 1-move: Add a vertex v not in selected if it conflicts with no one in selected.
        # This is just greedy completion.
        
        # Better: Try to remove a vertex u from selected and add two vertices v1, v2?
        # For N=242, a simple "add if possible" loop until stuck is effective if started well.
        
        improved = True
        while improved:
            improved = False
            # Try adding any unselected vertex that is compatible
            for v in range(n):
                if v in selected:
                    continue
                # Check compatibility
                compatible = True
                for u in selected:
                    if Conflict[v, u]:
                        compatible = False
                        break
                if compatible:
                    selected.add(v)
                    improved = True
                    break # restart loop to check new possibilities
        return selected

    # Since the graph is fixed, we can just find the MIS.
    # Let's use a simple backtracking with pruning for the first few vertices, then greedy?
    # Or just use the Tomita coloring bound idea.
    
    # Given the small size, let's try a few random greedy starts followed by local search.
    
    best_solution = set()
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Random order weighted by degree (prefer low degree)
        # Or just random permutation
        order = np.random.permutation(n)
        
        # Greedy step
        current_set = greedy_mis(order)
        
        # Local search step
        current_set = local_search(current_set)
        
        if len(current_set) > len(best_solution):
            best_solution = current_set.copy()
            
            # If we found 224, we can stop early as it's the theoretical max
            if len(best_solution) >= 224:
                break
                
    # Compare with the pure Type A set (size 220)
    # The pure Type A set is valid for the original problem (max norm 2, dist >= 2).
    # But is it valid for the "mixed" problem?
    # The problem statement: "minimum pairwise distance is at least the maximum vector norm".
    # If we return the 220 Type A vectors:
    # Max norm sq = 2. Min dist sq = 2. Condition 2 >= 2 holds.
    # So 220 is always a valid baseline.
    
    # If best_solution from MIS (which enforces dist^2 >= 4) has size > 220, it is better.
    # Note: Any set found by MIS on the "dist^2 < 4" conflict graph is a set where
    # all pairwise distances are >= 4.
    # The max norm in such a set is 4 (if it contains Type B) or 2 (if only Type A).
    # If it contains only Type A, max norm is 2. Dist >= 4 implies Dist >= 2. Valid.
    # If it contains Type B, max norm is 4. Dist >= 4. Valid.
    # So any independent set in G is a valid kissing configuration.
    
    if len(best_solution) > 220:
        result_indices = list(best_solution)
        result_vectors = V[result_indices]
        return result_vectors.astype(np.int64)
    else:
        # Fallback to the 220 Type A vectors
        return V[:220].astype(np.int64)

# EVOLVE-BLOCK-END
