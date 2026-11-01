"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys
from collections import defaultdict

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Identify all integer vectors with norm squared 4.
       - Weight 4: four +/-1s. (5280 vectors)
       - Weight 1 doubled: one +/-2. (22 vectors)
    2. Weight 1 doubled vectors are compatible with all Weight 4 vectors and each other.
    3. Solve Maximum Independent Set (MIS) on the conflict graph of Weight 4 vectors.
       Conflict if dot product > 2.
    4. Use Tomita's Branch-and-Bound algorithm with a time limit to find a large
       independent set.
    5. Combine the selected Weight 4 vectors with all Weight 1 doubled vectors.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # --- Step 1: Generate Vectors ---
    
    # Weight 4 vectors
    w4_supports = list(itertools.combinations(range(D), 4))
    w4_vectors = []
    support_to_indices = defaultdict(list)
    
    for support in w4_supports:
        for signs in itertools.product([1, -1], repeat=4):
            idx = len(w4_vectors)
            w4_vectors.append((support, signs))
            support_to_indices[support].append(idx)
            
    N = len(w4_vectors)
    
    # Weight 1 doubled vectors
    w1_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            w1_vectors.append(v)
    w1_vectors = np.array(w1_vectors, dtype=np.int64)
    
    # --- Step 2: Build Conflict Graph for Weight 4 ---
    # Conflict if dot product > 2.
    # As derived, conflicts only occur if supports overlap in 3 or 4 positions.
    # Overlap 4 (same support): distinct vectors have dot product <= 2. No conflict.
    # Overlap 3: conflict if signs on overlapping 3 positions are identical.
    
    neighbors = [set() for _ in range(N)]
    
    # For each pair of supports with intersection size 3
    support_list = list(support_to_indices.keys())
    n_sup = len(support_list)
    
    # To speed up, iterate over supports and find those sharing 3 elements
    # Map from frozenset of 3 elements to list of supports containing them?
    # Or just iterate pairs? 330 supports. 330^2 / 2 ~ 50k pairs. Fast.
    
    for i in range(n_sup):
        S1 = support_list[i]
        for j in range(i + 1, n_sup):
            S2 = support_list[j]
            inter = set(S1) & set(S2)
            if len(inter) != 3:
                continue
            
            # Get indices
            idx1_list = support_to_indices[S1]
            idx2_list = support_to_indices[S2]
            
            # Map positions in S1 and S2 to the intersection
            inter_sorted = sorted(inter)
            # pos_in_S1[k] is the index in S1 of inter_sorted[k]
            pos_in_S1 = [S1.index(p) for p in inter_sorted]
            pos_in_S2 = [S2.index(p) for p in inter_sorted]
            
            for idx1 in idx1_list:
                sig1 = w4_vectors[idx1][1]
                for idx2 in idx2_list:
                    sig2 = w4_vectors[idx2][1]
                    # Check if signs on inter_sorted match
                    all_same = True
                    for p1, p2 in zip(pos_in_S1, pos_in_S2):
                        if sig1[p1] != sig2[p2]:
                            all_same = False
                            break
                    if all_same:
                        neighbors[idx1].add(idx2)
                        neighbors[idx2].add(idx1)
                        
    # Convert to numpy arrays for faster access in B&B?
    # Or keep as list of lists/sets.
    # For Tomita, we need to efficiently remove neighbors.
    # Let's keep as list of sets.
    
    # --- Step 3: Solve MIS ---
    
    # Initial Greedy MIS
    def greedy_mis():
        remaining = set(range(N))
        selected = []
        while remaining:
            if time.time() > deadline - 10:
                break
            # Pick node with min degree in remaining
            min_deg = float('inf')
            best_node = -1
            for node in remaining:
                deg = len(neighbors[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
            selected.append(best_node)
            # Remove best_node and its neighbors
            to_remove = neighbors[best_node] | {best_node}
            remaining -= to_remove
        return selected

    best_set = greedy_mis()
    best_size = len(best_set)
    
    # Tomita's Branch and Bound
    # We will use a recursive function, but with a time check.
    # To avoid recursion depth issues and overhead, we might use a stack,
    # but recursion is easier to write. N=5280 is deep.
    # sys.setrecursionlimit(10000)
    
    # Actually, Tomita's algorithm is often implemented iteratively or with careful pruning.
    # Given the time limit, a full exact search might not complete.
    # We will use the best solution found by the search tree traversal.
    
    # We can also try Local Search (Kempe Chains) if B&B is too slow or stuck.
    # Let's implement a simple Local Search improvement on the greedy result first,
    # then run B&B with that as the initial bound.
    
    def local_search_improve(initial_set, time_limit=20.0):
        current_set = set(initial_set)
        current_size = len(current_set)
        best_size_ls = current_size
        best_set_ls = list(current_set)
        
        t_start = time.time()
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < t_start + time_limit:
            if no_improve_count > max_no_improve:
                break
                
            improved = False
            # Try 1-swap (useless for size increase)
            # Try 2-swap: Remove 2, Add 3?
            # Or simpler: Randomly remove 1 node, try to add back + others?
            
            # Let's try a "shake" and re-greedy on the remaining?
            # Or just pick a random node v NOT in set.
            # If v has no neighbors in set, add it. (Greedy should have done this)
            # So we need to remove some nodes.
            
            # Pick a random node v in set.
            # Remove it.
            # Then greedily add nodes that are now available?
            
            # This is a 1-move local search.
            # Let's try: Pick random v in current_set. Remove v.
            # Then run greedy on the remaining candidates (nodes not in set, not neighbors of set)
            # to see if we can add more than 1.
            
            v = list(current_set)[np.random.randint(len(current_set))]
            current_set.remove(v)
            
            # Greedy add
            added = 0
            available = set(range(N)) - current_set
            # Remove neighbors of current_set from available
            for u in current_set:
                available -= neighbors[u]
                
            # Greedy on available
            while available:
                if time.time() > t_start + time_limit:
                    break
                # Pick min degree in available
                min_deg = float('inf')
                best_node = -1
                for node in available:
                    deg = len(neighbors[node] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_node = node
                current_set.add(best_node)
                added += 1
                to_remove = neighbors[best_node] | {best_node}
                available -= to_remove
                
            if len(current_set) > best_size_ls:
                best_size_ls = len(current_set)
                best_set_ls = list(current_set)
                no_improve_count = 0
                improved = True
            else:
                no_improve_count += 1
                # Revert to best? Or keep current?
                # Keep current to explore, but track best.
                
        return best_set_ls, best_size_ls

    # Run Local Search
    best_set, best_size = local_search_improve(best_set, time_limit=10.0)
    
    # Run Tomita B&B if time permits
    # Tomita's algorithm requires careful implementation.
    # Given the complexity and time, we will stick to the Local Search result
    # unless we can implement a very fast B&B.
    # For now, we will try a few more restarts of Greedy + Local Search.
    
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # New Greedy with random tie-breaking
        remaining = set(range(N))
        selected = []
        while remaining:
            if time.time() > deadline - 5:
                break
            min_deg = float('inf')
            candidates = []
            for node in remaining:
                deg = len(neighbors[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [node]
                elif deg == min_deg:
                    candidates.append(node)
            best_node = np.random.choice(candidates)
            selected.append(best_node)
            to_remove = neighbors[best_node] | {best_node}
            remaining -= to_remove
            
        # Local Search on this new set
        ls_set, ls_size = local_search_improve(selected, time_limit=5.0)
        if ls_size > best_size:
            best_size = ls_size
            best_set = ls_set

    # --- Step 4: Combine and Return ---
    
    # Convert best_set indices to vectors
    w4_int_vectors = np.zeros((best_size, D), dtype=np.int64)
    for i, idx in enumerate(best_set):
        sup, sig = w4_vectors[idx]
        for pos, sign in zip(sup, sig):
            w4_int_vectors[i, pos] = sign
            
    all_vectors = np.vstack([w4_int_vectors, w1_vectors])
    
    return all_vectors

# EVOLVE-BLOCK-END
