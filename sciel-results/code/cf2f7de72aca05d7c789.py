"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate all candidate vectors
    # Weight 4 vectors: norm^2 = 4
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Weight 1 vectors: norm^2 = 4
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Combine all candidates
    all_vectors = v4_candidates + v1_candidates
    N = len(all_vectors)
    
    # 2. Precompute conflict masks
    # Conflict if dot product > 2 (for distinct vectors)
    # Note: W1 vectors are compatible with all W4 vectors and each other.
    # We can just compute for all pairs.
    
    # Using Python integers for bitsets
    # N is at most 5280 + 22 = 5302. Fits easily in a Python int.
    
    # Dot products
    # v4 dot v4:
    # v4 dot v1:
    # v1 dot v1:
    
    # We will build a list of integer masks `conflicts[i]`
    # where bit j is set if vector i and vector j conflict.
    
    conflicts = [0] * N
    all_mask = (1 << N) - 1
    
    # Optimization: Group by support for W4?
    # Or just brute force with numpy?
    # N=5302. N^2 = 28M.
    # Vectorized dot product:
    V = np.array(all_vectors, dtype=np.int8)
    # Dot product matrix
    # This might be memory heavy (5302x5302 int8 = 28MB). OK.
    # But we only need > 2.
    
    # Let's do it in chunks to be safe and fast.
    # Actually, Python int bit operations are very fast.
    # Let's compute dot products in numpy, then convert to masks.
    
    # To avoid 28MB float array, we can use int16.
    # Max dot product is 4. Min is -4.
    # So int8 is enough.
    
    # Compute dot products
    # We can do V @ V.T
    # This is 5302x5302.
    # Time: ~0.1s to 0.5s.
    
    dot_products = V @ V.T
    
    # Conflict if dot_products[i][j] > 2 and i != j
    # Also, if i == j, they conflict (can't pick same vector twice).
    
    # Build masks
    # Iterate i, find j where dot > 2.
    # Set bit j in conflicts[i].
    
    for i in range(N):
        # Find indices where dot product > 2
        # Exclude i itself? No, i conflicts with i.
        # We want the mask of ALL conflicting nodes.
        mask = 0
        # Vectorized check
        # This loop is 5302 iterations.
        # Each iteration checks 5302 elements.
        # Total 28M checks. In Python loop, this is slow (~1-2s).
        # Let's optimize.
        
        # Get the row
        row = dot_products[i]
        # Find indices where row > 2
        # We need to set bits in a Python int.
        # np.where returns array of indices.
        idxs = np.where(row > 2)[0]
        # Set bits
        # Iterating over idxs to set bits is slow if many conflicts.
        # Average conflicts?
        # Most vectors are compatible.
        # Conflict only if dot > 2.
        # For W4, dot > 2 means dot = 3 or 4.
        # Dot 4: self.
        # Dot 3: same support, 3 matching signs.
        # Number of such vectors: 1 (self) + ?
        # For a given W4 vector, how many others have dot 3?
        # Same support, 3 signs match, 1 differs.
        # There are 4 positions. Choose 3 to match, 1 to differ.
        # Signs on matching pos are fixed. Sign on differing pos is flipped.
        # So 4 such vectors.
        # Plus vectors from other supports?
        # If overlap 3, dot can be 3.
        # How many W4 vectors overlap in 3?
        # Support S. Overlap 3 with S' means S' shares 3 with S.
        # Number of such S' is 28.
        # For each S', how many vectors have dot 3?
        # Depends on signs.
        # Roughly, few conflicts per node.
        # So idxs will be small (maybe 10-50).
        
        for j in idxs:
            mask |= (1 << j)
        conflicts[i] = mask

    # 3. Search for Max Independent Set
    # We want a set of nodes S such that for all i, j in S, i != j, (conflicts[i] >> j) & 1 == 0.
    # Equivalent: (S_mask & conflicts[i]) == 0 for all i in S?
    # No. S_mask contains i. conflicts[i] contains i.
    # So we should check (S_mask & conflicts[i]) == (1 << i) ?
    # Or better: remove self from conflicts?
    # Let's define `neighbors[i]` as mask of j != i such that they conflict.
    # Then condition is (S_mask & neighbors[i]) == 0.
    
    neighbors = [0] * N
    for i in range(N):
        # Remove self bit
        neighbors[i] = conflicts[i] & ~(1 << i)
        
    # Heuristic: Greedy Min-Degree
    def greedy_is():
        remaining = all_mask
        selected = 0
        # We need to pick node with min degree in remaining graph?
        # Degree in remaining graph = pop_count(neighbors[i] & remaining)
        
        # This is expensive to compute for all i every step.
        # Simple greedy: pick random, or pick min degree initially.
        
        # Let's use a simple randomized greedy
        # Order nodes by random priority?
        # Or just iterate.
        
        # Better: 
        # 1. Calculate initial degrees.
        # 2. Pick min degree node.
        # 3. Add to set.
        # 4. Remove it and its neighbors from remaining.
        
        # To speed up, we can just use a fixed order or random restarts.
        
        # Let's do a simple pass:
        # Iterate nodes, if compatible with current set, add.
        # This is O(N^2) worst case, but fast in practice.
        
        current_set = 0
        # Shuffle nodes for randomness
        nodes = list(range(N))
        random.shuffle(nodes)
        
        for i in nodes:
            # Check if i conflicts with any in current_set
            # (current_set & neighbors[i]) == 0
            if (current_set & neighbors[i]) == 0:
                current_set |= (1 << i)
                
        return current_set

    # Local Search (Simulated Annealing style)
    def local_search(initial_set, max_time=5.0):
        current_set = initial_set
        current_size = current_set.bit_count()
        best_set = current_set
        best_size = current_size
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        start_ls = time.time()
        
        while time.time() - start_ls < max_time:
            # Pick a random move
            # Move 1: Add a node, remove all its neighbors
            # Move 2: Remove a node
            # Move 3: Swap (Remove one, add another)
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to add a node
                # Candidates: nodes not in current_set
                # We can sample a few random nodes not in set
                # Or iterate?
                
                # To find a good candidate, we can look for nodes with 0 conflicts in set.
                # If found, add it (improvement).
                # If not, try a node with 1 conflict (swap).
                
                # Random sample of 100 nodes
                candidates = [random.randint(0, N-1) for _ in range(100)]
                
                best_candidate = -1
                best_delta = -1000
                best_conflicts = 0
                
                for i in candidates:
                    if (current_set >> i) & 1:
                        continue # Already in set
                    
                    conf = current_set & neighbors[i]
                    k = conf.bit_count()
                    delta = 1 - k
                    
                    # We prefer high delta
                    # But also accept worse with prob exp(delta/T)
                    
                    # For simplicity, just pick the best delta among candidates
                    # Or accept first one that is good?
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_candidate = i
                        best_conflicts = conf
                        
                if best_candidate != -1:
                    i = best_candidate
                    conf = best_conflicts
                    delta = best_delta
                    
                    # Acceptance
                    if delta >= 0:
                        accept = True
                    else:
                        accept = random.random() < np.exp(delta / T)
                    
                    if accept:
                        # Apply move
                        # Remove conf, add i
                        new_set = (current_set & ~conf) | (1 << i)
                        current_set = new_set
                        current_size = new_set.bit_count()
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set
                            
            elif move_type < 0.8:
                # Try to remove a node (to escape local optima)
                # Pick random node in set
                # Convert set to list? Or pick random bit.
                # Random bit in set:
                # Pick random index, check if in set.
                
                i = random.randint(0, N-1)
                while not ((current_set >> i) & 1):
                    i = random.randint(0, N-1)
                    
                # Remove i
                new_set = current_set & ~(1 << i)
                current_set = new_set
                current_size -= 1
                
            else:
                # 2-for-1 swap?
                # Remove 2 nodes, add 1?
                # Or just pass.
                pass
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_set, best_size

    # Run search
    best_overall_set = 0
    best_overall_size = 0
    
    # Initial greedy
    init_set = greedy_is()
    init_size = init_set.bit_count()
    if init_size > best_overall_size:
        best_overall_size = init_size
        best_overall_set = init_set
        
    # Local search from init
    ls_set, ls_size = local_search(init_set, max_time=10.0)
    if ls_size > best_overall_size:
        best_overall_size = ls_size
        best_overall_set = ls_set
        
    # More restarts
    while time.time() < deadline - 10:
        # New greedy
        init_set = greedy_is()
        init_size = init_set.bit_count()
        
        # Short local search
        ls_set, ls_size = local_search(init_set, max_time=5.0)
        
        if ls_size > best_overall_size:
            best_overall_size = ls_size
            best_overall_set = ls_set

    # Convert best_set (int mask) to list of vectors
    selected_vectors = []
    for i in range(N):
        if (best_overall_set >> i) & 1:
            selected_vectors.append(all_vectors[i])
            
    result = np.array(selected_vectors, dtype=np.int64)
    
    # Fallback if empty (should not happen)
    if len(result) == 0:
        result = np.array(all_vectors[:22], dtype=np.int64)
        
    return result

# EVOLVE-BLOCK-END
