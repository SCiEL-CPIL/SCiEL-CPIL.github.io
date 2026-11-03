import itertools
import time
import numpy as np

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    # Using Python integers for bitsets is hard to vectorize with NumPy directly 
    # for bit_count, but we can use a matrix of 0/1s if memory allows.
    # 75,000 blocks * 11,628 t-subsets = 872 million bits = ~100 MB as bytes.
    # This is feasible!
    
    # Create a boolean matrix: coverage_matrix[i, j] = 1 if block i covers t-subset j
    coverage_matrix = np.zeros((n_k_subsets, n_t_subsets), dtype=bool)
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            coverage_matrix[i, idx] = True
            
    # Greedy Set Cover using NumPy
    uncovered = np.ones(n_t_subsets, dtype=bool)
    selected = []
    used = np.zeros(n_k_subsets, dtype=bool)
    
    while uncovered.any():
        # Calculate gain for each block: number of uncovered t-subsets it covers
        # gain = sum(coverage_matrix[i] & uncovered)
        # We can do this with matrix multiplication or sum.
        # Since coverage_matrix is boolean, we can cast to int8 or just use sum.
        # sum is O(N*M). 75k * 11k = 872M ops. Might be slow in pure Python loop, 
        # but NumPy vectorization helps if we do it in one go?
        # np.sum(coverage_matrix & uncovered, axis=1)
        
        gains = np.sum(coverage_matrix & uncovered, axis=1)
        
        # Mask out used blocks
        gains[used] = -1
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain <= 0:
            break
            
        selected.append(int(best_idx))
        used[best_idx] = True
        uncovered = uncovered & ~coverage_matrix[best_idx]
        
    best_solution = selected[:]
    
    # Local Search: Redundancy Removal
    # Convert selected to numpy array of indices
    # We need to check if removing a block leaves all t-subsets covered.
    
    def remove_redundant(sol):
        n = len(sol)
        if n == 0: return sol
        
        # Create a view of the coverage matrix for the selected blocks
        # sol_coverage = coverage_matrix[sol]
        
        changed = True
        while changed:
            changed = False
            n = len(sol)
            if n <= 1: break
            
            # Calculate union of all selected blocks
            # union_mask = np.any(coverage_matrix[sol], axis=0)
            # This is O(N*M) per iteration.
            
            # To speed up, we can maintain a count of how many times each t-subset is covered.
            # count[j] = number of blocks in sol that cover t-subset j.
            # A block is redundant if for all j it covers, count[j] > 1.
            
            # Let's use the count array approach.
            # Rebuild count array for current sol
            counts = np.zeros(n_t_subsets, dtype=int)
            for idx in sol:
                counts[coverage_matrix[idx]] += 1
            
            # Find redundant blocks
            # A block i is redundant if for all j where coverage_matrix[i, j] is True, counts[j] > 1.
            # i.e., coverage_matrix[i] & (counts == 1) is empty.
            
            redundant_indices = []
            for i, idx in enumerate(sol):
                # Check if this block is the *only* one covering any t-subset
                # unique_part = coverage_matrix[idx] & (counts == 1)
                # if unique_part.any() is False, it's redundant.
                if not np.any(coverage_matrix[idx] & (counts == 1)):
                    redundant_indices.append(i)
            
            if redundant_indices:
                # Remove the first one found
                rem_idx = redundant_indices[0]
                sol.pop(rem_idx)
                changed = True
                # Counts will be rebuilt in next loop iteration
                # Or we can update counts here, but rebuilding is simpler and n is small.
                
        return sol

    best_solution = remove_redundant(best_solution)
    
    # Try a few multi-starts if time permits
    start_time = time.time()
    best_found = best_solution[:]
    
    for _ in range(10):
        if time.time() - start_time > 100:
            break
            
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected = []
        used = np.zeros(n_k_subsets, dtype=bool)
        
        while uncovered.any():
            gains = np.sum(coverage_matrix & uncovered, axis=1)
            gains[used] = -1
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
                
            selected.append(int(best_idx))
            used[best_idx] = True
            uncovered = uncovered & ~coverage_matrix[best_idx]
            
        selected = remove_redundant(selected)
        
        if len(selected) < len(best_found):
            best_found = selected[:]
            
    return [list(k_subsets[i]) for i in best_found]
