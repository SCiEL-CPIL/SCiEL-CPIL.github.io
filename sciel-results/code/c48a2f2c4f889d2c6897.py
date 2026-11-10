"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000
    max_sums = 2 * N
    
    def is_valid(arr):
        """Check if the set represented by boolean array arr satisfies B2[2]."""
        # arr is boolean array of length N+1 (index 0 unused)
        # Compute all pair sums a+b with a<=b
        # Use convolution-like approach with numpy
        # For efficiency, we can use the fact that we need to count representations
        # Let's use a direct but vectorized approach
        
        # Get indices of selected elements
        idx = np.nonzero(arr[1:])[0] + 1  # 1-based indices
        if len(idx) == 0:
            return True
        
        # Compute all pairwise sums (including self-pairs)
        # This is O(k^2) where k is the size of the set
        # For k ~ 700, this is 490,000 pairs which is manageable
        sums = []
        for i in range(len(idx)):
            for j in range(i, len(idx)):
                sums.append(idx[i] + idx[j])
        
        if not sums:
            return True
        
        sums_arr = np.array(sums)
        # Count occurrences of each sum
        # Use bincount for efficiency
        counts = np.bincount(sums_arr, minlength=max_sums + 1)
        return np.max(counts) <= 2
    
    def violation_count(arr):
        """Return the number of sums that exceed the limit of 2 representations."""
        idx = np.nonzero(arr[1:])[0] + 1
        if len(idx) == 0:
            return 0
        
        sums = []
        for i in range(len(idx)):
            for j in range(i, len(idx)):
                sums.append(idx[i] + idx[j])
        
        if not sums:
            return 0
        
        sums_arr = np.array(sums)
        counts = np.bincount(sums_arr, minlength=max_sums + 1)
        # Count how many sums have more than 2 representations
        return int(np.sum(counts > 2))
    
    def sum_conflicts(arr):
        """Return dict of sum -> count for sums with count > 2."""
        idx = np.nonzero(arr[1:])[0] + 1
        if len(idx) == 0:
            return {}
        
        sums = []
        for i in range(len(idx)):
            for j in range(i, len(idx)):
                sums.append(idx[i] + idx[j])
        
        if not sums:
            return {}
        
        sums_arr = np.array(sums)
        counts = np.bincount(sums_arr, minlength=max_sums + 1)
        conflicts = {}
        for s in range(2, max_sums + 1):
            if counts[s] > 2:
                conflicts[s] = counts[s]
        return conflicts
    
    # Start with a random dense initialization
    np.random.seed(42)
    
    best_size = 0
    best_arr = np.zeros(N + 1, dtype=bool)
    
    # Try multiple random starts
    for trial in range(5):
        # Start with ~700 random elements
        target_size = 700
        initial_indices = np.random.choice(N, size=target_size, replace=False) + 1
        arr = np.zeros(N + 1, dtype=bool)
        arr[initial_indices] = True
        
        # Initial cleanup: remove elements that cause the most conflicts
        # Iterate to reduce violations
        for _ in range(100):
            vc = violation_count(arr)
            if vc == 0:
                break
            
            # Find elements to remove - try removing the one that reduces violations most
            # But that's expensive. Instead, just remove random violating elements.
            # Get all pairs that contribute to over-represented sums
            idx = np.nonzero(arr[1:])[0] + 1
            if len(idx) == 0:
                break
            
            # Compute conflicts
            sums = []
            pair_indices = []  # (i, j) indices into idx
            for i in range(len(idx)):
                for j in range(i, len(idx)):
                    sums.append(idx[i] + idx[j])
                    pair_indices.append((i, j))
            
            if not sums:
                break
            
            sums_arr = np.array(sums)
            counts = np.bincount(sums_arr, minlength=max_sums + 1)
            
            # Find sums with count > 2
            over_sums = np.nonzero(counts > 2)[0]
            if len(over_sums) == 0:
                break
            
            # For each over-represented sum, we need to remove at least one pair
            # Strategy: pick a random over-represented sum, pick a random pair for it,
            # and remove one of the elements in that pair
            removed_any = False
            for _ in range(50):  # Try up to 50 removals per iteration
                if not over_sums or len(over_sums) == 0:
                    break
                s = int(over_sums[np.random.randint(len(over_sums))])
                
                # Find all pairs that sum to s
                candidate_pairs = []
                for pi, (i, j) in enumerate(pair_indices):
                    if sums[pi] == s:
                        candidate_pairs.append((i, j))
                
                if not candidate_pairs:
                    break
                
                # Pick a random pair and remove one element
                i, j = candidate_pairs[np.random.randint(len(candidate_pairs))]
                # Remove idx[i] (the smaller one, or random)
                elem_to_remove = idx[i]
                arr[elem_to_remove] = False
                removed_any = True
                
                # Recompute
                idx = np.nonzero(arr[1:])[0] + 1
                if len(idx) == 0:
                    break
                
                sums = []
                pair_indices = []
                for i in range(len(idx)):
                    for j in range(i, len(idx)):
                        sums.append(idx[i] + idx[j])
                        pair_indices.append((i, j))
                
                if not sums:
                    break
                
                sums_arr = np.array(sums)
                counts = np.bincount(sums_arr, minlength=max_sums + 1)
                over_sums = np.nonzero(counts > 2)[0]
            
            if not removed_any:
                break
        
        # Now try to add elements back
        # Local search: try adding random elements
        for _ in range(2000):
            # Try adding a random element
            candidates = np.nonzero(~arr[1:])[0] + 1
            if len(candidates) == 0:
                break
            
            x = candidates[np.random.randint(len(candidates))]
            arr[x] = True
            
            if is_valid(arr):
                # Keep it
                continue
            else:
                # Revert
                arr[x] = False
        
        # Record best
        size = int(np.sum(arr[1:]))
        if size > best_size:
            best_size = size
            best_arr = arr.copy()
        
        # Try to improve further with more aggressive local search
        for _ in range(500):
            candidates = np.nonzero(~arr[1:])[0] + 1
            if len(candidates) == 0:
                break
            
            x = candidates[np.random.randint(len(candidates))]
            arr[x] = True
            
            if is_valid(arr):
                continue
            else:
                arr[x] = False
        
        size = int(np.sum(arr[1:]))
        if size > best_size:
            best_size = size
            best_arr = arr.copy()
    
    # Final greedy pass: try adding elements in order
    for x in range(1, N + 1):
        if not best_arr[x]:
            best_arr[x] = True
            if is_valid(best_arr):
                continue
            else:
                best_arr[x] = False
    
    result = [i for i in range(1, N + 1) if best_arr[i]]
    return result

# EVOLVE-BLOCK-END
