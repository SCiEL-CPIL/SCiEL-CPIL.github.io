import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 90s kill
    
    max_val = 4096
    max_sum = 3 * max_val + 1  # 12289
    # used_sums[i] is True if sum i is already taken
    used_sums = np.zeros(max_sum, dtype=bool)
    
    # Precompute pair sums for all possible pairs (i, j) with i <= j in [1, 4096]
    # This is O(N^2) memory which is ~16M entries, too large for a dense array of bools?
    # 4096*4096/2 = 8M. 8M bools is 8MB. This is fine.
    # However, we don't need ALL pairs, only those that might be relevant.
    # Actually, for the check "x + s_i + s_j", we need to know if x + (s_i+s_j) is in used_sums.
    # We can just iterate over pairs in S.
    
    def check_and_add(S, x, used_sums):
        """
        Check if x can be added to S.
        S is a numpy array of sorted integers.
        used_sums is a boolean array of size max_sum.
        Returns (True, new_sums_array) if valid, (False, None) otherwise.
        """
        k = len(S)
        
        # Type 1: 3*x
        s1 = 3 * x
        if used_sums[s1]:
            return False, None
        
        # Type 2: 2*x + s_i
        s2 = 2 * x + S
        if np.any(used_sums[s2]):
            return False, None
        
        # Type 3: x + s_i + s_j (i <= j)
        # Generate all pair sums s_i + s_j
        # Using broadcasting might be faster than loops for small k
        if k > 0:
            # Create indices for upper triangle
            i_idx, j_idx = np.triu_indices(k)
            pair_sums = S[i_idx] + S[j_idx]
            s3 = x + pair_sums
            
            # Check for duplicates within s3 and against used_sums
            # Note: s2 and s3 might overlap with s1, s2, s3 internally?
            # We need to ensure all new sums are unique among themselves and not in used_sums.
            
            # Combine all new sums
            all_new_sums = np.concatenate(([s1], s2, s3))
            
            # Check for internal duplicates
            unique_new = np.unique(all_new_sums)
            if len(unique_new) != len(all_new_sums):
                return False, None
            
            # Check against used_sums
            if np.any(used_sums[all_new_sums]):
                return False, None
                
            return True, all_new_sums
        else:
            # S is empty
            return True, np.array([s1])

    def run_greedy(order):
        """
        Greedily add elements from 'order' to the set.
        Returns the selected set and the used_sums array.
        """
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums = np.zeros(max_sum, dtype=bool)
        
        for x in order:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            valid, new_sums = check_and_add(S, x, used_sums)
            if valid:
                selected.append(x)
                S = np.append(S, x)
                used_sums[new_sums] = True
                
        return selected, used_sums

    def local_search_improve(selected, used_sums):
        """
        Attempt to improve the set by swapping one element out and adding two new ones.
        This is a heuristic to escape local optima.
        """
        if time.time() - start_time > DEADLINE - 5:
            return selected, used_sums
            
        # Convert to set for fast lookup
        S_set = set(selected)
        S_arr = np.array(selected, dtype=np.int32)
        
        # We will try to remove each element one by one and see if we can add more.
        # To be efficient, we only try a few candidates.
        # We'll iterate through a random subset of the current elements to remove.
        
        best_improved = None
        best_size = len(selected)
        
        # Shuffle the indices to try removals in random order
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            elem_to_remove = selected[idx]
            
            # Create a new state without this element
            # This is expensive if done from scratch, so we'll simulate.
            # We need to recalculate used_sums without this element.
            # This is O(N^3) if we redo everything.
            # Instead, let's just try to add new elements to the *current* set, 
            # but that's what greedy did.
            # The "2-for-1" move is complex.
            # Let's stick to a simpler local search: 
            # Try to add elements from a list of "almost valid" candidates.
            # Actually, the greedy algorithm already scans all candidates.
            # The issue is that the *order* matters.
            
            # A better local search:
            # 1. Pick an element to remove.
            # 2. Re-run greedy on the remaining elements + new candidates?
            # This is too slow.
            
            # Alternative: 
            # We can try to replace a "large" element with a "smaller" one that fits better.
            # Or, we can just rely on more randomized restarts with different biases.
            
            pass
            
        return selected, used_sums

    # Strategy:
    # 1. Run sequential greedy (1, 2, 3...)
    # 2. Run reverse greedy (4096, 4095, ...)
    # 3. Run many random shuffles.
    # 4. Try "biased" shuffles: e.g., prefer smaller numbers, or prefer numbers with certain properties.
    
    best_set = []
    best_size = 0
    
    # 1. Sequential
    res_seq, _ = run_greedy(range(1, max_val + 1))
    if len(res_seq) > best_size:
        best_set = res_seq
        best_size = len(res_seq)
        
    # 2. Reverse
    res_rev, _ = run_greedy(range(max_val, 0, -1))
    if len(res_rev) > best_size:
        best_set = res_rev
        best_size = len(res_rev)
        
    # 3. Randomized Restarts
    # We can also try to bias the random order.
    # For B_h sequences, it's often beneficial to pick smaller numbers first to pack densely.
    # But sometimes picking a large number early opens up space for many small numbers.
    
    while time.time() - start_time < DEADLINE - 5:
        # Create a random order
        # To make it more effective, we can mix strategies.
        # E.g., 50% uniform random, 50% biased towards smaller numbers.
        
        if random.random() < 0.5:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
        else:
            # Biased: sort by random key, but favor smaller numbers
            # e.g., key = x * random.uniform(0.5, 1.5)
            candidates = list(range(1, max_val + 1))
            keys = [x * random.uniform(0.8, 1.2) for x in candidates]
            candidates = [x for _, x in sorted(zip(keys, candidates))]
            
        res_rand, _ = run_greedy(candidates)
        if len(res_rand) > best_size:
            best_set = res_rand
            best_size = len(res_rand)
            
    return best_set
