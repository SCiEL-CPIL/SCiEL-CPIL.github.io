import time
import random
import itertools
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute masks as integers for fast bitwise ops
    triple_masks = [0] * num_triples
    triple_pairs = [None] * num_triples
    for i, t in enumerate(all_triples):
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[i] = m
        a, b, c = t
        triple_pairs[i] = [(a,b), (a,c), (b,c)]
        
    # Map pair to index for O(1) conflict check
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_to_idx[(i,j)] = idx
            idx += 1
    num_pairs = idx
            
    def get_indep_number_numpy(system_masks):
        """
        Computes the independence number using NumPy vectorization.
        A set S is independent if it does not contain any triple from system_masks.
        """
        if not system_masks:
            return n
            
        # Create array of all subsets 0..2^16-1
        # We need to check if any triple mask is a subset of S.
        # triple_mask & S == triple_mask  =>  S contains triple_mask
        
        # Optimization: 
        # Instead of checking all triples for all subsets (too slow),
        # we can mark all subsets that contain ANY triple as "not independent".
        
        # There are 65536 subsets. 
        # Let's create a boolean array of size 65536, initially True (independent).
        is_ind = np.ones(1 << n, dtype=bool)
        
        # For each triple, mark all its supersets as False.
        # Supersets of mask m: S such that (S & m) == m.
        # This is equivalent to: S = m | sub, where sub is a subset of ~m.
        
        # Iterating over all triples and all their supersets in Python is slow.
        # Vectorized approach:
        # Let subsets = np.arange(1 << n)
        # For a specific triple mask m, the supersets are those S where (S & m) == m.
        # We can compute this condition for all S at once for one triple, but doing it 
        # for ~40 triples is 40 * 65536 operations, which is fine.
        
        subsets = np.arange(1 << n)
        
        for m in system_masks:
            # Check which subsets contain m
            # (subsets & m) == m
            # This creates a boolean array of size 65536
            contains_m = (subsets & m) == m
            is_ind[contains_m] = False
            
        # Find the max popcount of subsets where is_ind is True
        # We only need to check subsets where is_ind is True.
        # To find max popcount, we can iterate through sizes from n down to 0?
        # Or just compute popcount for all and take max where is_ind is True.
        
        # Computing popcount for 65536 integers is fast in numpy?
        # np.bitwise_count is not standard in older numpy, but we can use lookup or bin.
        # Better: Precompute popcounts for all 16-bit integers? 
        # Or just use a loop over the 65536 elements? 65536 iterations in Python is ~10-20ms.
        # Let's try to be efficient.
        
        # Method: Iterate k from n down to 0.
        # If there exists a subset of size k that is independent, return k.
        # How to check existence of independent set of size k?
        # Filter subsets by popcount == k, then check if any is in is_ind.
        
        # Precomputing popcounts for 0..65535
        popcounts = np.zeros(1 << n, dtype=np.uint8)
        # Efficient popcount calculation for 16-bit integers
        # Using lookup table for bytes
        # Or just use bin(i).count('1') in a list comprehension? 65536 * ~50ns = 3ms. Acceptable.
        # Let's use a faster method if possible.
        # Actually, we can just iterate through the `is_ind` array indices.
        
        max_size = 0
        # To avoid iterating all 65536 in Python if we find a high one early?
        # No, we need the max.
        
        # Let's group by popcount.
        # Since n=16, max size is 16.
        # We can check sizes from 16 down to 0.
        
        # Precompute list of subsets by popcount?
        # subsets_by_popcount[k] = list of subsets with k bits set.
        # This takes space but is fast to build once.
        
        # Building subsets_by_popcount once outside the function would be better,
        # but we can't modify global state easily without risk. 
        # Let's build it locally if needed, or just use a simple loop.
        
        # Simple loop:
        # For each i where is_ind[i] is True, compute popcount.
        # This is slow if many are True.
        
        # Better:
        # Check if any subset of size 16 is independent? Only the full set.
        # Check size 15?
        
        # Let's precompute popcounts for all 65536 values using a fast method.
        # popcount[i] = popcount[i >> 1] + (i & 1)
        # This is O(2^16) which is very fast in Python if optimized? 
        # 65536 iterations is negligible.
        
        pc = [0] * (1 << n)
        for i in range(1, 1 << n):
            pc[i] = pc[i >> 1] + (i & 1)
        
        # Now find max pc[i] where is_ind[i] is True
        # is_ind is a numpy bool array.
        # We can get the indices of True values.
        # true_indices = np.where(is_ind)[0]
        # Then find max(pc[i] for i in true_indices)
        
        # To avoid converting to Python list which is slow:
        # We can create an array of popcounts and mask it.
        pc_arr = np.array(pc, dtype=np.uint8)
        
        # Mask out non-independent sets
        # Set pc_arr to 0 where is_ind is False
        pc_arr[~is_ind] = 0
        
        return int(np.max(pc_arr))

    # Precompute subsets by popcount for faster max search?
    # Actually, the numpy method above is ~50-100ms per evaluation.
    # We can afford ~1000 evaluations in 100s.
    
    def backtrack_search(max_depth=40):
        """
        Attempts to find a large linear triple system using backtracking.
        Returns a list of triple indices.
        """
        used_pairs = set()
        current_system = []
        
        # Order triples by some heuristic? 
        # Random order with restarts is better for finding diverse large packings.
        
        def dfs(depth, remaining_limit):
            if time.time() > deadline - 5.0:
                return False
            if depth == max_depth:
                return True
            if remaining_limit == 0:
                return False
                
            # Choose a triple to add
            # To speed up, we can filter available triples first.
            # But filtering 560 triples every step is expensive.
            # Instead, iterate and check.
            
            # Heuristic: Try to add triples that conflict with fewest future options?
            # For now, random order passed in.
            
            # This generic DFS is too slow without good pruning/ordering.
            # Let's use a different approach: Greedy + Local Search.
            return False
            
        return current_system

    def greedy_construct(random_seed=None):
        """
        Constructs a maximal linear triple system using greedy algorithm.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        current_system = []
        used_pairs = set()
        
        for idx_t in indices:
            pairs = triple_pairs[idx_t]
            
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
                    
            if not conflict:
                current_system.append(idx_t)
                for p in pairs:
                    used_pairs.add(pair_to_idx[p])
                    
        return current_system

    def local_search(system, max_iterations=500):
        """
        Improves the system by swapping triples or adding/removing.
        Objective: Minimize independence number.
        """
        current_system = system[:]
        current_indep = get_indep_number_numpy([triple_masks[i] for i in current_system])
        best_indep = current_indep
        best_system = current_system[:]
        
        used_pairs = set()
        for idx_t in current_system:
            for p in triple_pairs[idx_t]:
                used_pairs.add(pair_to_idx[p])
                
        # Build a set of available triples (not in current system, no conflict)
        # This is expensive to rebuild every time. 
        # Instead, try random swaps.
        
        for i in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            # Pick a random triple from current system to remove
            if not current_system:
                break
                
            rem_idx_pos = random.randint(0, len(current_system) - 1)
            rem_idx = current_system[rem_idx_pos]
            
            # Remove it
            current_system.pop(rem_idx_pos)
            rem_pairs = triple_pairs[rem_idx]
            for p in rem_pairs:
                used_pairs.discard(pair_to_idx[p])
                
            # Try to add a new triple
            # Pick a random triple that is now valid
            # To be efficient, we can pick a few random candidates
            found_new = False
            for _ in range(20): # Try 20 random candidates
                cand_idx = random.randint(0, num_triples - 1)
                if cand_idx in current_system:
                    continue
                cand_pairs = triple_pairs[cand_idx]
                valid = True
                for p in cand_pairs:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    # Add it
                    current_system.append(cand_idx)
                    for p in cand_pairs:
                        used_pairs.add(pair_to_idx[p])
                    found_new = True
                    break
                    
            if not found_new:
                # If we couldn't add, put the old one back?
                # Or keep it removed? 
                # Keeping it removed might allow a different structure later.
                # But we want to maintain maximality roughly.
                # Let's put it back if no better found.
                current_system.append(rem_idx)
                for p in rem_pairs:
                    used_pairs.add(pair_to_idx[p])
                continue
                
            # Evaluate new system
            new_indep = get_indep_number_numpy([triple_masks[i] for i in current_system])
            
            if new_indep < best_indep:
                best_indep = new_indep
                best_system = current_system[:]
                
            # Accept move if it doesn't increase independence number too much?
            # Or just hill climb.
            # If new_indep <= current_indep, accept.
            # Else, accept with some probability (simulated annealing)?
            # For simplicity, hill climb with restarts.
            if new_indep > current_indep:
                # Revert
                current_system.pop()
                for p in cand_pairs:
                    used_pairs.discard(pair_to_idx[p])
                current_system.append(rem_idx)
                for p in rem_pairs:
                    used_pairs.add(pair_to_idx[p])
            else:
                current_indep = new_indep

        return best_system, best_indep

    best_system_global = None
    best_indep_global = n + 1
    
    # Phase 1: Find a large packing
    # Run greedy multiple times to find a maximal packing with many triples
    max_greedy_restarts = 200
    for i in range(max_greedy_restarts):
        if time.time() > deadline - 30.0:
            break
        sys_greedy = greedy_construct(i)
        # We want to maximize size first to have more constraints
        # But the score is independence number.
        # A larger packing generally has a smaller independence number.
        # Let's just run local search on these.
        
        # To save time, only evaluate if size is promising
        if len(sys_greedy) < 30:
            continue
            
        sys_optimized, indep = local_search(sys_greedy, max_iterations=100)
        
        if indep < best_indep_global:
            best_indep_global = indep
            best_system_global = sys_optimized
            
        if best_indep_global <= 6:
            break

    # Phase 2: If we have time, do more local search from the best found
    if best_system_global and time.time() < deadline - 15.0:
        sys_optimized, indep = local_search(best_system_global, max_iterations=500)
        if indep < best_indep_global:
            best_indep_global = indep
            best_system_global = sys_optimized

    if best_system_global is None:
        # Fallback: return empty or minimal
        return []

    return [list(all_triples[i]) for i in best_system_global]

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found system with {len(res)} triples")
    # Calculate indep for verification
    masks = [0]
    for t in res:
        m = 0
        for p in t:
            m |= 1 << p
        masks.append(m)
    # This is just for local debugging, not in the final flow
