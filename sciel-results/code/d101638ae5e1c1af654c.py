import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precomputation
    points = range(v)
    
    # Generate all t-subsets and map to IDs
    t_subs = list(itertools.combinations(points, t))
    t_sub_to_id = {s: i for i, s in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(k_blocks)
    
    # For each k-block, find which t-subsets it covers
    # block_to_subs[i] is a list of t-subset IDs covered by block i
    block_to_subs = []
    for b in k_blocks:
        subs_in_b = []
        for ts in itertools.combinations(b, t):
            subs_in_b.append(t_sub_to_id[ts])
        block_to_subs.append(subs_in_b)
        
    # Inverse index: for each t-subset, which k-blocks cover it?
    # sub_to_blocks[j] is a list of k-block IDs that cover t-subset j
    sub_to_blocks = [[] for _ in range(num_t_subs)]
    for b_idx, subs in enumerate(block_to_subs):
        for s_idx in subs:
            sub_to_blocks[s_idx].append(b_idx)
            
    # Convert sub_to_blocks to numpy arrays for faster iteration if needed?
    # Lists are probably fine for 455 elements.
    
    best_solution = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        random.seed(seed_counter)
        
        uncovered = set(range(num_t_subs))
        # counts[i] is the number of currently uncovered t-subsets covered by block i
        counts = np.array([len(subs) for subs in block_to_subs], dtype=np.int32)
        
        solution = []
        
        while uncovered:
            # Find block with max count
            # To add randomization, we can pick from top K
            # Or just pick argmax, but if ties, pick randomly.
            
            max_count = np.max(counts)
            if max_count == 0:
                # Should not happen if uncovered is not empty
                break
                
            # Find all blocks with max_count
            candidates = np.where(counts == max_count)[0]
            
            # Pick one randomly
            if len(candidates) > 0:
                chosen_block = candidates[random.randint(0, len(candidates) - 1)]
            else:
                break
                
            solution.append(chosen_block)
            
            # Update counts
            for s_idx in block_to_subs[chosen_block]:
                if s_idx in uncovered:
                    uncovered.remove(s_idx)
                    # Decrement counts for all blocks covering this s_idx
                    for b_idx in sub_to_blocks[s_idx]:
                        counts[b_idx] -= 1
                        
        # Post-processing: Local search to remove unnecessary blocks
        # We check if we can remove any block from the solution
        # We do this in a loop until no more blocks can be removed
        # To make it faster, we just do one pass or a few passes
        
        # Efficient check: 
        # For a block B to be removable, all t-subsets in B must be covered by other blocks in the solution.
        # Let's build a coverage count for t-subsets based on the current solution.
        
        # Re-calculate coverage counts for the solution
        # This is O(num_blocks_in_solution * 35)
        sub_coverage = np.zeros(num_t_subs, dtype=np.int32)
        for b_idx in solution:
            for s_idx in block_to_subs[b_idx]:
                sub_coverage[s_idx] += 1
                
        # Try to remove blocks
        # We can try removing blocks in reverse order of addition or random order
        # Let's try random order for diversity
        sol_indices = list(range(len(solution)))
        random.shuffle(sol_indices)
        
        changed = True
        while changed:
            changed = False
            for idx in sol_indices:
                b_idx = solution[idx]
                # Check if all subsets in b_idx are covered by other blocks
                # i.e., sub_coverage[s] > 1 for all s in b_idx
                # If yes, we can remove b_idx
                removable = True
                for s_idx in block_to_subs[b_idx]:
                    if sub_coverage[s_idx] <= 1:
                        removable = False
                        break
                        
                if removable:
                    # Remove b_idx
                    for s_idx in block_to_subs[b_idx]:
                        sub_coverage[s_idx] -= 1
                    solution.pop(idx)
                    # Adjust indices? 
                    # Since we are using indices into a list that changes, 
                    # it's safer to rebuild the list or use a set of block IDs.
                    # Let's use a set of block IDs for the solution.
                    # Actually, let's restart the local search logic with a set.
                    break # Restart the while loop with new list length
            
        # The above local search logic with indices is tricky if we pop.
        # Let's implement a cleaner local search using a set of block IDs.
        
        # Re-implement local search cleanly
        sol_set = set(solution)
        
        # Recalculate coverage
        sub_coverage = np.zeros(num_t_subs, dtype=np.int32)
        for b_idx in sol_set:
            for s_idx in block_to_subs[b_idx]:
                sub_coverage[s_idx] += 1
                
        # Try to remove
        # We iterate through a copy of the set
        # To avoid O(N^2) behavior, we just do a few passes
        for _ in range(5): # Max 5 passes
            removed_any = False
            for b_idx in list(sol_set):
                removable = True
                for s_idx in block_to_subs[b_idx]:
                    if sub_coverage[s_idx] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove
                    for s_idx in block_to_subs[b_idx]:
                        sub_coverage[s_idx] -= 1
                    sol_set.remove(b_idx)
                    removed_any = True
            if not removed_any:
                break
        
        if len(sol_set) < best_size:
            best_size = len(sol_set)
            best_solution = list(sol_set)
            
    # Convert block IDs back to blocks (tuples/lists)
    if best_solution is None:
        # Fallback: return empty or error? 
        # Should not happen.
        return []
        
    final_blocks = [list(k_blocks[b_idx]) for b_idx in best_solution]
    return final_blocks
