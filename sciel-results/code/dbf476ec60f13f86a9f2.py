import itertools
import time
import random

def construct(v, k, t):
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Map all t-subsets to indices
    # v=19, t=4 -> 3876 subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Initial uncovered mask: all bits set
    # Using a Python int as a bitset
    full_mask = (1 << num_t_subsets) - 1
    
    # 2. Pre-compute masks for all k-subsets
    # v=19, k=7 -> 50388 blocks
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    block_tuples = []
    
    for block in all_k_subsets:
        mask = 0
        # For each t-subset within this block, set the bit
        # Number of t-subsets in a block is C(k, t) = C(7, 4) = 35
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        block_tuples.append(block)
        
    num_blocks = len(all_k_subsets)
    
    # 3. Greedy Set Cover with Randomized Tie-Breaking
    # We will run multiple restarts to try to find a better solution
    best_blocks = None
    best_count = float('inf')
    
    # Estimate max iterations for greedy
    # Lower bound is ~111, known best is 151. Let's allow up to 200.
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Find the block with max gain
            # We iterate through all blocks
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if uncovered is not 0 and blocks exist
            
            # Pick randomly from best candidates
            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
            # Check time
            if time.time() > deadline - 5:
                return None, len(selected_indices)
                
        return selected_indices, len(selected_indices)

    # Run multiple restarts
    # Each restart takes roughly 150 * 50388 * (bit_count cost)
    # Let's estimate time per restart.
    # 7.5M iterations. If each takes 0.1us (optimistic), 0.75s.
    # If each takes 1us, 7.5s.
    # Let's try to run as many as possible, but at least a few.
    
    num_restarts = 0
    while time.time() < deadline - 10:
        result, count = run_greedy(num_restarts)
        if result is not None:
            if count < best_count:
                best_count = count
                best_blocks = [block_tuples[i] for i in result]
        num_restarts += 1
        
        # If we've found the lower bound or close to it, we can stop
        # Lower bound is ceil(3876/35) = 111
        if best_count <= 115:
            break

    # 4. Local Search / Cleanup: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks.
    if best_blocks:
        # Convert to set of tuples for fast lookup? No, we need to check coverage.
        # Re-compute coverage mask for the current solution
        # This is O(num_blocks * 35)
        
        # Actually, we can just try to remove each block one by one
        # and check if coverage is still complete.
        # Checking coverage:
        # 1. Compute union of masks of all blocks except the one being tested.
        # 2. Check if union covers full_mask.
        
        # This is O(N^2 * bit_op). N is ~150. 150^2 = 22500.
        # 22500 * bit_count on 3876-bit int. Very fast.
        
        current_blocks = list(best_blocks)
        current_indices = [block_tuples.index(b) for b in current_blocks] # This is slow, O(N*M)
        
        # Better: keep the indices
        # Let's restart the cleanup with indices
        # We lost the indices in the list comprehension above. Let's refactor.
        
        pass # We'll do a proper cleanup pass below

    # Proper Cleanup Pass
    # We need the indices of the selected blocks
    # Let's re-run the best greedy to get indices, or store them.
    # It's better to store indices in run_greedy.
    
    # Let's modify run_greedy to return indices
    # And then do cleanup on indices.
    
    # Since I already wrote run_greedy to return tuples, let's just do a cleanup on tuples
    # by converting them back to indices efficiently.
    # Or, better, just modify the logic to keep indices.
    
    # Let's do a quick cleanup on the best_blocks list found.
    # We need to know which t-subsets are covered by which blocks.
    # We can re-calculate the "covered mask" for the current set of blocks.
    
    if best_blocks:
        # Create a mapping from block tuple to index for fast lookup
        block_to_idx = {b: i for i, b in enumerate(all_k_subsets)}
        
        current_indices = [block_to_idx[b] for b in best_blocks]
        
        # Try to remove each block
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_indices = []
            for i, idx in enumerate(current_indices):
                # Temporarily remove idx
                remaining_indices = [j for j in range(len(current_indices)) if j != i]
                
                # Calculate coverage of remaining
                cov_mask = 0
                for j in remaining_indices:
                    cov_mask |= block_masks[current_indices[j]]
                    
                if cov_mask == full_mask:
                    # It was redundant, remove it
                    changed = True
                    # Don't add idx to new_indices
                else:
                    new_indices.append(idx)
            
            current_indices = new_indices
            
        # Convert back to blocks
        best_blocks = [all_k_subsets[i] for i in current_indices]
        
    if best_blocks is None:
        # Fallback to a simple greedy if something went wrong
        # This should not happen
        return []
        
    return best_blocks
