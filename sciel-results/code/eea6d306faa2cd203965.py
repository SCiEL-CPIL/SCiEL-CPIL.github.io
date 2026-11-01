<<<<<<< SEARCH
import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s kill
=======
import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
>>>>>>> REPLACE
<<<<<<< SEARCH
    while np.any(uncovered):
        if time.time() > deadline - 60.0:
            break
=======
    while np.any(uncovered):
        if time.time() > deadline - 80.0:
            break
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Main optimization loop
    random.seed(42)
    no_improvement_count = 0
    
    def repair_greedy(blocks, counts):
        """Add blocks greedily to cover any uncovered t-subsets."""
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            return blocks, counts
        
        # Build a set of current blocks for fast lookup
        block_set = set(blocks)
        
        # Calculate gain for each block: number of currently uncovered t-subsets it covers
        gains = np.zeros(n_blocks, dtype=np.int32)
        for s in uncovered:
            gains[tsub_to_blocks_np[s]] += 1
        
        while len(uncovered) > 0:
            if time.time() > deadline - 15.0:
                break
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if best_idx in block_set:
                # Should not happen if we update gains correctly, but safety
                gains[best_idx] = -1
                continue
                
            blocks.append(best_idx)
            block_set.add(best_idx)
            
            # Update counts and gains
            for s in block_to_tsubs[best_idx]:
                if counts[s] == 0:
                    counts[s] = 1
                    # This t-subset is now covered, so it no longer contributes to gains
                    # But we are using a static 'uncovered' list? No, we need to update uncovered.
                    # For efficiency in repair, we can just recompute uncovered if it's small.
                    pass
            
            # Recompute uncovered and gains for next iteration if set is small
            # Or just update counts and check if all covered
            # For speed, let's just update counts and rely on the outer loop to check validity?
            # No, repair must ensure validity.
            
            # Efficient update:
            # We added best_idx.
            # Any t-subset in best_idx that was 0 is now 1.
            # Gains for all blocks covering those t-subsets should decrease?
            # Actually, gains were calculated for the *initial* set of uncovered.
            # As we add blocks, the set of uncovered shrinks.
            # We should re-calculate gains for the remaining uncovered.
            
            remaining_uncovered = np.where(counts == 0)[0]
            if len(remaining_uncovered) == 0:
                break
            
            gains = np.zeros(n_blocks, dtype=np.int32)
            for s in remaining_uncovered:
                gains[tsub_to_blocks_np[s]] += 1
        
        return blocks, counts
>>>>>>> REPLACE
<<<<<<< SEARCH
    while time.time() < deadline - 10.0:
        # Step 1: Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(current_blocks)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count > 50:
            # Restart from best with different perturbation
            current_blocks = list(best_blocks)
            cover_counts = np.zeros(n_tsubs, dtype=np.int32)
            for idx in current_blocks:
                for s in block_to_tsubs[idx]:
                    cover_counts[s] += 1
            no_improvement_count = 0
        
        # Step 2: Try criticality-based swaps
        # Pick a random block to try removing
        if len(current_blocks) > 1:
            random.shuffle(current_blocks)
            for block_to_try in current_blocks[:min(5, len(current_blocks))]:
                success, new_blocks, new_counts = try_swap_remove(
                    current_blocks, cover_counts, block_to_try
                )
                if success:
                    # Check if we can further reduce
                    new_blocks = reduce_redundant(new_blocks, new_counts)
                    
                    if len(new_blocks) < len(current_blocks):
                        current_blocks = new_blocks
                        cover_counts = new_counts
                        if len(current_blocks) < best_size:
                            best_blocks = list(current_blocks)
                            best_size = len(current_blocks)
                            no_improvement_count = 0
                        break
                    elif len(new_blocks) == len(current_blocks):
                        # Same size, but might enable future reductions
                        # Accept with some probability
                        if random.random() < 0.3:
                            current_blocks = new_blocks
                            cover_counts = new_counts
        
        # Step 3: Kick - Remove a random block and repair (Packing-First style perturbation)
        if time.time() < deadline - 12.0 and len(current_blocks) > 10:
            # Pick a random block to remove
            kick_idx = random.choice(current_blocks)
            
            # Remove it
            temp_blocks = [b for b in current_blocks if b != kick_idx]
            temp_counts = cover_counts.copy()
            for s in block_to_tsubs[kick_idx]:
                temp_counts[s] -= 1
            
            # Repair the holes
            repaired_blocks, repaired_counts = repair_greedy(temp_blocks, temp_counts)
            
            # Check if valid and better
            if np.all(repaired_counts > 0):
                # Reduce redundancy in the repaired set
                repaired_blocks = reduce_redundant(repaired_blocks, repaired_counts)
                
                if len(repaired_blocks) < len(current_blocks):
                    current_blocks = repaired_blocks
                    cover_counts = repaired_counts
                    if len(current_blocks) < best_size:
                        best_blocks = list(current_blocks)
                        best_size = len(current_blocks)
                        no_improvement_count = 0
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Final polish: aggressive redundancy removal and single-block swaps
    if time.time() < deadline - 5.0:
        final_blocks = list(best_blocks)
        final_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in final_blocks:
            for s in block_to_tsubs[idx]:
                final_counts[s] += 1
        
        # Multiple passes of redundancy removal
        for _ in range(3):
            final_blocks = reduce_redundant(final_blocks, final_counts)
        
        # Try single-block removals one more time
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for block_to_try in final_blocks[:]:
                success, new_blocks, new_counts = try_swap_remove(
                    final_blocks, final_counts, block_to_try
                )
                if success and len(new_blocks) < len(final_blocks):
                    final_blocks = new_blocks
                    final_counts = new_counts
                    improved = True
                    break
    
    # Use the best of best_blocks and final_blocks
    if len(final_blocks) < len(best_blocks):
        best_blocks = final_blocks
    
    # Convert to actual blocks
    return [all_kblocks[i] for i in best_blocks]
=======
    # Final polish: aggressive redundancy removal and single-block swaps
    if time.time() < deadline - 5.0:
        final_blocks = list(best_blocks)
        final_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in final_blocks:
            for s in block_to_tsubs[idx]:
                final_counts[s] += 1
        
        # Multiple passes of redundancy removal
        for _ in range(3):
            final_blocks = reduce_redundant(final_blocks, final_counts)
        
        # Try single-block removals one more time
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for block_to_try in final_blocks[:]:
                success, new_blocks, new_counts = try_swap_remove(
                    final_blocks, final_counts, block_to_try
                )
                if success and len(new_blocks) < len(final_blocks):
                    final_blocks = new_blocks
                    final_counts = new_counts
                    improved = True
                    break
    
    # Use the best of best_blocks and final_blocks
    if len(final_blocks) < len(best_blocks):
        best_blocks = final_blocks
    
    # Convert to actual blocks
    return [all_kblocks[i] for i in best_blocks]
>>>>>>> REPLACE
