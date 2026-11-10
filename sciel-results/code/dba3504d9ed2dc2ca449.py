"""
Optimized covering design solver for C(v, k, t) using bitsets and simulated annealing.

Strategy:
1. Precompute bitmasks for all possible k-subsets.
2. Use a fast greedy algorithm to get an initial valid covering.
3. Apply Simulated Annealing to minimize the number of blocks, allowing temporary
   invalid states (uncovered subsets) to escape local minima.
"""

import time
import random
import itertools
import numpy as np

# Global constants and cache
_DEADLINE = 120.0
_MARGIN = 5.0

# Cache for precomputed data per (v, k, t)
_precomputed = {}

def _get_precomputed(v, k, t):
    """
    Precompute all k-subsets and their coverage bitmasks.
    Returns:
        blocks: List of tuples (the k-subsets)
        masks: Numpy array of shape (N_blocks, n_words) where each row is the bitmask for that block.
        t_subset_indices: Dict mapping t-subset tuple to bit index.
        n_subsets: Total number of t-subsets.
    """
    key = (v, k, t)
    if key in _precomputed:
        return _precomputed[key]
        
    # Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    t_map = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Number of 64-bit words needed
    n_words = (n_subsets + 63) // 64
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(k_subsets)
    
    # Initialize masks
    masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    # Fill masks
    for i, block in enumerate(k_subsets):
        block_set = set(block)
        # Iterate through all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = t_map[t_sub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks[i, word_idx] |= (1 << bit_idx)
            
    _precomputed[key] = (k_subsets, masks, t_map, n_subsets)
    return _precomputed[key]

def _count_uncovered(covered_mask):
    """Count the number of zero bits in the covered_mask array."""
    # Invert bits within the valid range
    # The last word might have extra bits, so we need to be careful.
    # For simplicity, we assume n_subsets is handled by the mask structure.
    # A bit is 'covered' if it is 1.
    # We want count of 0s.
    # Invert: ~mask
    # Sum bits.
    # But ~mask will have 1s in the padding bits of the last word.
    # We need to mask out the padding.
    
    # Create a mask for valid bits
    # This is slightly expensive if done every time. 
    # Better: Precompute a 'valid_mask' array.
    # But for now, let's just count and subtract padding?
    # Or, simpler: 
    # total_bits = sum of popcount(~word)
    # padding_bits = (n_words * 64) - n_subsets
    # uncovered = total_bits - padding_bits
    
    # popcount in numpy:
    # np.bit_count is available in newer numpy? 
    # If not, we can use a lookup table or bin(x).count('1') which is slow.
    # Let's use a vectorized popcount trick or just sum bits.
    # Actually, `np.bit_count` is in NumPy 1.24+. Let's assume it's available or use a fallback.
    # To be safe, we'll use a helper.
    
    # Fallback popcount for uint64 array
    # x = x - ((x >> 1) & 0x5555555555555555)
    # x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    # x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    # x = (x * 0x0101010101010101) >> 56
    
    # However, doing this for every SA step might be slow if not optimized.
    # Alternative: Maintain `uncovered_count` incrementally?
    # In SA, we change one block at a time.
    # When we remove a block, we AND its mask out? No, we need to recompute coverage from scratch or use reference counts.
    # Reference counts are better for incremental updates.
    
    # Let's switch strategy for SA:
    # Maintain `counts` array of size n_subsets (int16 or int32).
    # `uncovered_count` = number of indices where counts == 0.
    # When adding/removing a block, update counts and adjust uncovered_count.
    
    pass

def _greedy_construct(v, k, t, blocks_list, masks, deadline):
    """
    Fast greedy construction using precomputed masks.
    """
    n_blocks = len(blocks_list)
    n_words = masks.shape[1]
    
    covered_mask = np.zeros(n_words, dtype=np.uint64)
    selected_indices = []
    start_time = time.time()
    
    while time.time() < deadline:
        # Check if all covered
        # We need to know if covered_mask has all 1s in the valid bits.
        # For greedy, we can just check if any gain is possible.
        
        # Calculate gain for all blocks
        # gain = popcount(mask & ~covered_mask)
        # To avoid popcount overhead, we can just sum the bits?
        # Or, we can just pick the block that covers the most *new* bits.
        
        # Vectorized gain calculation:
        # uncovered_bits = ~covered_mask
        # intersected = masks & uncovered_bits
        # gains = popcount(intersected)
        
        # Since popcount is the bottleneck, let's see if we can optimize.
        # For 50k blocks and 61 words, this is 3M operations. Numpy handles this fast.
        
        uncovered_bits = ~covered_mask
        intersected = masks & uncovered_bits
        
        # Popcount
        # Using a lookup table for bytes might be faster than bit manipulation in numpy?
        # Or just use np.bit_count if available.
        try:
            # NumPy 1.24+
            gains = np.bit_count(intersected).sum(axis=1)
        except AttributeError:
            # Fallback: sum of bits
            # This is slower.
            # Let's use a trick:
            # x = x - ((x >> 1) & 0x5555555555555555)
            # x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            # x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            # x = (x * 0x0101010101010101) >> 56
            x = intersected
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            x = (x * 0x0101010101010101) >> 56
            gains = x.sum(axis=1)

        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # No block can cover any new subsets. 
            # This implies we are stuck or fully covered.
            # If fully covered, break.
            # If not, we have a problem (should not happen in greedy if we pick max gain).
            break
            
        # Add block
        selected_indices.append(best_idx)
        covered_mask |= masks[best_idx]
        
        # Check if fully covered
        # We need to check if all valid bits are 1.
        # Create a valid mask
        # n_subsets = masks.shape[1] * 64 - padding?
        # We don't have n_subsets here directly, but we can infer it or pass it.
        # Let's just run for a fixed number of iterations or until gain is 0.
        # Actually, if gain is 0, we are done.
        
    return selected_indices

def _simulate_annealing(v, k, t, initial_indices, blocks_list, masks, deadline, rng):
    """
    Simulated Annealing to minimize block count.
    State: List of block indices.
    Objective: Minimize (N_blocks * 1000 + N_uncovered).
    """
    n_blocks = len(blocks_list)
    n_subsets = len(masks) # Wait, masks shape is (N_blocks, N_Words). 
    # We need n_subsets for the valid bit count.
    # Let's retrieve it from the precomputed cache.
    _, _, t_map, n_subsets = _get_precomputed(v, k, t)
    
    # Initialize state
    current_indices = list(initial_indices)
    
    # Calculate initial coverage counts
    # counts[i] = number of blocks covering subset i
    counts = np.zeros(n_subsets, dtype=np.int32)
    for idx in current_indices:
        # Expand mask to counts
        # This is slow if done naively.
        # Better: Use the mask to update counts?
        # We can just recompute from scratch initially.
        pass
    
    # To make SA fast, we need incremental updates.
    # Let's maintain `counts` and `uncovered_count`.
    
    # Helper to update counts and uncovered_count when adding/removing a block
    def _add_block(idx):
        nonlocal uncovered_count
        mask = masks[idx]
        # Convert mask to list of indices?
        # Or iterate through words?
        # Since n_words is small (~61), we can iterate.
        for w in range(len(mask)):
            word = mask[w]
            if word == 0: continue
            # Find set bits
            while word:
                bit = (word & -word).bit_length() - 1
                global_idx = w * 64 + bit
                if global_idx < n_subsets:
                    if counts[global_idx] == 0:
                        uncovered_count -= 1
                    counts[global_idx] += 1
                word &= word - 1 # clear lowest set bit
                
    def _remove_block(idx):
        nonlocal uncovered_count
        mask = masks[idx]
        for w in range(len(mask)):
            word = mask[w]
            if word == 0: continue
            while word:
                bit = (word & -word).bit_length() - 1
                global_idx = w * 64 + bit
                if global_idx < n_subsets:
                    counts[global_idx] -= 1
                    if counts[global_idx] == 0:
                        uncovered_count += 1
                word &= word - 1

    # Initialize counts and uncovered_count
    uncovered_count = 0
    counts = np.zeros(n_subsets, dtype=np.int32)
    for idx in current_indices:
        _add_block(idx)
        
    # Note: The above _add_block is O(n_words * popcount) per block.
    # With ~150 blocks, this is fine for initialization.
    # But in the SA loop, we call this for every move.
    # Is it fast enough?
    # 150 blocks * 61 words * avg 17 bits = ~150k ops per state update?
    # No, per move we update ONE block.
    # So per move: 61 words * 17 bits = ~1000 ops.
    # This is very fast.
    
    best_indices = list(current_indices)
    best_energy = len(current_indices) * 1000 + uncovered_count
    
    T = 100.0
    T_min = 1e-5
    cooling = 0.99
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        move_type = rng.random()
        neighbor_indices = list(current_indices)
        
        if move_type < 0.4:
            # Remove a random block
            if len(neighbor_indices) > 0:
                rem_idx = rng.randint(0, len(neighbor_indices) - 1)
                block_id = neighbor_indices[rem_idx]
                neighbor_indices.pop(rem_idx)
                # Apply move
                _remove_block(block_id)
                new_energy = len(neighbor_indices) * 1000 + uncovered_count
                
                # Check acceptance
                delta_e = new_energy - best_energy # Wait, compare to current?
                # Standard SA compares to current state energy.
                # Let's store current_energy.
                
                # Rollback for now, decide later
                _add_block(block_id) # Rollback
                # Re-evaluate properly
                
        # This rollback logic is messy. Let's compute delta directly.
        
        # Let's restart the loop logic with cleaner state management.
        
        pass

def construct(v, k, t):
    """
    Return a list of k-subsets of {0..v-1} covering every t-subset.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    rng = random.Random(42)
    
    # 1. Precompute
    blocks_list, masks, t_map, n_subsets = _get_precomputed(v, k, t)
    n_words = masks.shape[1]
    
    # 2. Greedy Construction
    greedy_indices = _greedy_construct(v, k, t, blocks_list, masks, deadline)
    
    # 3. Simulated Annealing
    # We need a robust SA implementation.
    
    # State: list of block indices
    current_indices = list(greedy_indices)
    
    # Counts of coverage for each t-subset
    counts = np.zeros(n_subsets, dtype=np.int32)
    
    # Helper to update counts
    def _update_counts(idx, sign):
        """
        Update counts and return change in uncovered_count.
        sign: +1 for add, -1 for remove.
        """
        delta_uncovered = 0
        mask = masks[idx]
        for w in range(n_words):
            word = mask[w]
            if word == 0:
                continue
            while word:
                bit = (word & -word).bit_length() - 1
                global_idx = w * 64 + bit
                if global_idx < n_subsets:
                    counts[global_idx] += sign
                    if sign == 1:
                        if counts[global_idx] == 1:
                            delta_uncovered -= 1
                    else:
                        if counts[global_idx] == 0:
                            delta_uncovered += 1
                word &= word - 1
        return delta_uncovered

    # Initialize counts
    uncovered_count = 0
    for idx in current_indices:
        uncovered_count += _update_counts(idx, 1) # Wait, _update_counts returns delta.
        # If we start with 0, and add 1, delta is -1 if it was 0.
        # So uncovered_count += delta is correct.
        
    # Actually, _update_counts(idx, 1) returns -1 if it was 0, 0 otherwise.
    # So uncovered_count starts at 0, and we subtract 1 for each newly covered.
    # This is correct.
    
    current_energy = len(current_indices) * 1000 + uncovered_count
    best_indices = list(current_indices)
    best_energy = current_energy
    
    T = 50.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a random move
        # 1. Remove a block
        # 2. Add a block
        # 3. Swap a block
        
        move = rng.random()
        
        if move < 0.5 and len(current_indices) > 0:
            # Remove
            pos = rng.randint(0, len(current_indices) - 1)
            block_id = current_indices[pos]
            
            # Calculate delta
            delta_unc = _update_counts(block_id, -1)
            new_len = len(current_indices) - 1
            new_energy = new_len * 1000 + (uncovered_count + delta_unc)
            
            # Accept?
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                # Accept
                current_indices.pop(pos)
                uncovered_count += delta_unc
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_indices = list(current_indices)
            else:
                # Reject, rollback
                _update_counts(block_id, 1)
                
        elif move < 0.8:
            # Add
            block_id = rng.randint(0, len(blocks_list) - 1)
            # Check if already present? 
            # If present, adding it again is redundant but allowed in state?
            # Better to ensure distinct blocks.
            if block_id in current_indices:
                continue
                
            delta_unc = _update_counts(block_id, 1)
            new_len = len(current_indices) + 1
            new_energy = new_len * 1000 + (uncovered_count + delta_unc)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_indices.append(block_id)
                uncovered_count += delta_unc
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_indices = list(current_indices)
            else:
                _update_counts(block_id, -1)
                
        else:
            # Swap
            if len(current_indices) == 0:
                continue
            pos = rng.randint(0, len(current_indices) - 1)
            old_id = current_indices[pos]
            new_id = rng.randint(0, len(blocks_list) - 1)
            
            if old_id == new_id:
                continue
            if new_id in current_indices:
                continue # Swapping with a duplicate doesn't change the set, just order.
                
            # Remove old, Add new
            delta_unc_old = _update_counts(old_id, -1)
            delta_unc_new = _update_counts(new_id, 1)
            total_delta_unc = delta_unc_old + delta_unc_new
            
            new_len = len(current_indices)
            new_energy = new_len * 1000 + (uncovered_count + total_delta_unc)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_indices[pos] = new_id
                uncovered_count += total_delta_unc
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_indices = list(current_indices)
            else:
                _update_counts(old_id, 1)
                _update_counts(new_id, -1)
                
        T *= cooling
        
    # Ensure best solution is valid
    # If best_energy has uncovered > 0, it's invalid.
    # We need to return a valid solution.
    # If best is invalid, we should fall back to the last valid state or greedy.
    # Let's track the last valid state separately or just ensure we don't lose track.
    # Actually, if best_energy includes uncovered penalty, and we minimize it,
    # it will prefer valid solutions (uncovered=0) with fewer blocks.
    # If it ends up with uncovered > 0, it means it couldn't find a valid one with fewer blocks?
    # Or it got stuck in an invalid basin.
    
    # To be safe, let's verify best_indices validity.
    # If invalid, we repair it or use greedy.
    
    # Simple repair: If best is invalid, use greedy (which is always valid).
    # Or, we can just use the best valid state we saw.
    # Let's track `best_valid_indices`.
    
    # Let's re-run the loop logic to track best valid.
    # This requires refactoring the loop above.
    
    # For now, let's just return the best if it's valid, otherwise greedy.
    # Check validity of best_indices
    test_counts = np.zeros(n_subsets, dtype=np.int32)
    for idx in best_indices:
        _update_counts(idx, 1) # Reuse helper? No, it modifies global counts.
        
    # Let's write a quick validity check
    valid = True
    for idx in best_indices:
        pass # This is getting complicated.
        
    # Simpler: Just use the greedy solution if we are not confident.
    # But the SA should be good.
    # Let's assume best_energy < 1000 implies valid (uncovered=0).
    # If best_energy >= 1000, it has uncovered subsets.
    
    if best_energy < 1000:
        return [list(blocks_list[i]) for i in best_indices]
    else:
        # Fallback to greedy
        return [list(blocks_list[i]) for i in greedy_indices]

# Refactor to track best valid state properly in SA
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    rng = random.Random(42)
    
    blocks_list, masks, t_map, n_subsets = _get_precomputed(v, k, t)
    n_words = masks.shape[1]
    
    # Greedy
    greedy_indices = _greedy_construct(v, k, t, blocks_list, masks, deadline)
    
    # SA
    current_indices = list(greedy_indices)
    counts = np.zeros(n_subsets, dtype=np.int32)
    
    def _update_counts(idx, sign):
        delta_uncovered = 0
        mask = masks[idx]
        for w in range(n_words):
            word = mask[w]
            if word == 0: continue
            while word:
                bit = (word & -word).bit_length() - 1
                global_idx = w * 64 + bit
                if global_idx < n_subsets:
                    counts[global_idx] += sign
                    if sign == 1:
                        if counts[global_idx] == 1:
                            delta_uncovered -= 1
                    else:
                        if counts[global_idx] == 0:
                            delta_uncovered += 1
                word &= word - 1
        return delta_uncovered

    uncovered_count = 0
    for idx in current_indices:
        uncovered_count += _update_counts(idx, 1)
        
    current_energy = len(current_indices) * 1000 + uncovered_count
    best_valid_indices = list(current_indices)
    best_valid_energy = current_energy
    
    T = 50.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min: break
        
        move = rng.random()
        
        if move < 0.5 and len(current_indices) > 0:
            pos = rng.randint(0, len(current_indices) - 1)
            block_id = current_indices[pos]
            
            delta_unc = _update_counts(block_id, -1)
            new_len = len(current_indices) - 1
            new_energy = new_len * 1000 + (uncovered_count + delta_unc)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_indices.pop(pos)
                uncovered_count += delta_unc
                current_energy = new_energy
                
                if uncovered_count == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_indices = list(current_indices)
            else:
                _update_counts(block_id, 1)
                
        elif move < 0.8:
            block_id = rng.randint(0, len(blocks_list) - 1)
            if block_id in current_indices: continue
            
            delta_unc = _update_counts(block_id, 1)
            new_len = len(current_indices) + 1
            new_energy = new_len * 1000 + (uncovered_count + delta_unc)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_indices.append(block_id)
                uncovered_count += delta_unc
                current_energy = new_energy
                
                if uncovered_count == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_indices = list(current_indices)
            else:
                _update_counts(block_id, -1)
                
        else:
            if len(current_indices) == 0: continue
            pos = rng.randint(0, len(current_indices) - 1)
            old_id = current_indices[pos]
            new_id = rng.randint(0, len(blocks_list) - 1)
            if old_id == new_id or new_id in current_indices: continue
            
            delta_unc_old = _update_counts(old_id, -1)
            delta_unc_new = _update_counts(new_id, 1)
            total_delta = delta_unc_old + delta_unc_new
            
            new_len = len(current_indices)
            new_energy = new_len * 1000 + (uncovered_count + total_delta)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_indices[pos] = new_id
                uncovered_count += total_delta
                current_energy = new_energy
                
                if uncovered_count == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_indices = list(current_indices)
            else:
                _update_counts(old_id, 1)
                _update_counts(new_id, -1)
                
        T *= cooling
        
    return [list(blocks_list[i]) for i in best_valid_indices]

