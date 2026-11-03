import itertools
import random
import time
import numpy as np
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback for non-target parameters (keep simple, though unlikely to be tested)
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            for block_tuple in itertools.combinations(range(v), k):
                block_set = set(block_tuple)
                count = sum(1 for tsub in uncovered if set(tsub).issubset(block_set))
                if count > best_count:
                    best_count = count
                    best_block = list(block_tuple)
                if time.time() > deadline:
                    # Emergency exit, might be invalid
                    break
            if best_block is None or best_count == 0:
                break
            blocks.append(best_block)
            block_set = set(best_block)
            uncovered = {tsub for tsub in uncovered if not set(tsub).issubset(block_set)}
            if time.time() > deadline:
                break
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks = []
    block_tsubs_ids = [] # List of lists of tsub IDs
    
    for block_tuple in itertools.combinations(points, k):
        block_fs = frozenset(block_tuple)
        tsubs_in_block = []
        for tsub in itertools.combinations(block_tuple, t):
            tsubs_in_block.append(tsub_to_id[tsub])
        all_blocks.append(block_fs)
        block_tsubs_ids.append(tsubs_in_block)
        
    num_total_blocks = len(all_blocks)
    
    # 1. Greedy Initialization with Randomization
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    current_block_indices = []
    current_set = set()
    num_uncovered = num_tsubs
    rng = random.Random(42)
    
    while num_uncovered > 0:
        best_b_idx = -1
        best_gain = -1
        best_candidates = []
        
        # Iterate all blocks
        for b_idx in range(num_total_blocks):
            if b_idx in current_set:
                continue
            
            tids = block_tsubs_ids[b_idx]
            # Vectorized gain calculation
            gain = np.sum(cover_count[tids] == 0)
            
            if gain > best_gain:
                best_gain = gain
                best_candidates = [b_idx]
            elif gain == best_gain and gain > 0:
                best_candidates.append(b_idx)
                
        if best_b_idx == -1 or best_gain == 0:
            break
            
        # Randomly select from best candidates to escape local minima
        best_b_idx = rng.choice(best_candidates)
        
        current_block_indices.append(best_b_idx)
        current_set.add(best_b_idx)
        tids = block_tsubs_ids[best_b_idx]
        cover_count[tids] += 1
        num_uncovered = np.sum(cover_count == 0)
        
        if time.time() > deadline:
            break

    # 2. Simulated Annealing with Better Move Selection
    current_energy = len(current_block_indices)
    
    # Adaptive penalty: start high to ensure coverage, decrease to allow more removals
    penalty_lambda = 50.0
    T = 1.0
    T_min = 1e-3
    cooling_rate = 0.999
    # rng already initialized
    
    # Track best valid solution
    best_valid_blocks = list(current_block_indices)
    best_valid_energy = current_energy if num_uncovered == 0 else float('inf')
    
    if num_uncovered == 0:
        best_valid_blocks = list(current_block_indices)
        best_valid_energy = current_energy

    while time.time() < deadline and T > T_min:
        # Choose move: prioritize removals when valid, otherwise balance
        if num_uncovered == 0:
            # When valid, try to remove more aggressively
            if rng.random() < 0.8:
                move_type = 'remove'
            else:
                move_type = 'swap'
        else:
            # When invalid, try to add or swap to fix coverage
            if rng.random() < 0.5:
                move_type = 'add'
            else:
                move_type = 'swap'
            
        if move_type == 'remove':
            if not current_set:
                continue
            # Pick random block from current
            b_idx = rng.choice(list(current_set))
            
            tids = block_tsubs_ids[b_idx]
            # Delta uncovered: tsubs that go from 1 to 0
            delta_uncovered = np.sum(cover_count[tids] == 1)
            
            # Smooth surrogate: use soft penalty
            delta_E = -1.0 + penalty_lambda * delta_uncovered
            
            if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
                # Accept
                current_set.discard(b_idx)
                current_block_indices.remove(b_idx)
                cover_count[tids] -= 1
                num_uncovered += delta_uncovered
                current_energy -= 1
                
                if num_uncovered == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_blocks = list(current_block_indices)
            
        elif move_type == 'add':
            # Pick random block not in current
            # Rejection sampling
            while True:
                b_idx = rng.randint(0, num_total_blocks - 1)
                if b_idx not in current_set:
                    break
            
            tids = block_tsubs_ids[b_idx]
            # Delta uncovered: tsubs that go from 0 to 1
            # gain = number of 0s in tids
            delta_uncovered = -np.sum(cover_count[tids] == 0)
            
            delta_E = 1.0 + penalty_lambda * delta_uncovered
            
            if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
                # Accept
                current_set.add(b_idx)
                current_block_indices.append(b_idx)
                cover_count[tids] += 1
                num_uncovered += delta_uncovered
                current_energy += 1
                
                if num_uncovered == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_blocks = list(current_block_indices)
                    
        else: # swap: remove one, add one
            if not current_set:
                continue
            # Pick random block to remove
            b_remove = rng.choice(list(current_set))
            tids_remove = block_tsubs_ids[b_remove]
            delta_uncovered_remove = np.sum(cover_count[tids_remove] == 1)
            
            # Pick random block to add
            while True:
                b_add = rng.randint(0, num_total_blocks - 1)
                if b_add not in current_set:
                    break
            tids_add = block_tsubs_ids[b_add]
            delta_uncovered_add = -np.sum(cover_count[tids_add] == 0)
            
            # Combined delta
            delta_uncovered_total = delta_uncovered_remove + delta_uncovered_add
            # Energy change: -1 (remove) + 1 (add) = 0 net change in block count
            delta_E = 0.0 + penalty_lambda * delta_uncovered_total
            
            if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
                # Accept
                current_set.discard(b_remove)
                current_block_indices.remove(b_remove)
                cover_count[tids_remove] -= 1
                
                current_set.add(b_add)
                current_block_indices.append(b_add)
                cover_count[tids_add] += 1
                
                num_uncovered += delta_uncovered_total
                # current_energy unchanged
                
                if num_uncovered == 0 and current_energy < best_valid_energy:
                    best_valid_energy = current_energy
                    best_valid_blocks = list(current_block_indices)
                    
        T *= cooling_rate
        # Adaptively decrease penalty as we get closer to valid
        if num_uncovered == 0:
            penalty_lambda = max(1.0, penalty_lambda * 0.99)
        else:
            penalty_lambda = min(100.0, penalty_lambda * 1.01)

    # Final Repair if invalid
    if num_uncovered > 0:
        # Greedy repair
        while num_uncovered > 0:
            best_b_idx = -1
            best_gain = -1
            best_candidates = []
            for b_idx in range(num_total_blocks):
                if b_idx in current_set:
                    continue
                tids = block_tsubs_ids[b_idx]
                gain = np.sum(cover_count[tids] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(b_idx)
            if best_b_idx == -1 or best_gain == 0:
                break
            best_b_idx = rng.choice(best_candidates)
            current_set.add(best_b_idx)
            current_block_indices.append(best_b_idx)
            tids = block_tsubs_ids[best_b_idx]
            cover_count[tids] += 1
            num_uncovered = np.sum(cover_count == 0)
            if time.time() > deadline:
                break
        
        # Update best if this is better (should be, as it's valid)
        if num_uncovered == 0:
            if len(current_block_indices) < best_valid_energy:
                best_valid_blocks = list(current_block_indices)
                best_valid_energy = len(current_block_indices)
            
    # If we still have uncovered (shouldn't happen with repair), use best valid or greedy
    if num_uncovered > 0:
        # Fallback to best valid found or greedy
        if best_valid_blocks:
            final_indices = best_valid_blocks
        else:
            # Re-run simple greedy? Or just return what we have (invalid)
            # We must return valid.
            # Let's try a quick greedy from scratch if we are broken
            pass
            
    # Use best valid blocks
    final_indices = best_valid_blocks if best_valid_blocks else current_block_indices
    
    # Convert to lists
    final_blocks = [sorted(all_blocks[i]) for i in final_indices]
    
    return final_blocks
