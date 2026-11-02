import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit

    # 1. Precompute subset IDs
    # Map each t-subset to a unique integer ID
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    num_subsets = len(all_subsets)
    
    # Helper to get subset IDs for a block
    def get_block_subsets(block):
        return [subset_to_id[s] for s in itertools.combinations(block, t)]

    # 2. Greedy Construction
    # uncovered: set of subset IDs not yet covered
    uncovered = set(range(num_subsets))
    blocks = []
    counts = np.zeros(num_subsets, dtype=np.int32)
    
    # Precompute candidates for each t-subset? 
    # For a given subset S, blocks containing S are formed by choosing k-t points from v-t.
    # We can generate these on the fly.
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        # Pick a random uncovered subset to anchor the search
        # Using random helps escape local biases of greedy
        s_uncovered = random.choice(list(uncovered))
        s_points = all_subsets[s_uncovered]
        
        # Generate all blocks containing s_points
        # We need to choose k-t points from the remaining v-t points
        remaining_points = [p for p in range(v) if p not in s_points]
        candidate_blocks = []
        
        # Optimization: If v-t is large, this is big. 
        # For v=24, t=3, k=7: choose 4 from 21. 5985 candidates.
        # This is acceptable.
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(s_points + combo))
            candidate_blocks.append(block)
            
        best_block = None
        best_gain = -1
        
        for block in candidate_blocks:
            # Calculate gain: number of subsets in this block that are currently uncovered
            gain = 0
            for s_id in get_block_subsets(block):
                if s_id in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        # Add best_block
        blocks.append(best_block)
        for s_id in get_block_subsets(best_block):
            counts[s_id] += 1
            if counts[s_id] == 1:
                uncovered.discard(s_id)

    # If we timed out during greedy, we might have an invalid covering.
    # The problem states "Valid when every t-subset ... lies in at least one block".
    # If we break early, we must ensure validity or return best valid.
    # The greedy loop only breaks if uncovered is empty OR timeout.
    # If timeout, uncovered is not empty -> Invalid.
    # We need a fallback or ensure we finish greedy.
    # Greedy is fast enough (approx 1-2s). So we should be fine.
    
    # 3. Reduction Phase (Remove Redundant Blocks)
    def reduce_blocks(blocks, counts):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(blocks):
                block = blocks[i]
                s_ids = get_block_subsets(block)
                
                # Check if any subset in this block has count 1 (and thus is only covered by this block)
                # If all counts > 1, block is redundant.
                is_redundant = True
                for s_id in s_ids:
                    if counts[s_id] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block
                    for s_id in s_ids:
                        counts[s_id] -= 1
                    blocks.pop(i)
                    changed = True
                    # Do not increment i, check new block at same index
                else:
                    i += 1
        return blocks, counts

    blocks, counts = reduce_blocks(blocks, counts)

    # 4. Local Search (Simulated Annealing / Hill Climbing)
    # We try to swap blocks to allow further reductions or direct reductions.
    # Move: Replace block B_i with a random neighbor B_j.
    # Valid if coverage is maintained.
    
    current_blocks = blocks
    current_counts = counts
    
    def try_swap(i, new_block):
        # Simulate removing current_blocks[i] and adding new_block
        old_block = current_blocks[i]
        
        old_s_ids = get_block_subsets(old_block)
        new_s_ids = get_block_subsets(new_block)
        
        # Check validity
        # Subsets in old_block: count decreases by 1, increases by 1 if in new_block
        # Subsets in new_block: count increases by 1, decreases by 1 if in old_block
        
        # We need to ensure no count drops below 1.
        # Only subsets in old_block are at risk of dropping to 0.
        # Specifically, if s in old_block and s NOT in new_block, count[s] becomes counts[s]-1.
        # If counts[s] == 1, it becomes 0 -> Invalid.
        
        for s_id in old_s_ids:
            if counts[s_id] == 1 and s_id not in new_s_ids:
                return False
        
        # If valid, perform the swap
        for s_id in old_s_ids:
            current_counts[s_id] -= 1
        for s_id in new_s_ids:
            current_counts[s_id] += 1
        
        current_blocks[i] = new_block
        return True

    best_score = len(current_blocks)
    best_blocks = list(current_blocks)
    best_counts = current_counts.copy()
    
    # Local Search Loop
    # We use a simple hill climbing with random restarts/perturbations.
    # Since reduction is powerful, we focus on swaps that enable reduction.
    
    iterations = 0
    last_improvement = time.time()
    
    while time.time() - start_time < timeout - 5:
        # Randomly pick a block index
        i = random.randint(0, len(current_blocks) - 1)
        old_block = current_blocks[i]
        
        # Generate a random neighbor
        # Strategy: Pick 1-3 points to change
        num_changes = random.choice([1, 1, 2, 2, 3])
        points_in_block = list(old_block)
        points_out_block = [p for p in range(v) if p not in old_block]
        
        # Choose points to remove
        to_remove = random.sample(points_in_block, num_changes)
        # Choose points to add
        to_add = random.sample(points_out_block, num_changes)
        
        new_block_set = set(old_block)
        for p in to_remove:
            new_block_set.remove(p)
        for p in to_add:
            new_block_set.add(p)
        new_block = tuple(sorted(new_block_set))
        
        # Check if new_block is already in current_blocks (avoid duplicates)
        if new_block in current_blocks:
            continue
            
        # Try the swap
        if try_swap(i, new_block):
            # After a successful swap, try to reduce
            prev_len = len(current_blocks)
            current_blocks, current_counts = reduce_blocks(current_blocks, current_counts)
            
            if len(current_blocks) < best_score:
                best_score = len(current_blocks)
                best_blocks = list(current_blocks)
                best_counts = current_counts.copy()
                last_improvement = time.time()
                
            # If no improvement in a long time, restart from best or perturb
            if time.time() - last_improvement > 5.0:
                # Reset to best
                current_blocks = list(best_blocks)
                current_counts = best_counts.copy()
                # Perturb: Randomly swap a few blocks
                for _ in range(10):
                    idx = random.randint(0, len(current_blocks) - 1)
                    ob = current_blocks[idx]
                    # Simple perturbation: change 2 points
                    to_rem = random.sample(list(ob), 2)
                    to_add = random.sample([p for p in range(v) if p not in ob], 2)
                    nb_set = set(ob) - set(to_rem) | set(to_add)
                    nb = tuple(sorted(nb_set))
                    if nb not in current_blocks:
                         if try_swap(idx, nb):
                             current_blocks, current_counts = reduce_blocks(current_blocks, current_counts)

    # Final reduction on best found
    best_blocks, best_counts = reduce_blocks(best_blocks, best_counts)
    
    return [list(b) for b in best_blocks]
