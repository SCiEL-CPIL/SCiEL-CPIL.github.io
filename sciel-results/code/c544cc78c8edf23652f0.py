import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to subsets mapping
    block_to_subsets = [[] for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            block_to_subsets[b_idx].append(subset_to_idx[s])
            
    # Precompute subset to blocks mapping
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    def get_greedy_solution():
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        selected = []
        for _ in range(num_blocks):
            if not uncovered.any():
                break
            best_idx = -1
            best_gain = 0
            # Randomized tie breaking
            # We pick the max gain, and among ties, pick randomly
            max_gain_val = gains.max()
            if max_gain_val == 0:
                break
            candidates = np.where(gains == max_gain_val)[0]
            best_idx = random.choice(candidates.tolist())
            
            selected.append(best_idx)
            b_subs = block_to_subsets[best_idx]
            for s in b_subs:
                if uncovered[s]:
                    uncovered[s] = False
                    for b in subset_to_blocks[s]:
                        gains[b] -= 1
        return selected

    def try_delete_redundant(current_set, coverage_counts):
        """Remove any block that covers no unique subsets. Returns True if any removed."""
        removed = False
        # Iterate backwards to safely pop
        for i in range(len(current_set) - 1, -1, -1):
            b = current_set[i]
            is_redundant = True
            for s in block_to_subsets[b]:
                if coverage_counts[s] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for s in block_to_subsets[b]:
                    coverage_counts[s] -= 1
                current_set.pop(i)
                removed = True
        return removed

    def is_valid_swap(old_block, new_block, coverage_counts):
        """Check if replacing old_block with new_block keeps coverage valid."""
        # New block must cover all subsets that were ONLY covered by old_block
        for s in block_to_subsets[old_block]:
            if coverage_counts[s] == 1:
                # This subset must be covered by new_block
                if s not in block_to_subsets[new_block]:
                    return False
        return True

    def local_search(start_time, time_limit):
        current_set = get_greedy_solution()
        if not current_set:
            return []
        
        coverage_counts = np.zeros(num_subsets, dtype=np.int32)
        for b in current_set:
            for s in block_to_subsets[b]:
                coverage_counts[s] += 1
                
        best_set = list(current_set)
        best_len = len(current_set)
        
        # Set of current block indices for fast lookup
        current_set_set = set(current_set)
        
        # Precompute a list of block indices for random sampling
        all_block_indices = list(range(num_blocks))
        
        while time.time() - start_time < time_limit:
            # 1. Delete Pass
            while try_delete_redundant(current_set, coverage_counts):
                pass
            
            if len(current_set) < best_len:
                best_len = len(current_set)
                best_set = list(current_set)
                # Reset stagnation counter if we improved
                # (Not strictly needed if we just accept same-size moves)
                
            # 2. Try Moves: Add or Swap
            # We will try a few random moves per iteration
            move_made = False
            
            # Strategy: 50% Add, 50% Swap
            if random.random() < 0.5:
                # --- ADD MOVE ---
                # Pick a random block not in current set
                # To avoid picking the same one repeatedly, we can just sample
                new_block = random.choice(all_block_indices)
                if new_block in current_set_set:
                    continue
                
                # Temporarily add
                for s in block_to_subsets[new_block]:
                    coverage_counts[s] += 1
                current_set.append(new_block)
                current_set_set.add(new_block)
                
                # Try to delete redundant blocks
                while try_delete_redundant(current_set, coverage_counts):
                    pass
                
                # Check if we improved or stayed same
                new_len = len(current_set)
                if new_len < best_len or (new_len == best_len and random.random() < 0.1):
                    # Accept (it's already applied)
                    move_made = True
                else:
                    # Revert
                    current_set.pop()
                    current_set_set.remove(new_block)
                    for s in block_to_subsets[new_block]:
                        coverage_counts[s] -= 1
            else:
                # --- SWAP MOVE ---
                if not current_set:
                    continue
                old_block = random.choice(current_set)
                new_block = random.choice(all_block_indices)
                
                if new_block in current_set_set:
                    continue
                if new_block == old_block:
                    continue
                
                # Check validity
                if not is_valid_swap(old_block, new_block, coverage_counts):
                    continue
                
                # Perform swap
                # Remove old
                for s in block_to_subsets[old_block]:
                    coverage_counts[s] -= 1
                current_set_set.remove(old_block)
                
                # Add new
                for s in block_to_subsets[new_block]:
                    coverage_counts[s] += 1
                current_set_set.add(new_block)
                
                # Replace in list (find index)
                # Since list is small, linear search is ok
                idx = current_set.index(old_block)
                current_set[idx] = new_block
                
                # Try to delete redundant blocks
                while try_delete_redundant(current_set, coverage_counts):
                    pass
                
                move_made = True

            # If we didn't make a move, we might be stuck. 
            # The randomization in selection should handle this.
            # To avoid infinite loops on no-change, we rely on the time limit.
            
        return best_set

    start_time = time.time()
    result = local_search(start_time, 115)
    
    return [list(all_blocks[i]) for i in result]
