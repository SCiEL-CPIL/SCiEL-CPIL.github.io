"""Combined approach: Greedy initialization followed by Simulated Annealing with incremental coverage and greedy repair."""
import time
import random
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute C(k, t)
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover_matrix: shape (n_k, c_k_t), each entry is an index into all_t_subsets
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]

    # Build t_to_blocks: for each t-subset, list of block indices that cover it
    # This is useful for finding replacement blocks quickly
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for ts_idx in cover_matrix[i]:
            t_to_blocks[ts_idx].append(i)

    # 1. Greedy Initialization
    best_blocks = []
    uncovered = np.ones(n_t, dtype=bool)
    
    # Standard Greedy
    while np.any(uncovered):
        gains = np.sum(uncovered[cover_matrix], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # Random tie-breaking
        idx = int(candidates[random.randint(0, len(candidates) - 1)])
        best_blocks.append(idx)
        uncovered[cover_matrix[idx]] = False
        
    # Convert to list of indices for SA
    current_blocks = list(best_blocks)
    best_count = len(current_blocks)
    
    # Build initial cover_counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_blocks:
        cover_counts[cover_matrix[idx]] += 1

    # 2. Simulated Annealing
    # We want to minimize the number of blocks.
    # State: set of block indices.
    # Move: Remove one block, try to repair.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # Helper to check if a set of blocks covers a set of t-subsets
    # U: numpy array of t-subset indices
    # block_idx: int
    def covers_all(block_idx, U):
        if len(U) == 0:
            return True
        # Check if all elements in U are in cover_matrix[block_idx]
        # This is O(len(U) * c_k_t) if done naively, but U is small.
        # Faster: use a set? No, numpy is faster for small arrays.
        # Actually, we can just check if np.isin(U, cover_matrix[block_idx]).all()
        # But U is an array of indices.
        return np.isin(U, cover_matrix[block_idx]).all()

    def greedy_repair(U, current_block_set, max_blocks=2):
        """
        Try to cover U with at most max_blocks blocks not in current_block_set.
        Returns a list of block indices or None.
        """
        if len(U) == 0:
            return []
        
        U_remaining = set(U)
        repair_blocks = []
        
        for _ in range(max_blocks):
            if not U_remaining:
                break
            
            # Find block covering most of U_remaining
            best_block = -1
            best_covered = 0
            
            # Optimization: Instead of checking all n_k blocks, 
            # we can pick a random element from U_remaining and check candidates from t_to_blocks
            # But we need to cover ALL of U_remaining.
            # So we must check candidates for one element, and verify against the rest.
            
            target_t = next(iter(U_remaining))
            candidates = t_to_blocks[target_t]
            
            for cand_idx in candidates:
                if cand_idx in current_block_set:
                    continue
                # Check how many of U_remaining this block covers
                covered = np.isin(U_remaining, cover_matrix[cand_idx])
                count = np.sum(covered)
                if count > best_covered:
                    best_covered = count
                    best_block = cand_idx
                    if count == len(U_remaining):
                        break # Found a perfect block for this step
            
            if best_block == -1:
                return None # Cannot cover
            
            repair_blocks.append(best_block)
            current_block_set.add(best_block)
            
            # Update U_remaining
            covered_mask = np.isin(U_remaining, cover_matrix[best_block])
            # Remove covered elements from U_remaining
            # U_remaining is a set of ints
            # We need to subtract the covered ones
            to_remove = [u for u in U_remaining if u in cover_matrix[best_block]]
            for u in to_remove:
                U_remaining.discard(u)
        
        if U_remaining:
            return None
        return repair_blocks

    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Pick a random block to remove
        if not current_blocks:
            break
            
        remove_idx_pos = rng.randint(0, len(current_blocks) - 1)
        block_to_remove = current_blocks[remove_idx_pos]
        
        # Tentatively remove
        current_blocks.pop(remove_idx_pos)
        current_block_set = set(current_blocks)
        
        cover_counts[cover_matrix[block_to_remove]] -= 1
        
        # Find uncovered
        uncovered_mask = cover_counts == 0
        if not np.any(uncovered_mask):
            # Block was redundant
            best_count = len(current_blocks)
            # We can keep going, T doesn't need to change or can cool down faster
            T *= cooling
            continue
        
        U = np.where(uncovered_mask)[0]
        
        # Try to repair
        # First, try to find a single block
        repair = greedy_repair(U, current_block_set, max_blocks=1)
        
        if repair is not None:
            # Swap: 1 removed, 1 added
            current_blocks.extend(repair)
            for b in repair:
                cover_counts[cover_matrix[b]] += 1
            # Count unchanged
            T *= cooling
        else:
            # Try to repair with 2 blocks
            repair = greedy_repair(U, current_block_set, max_blocks=2)
            
            if repair is not None:
                # Increase: 1 removed, 2 added -> +1
                delta = 1
                if rng.random() < np.exp(-delta / T):
                    current_blocks.extend(repair)
                    for b in repair:
                        cover_counts[cover_matrix[b]] += 1
                    T *= cooling
                else:
                    # Revert
                    current_blocks.append(block_to_remove)
                    cover_counts[cover_matrix[block_to_remove]] += 1
                    T *= cooling
            else:
                # Cannot repair with 2 blocks, try 3? Or just revert.
                # Revert
                current_blocks.append(block_to_remove)
                cover_counts[cover_matrix[block_to_remove]] += 1
                T *= cooling

    # Final cleanup: Remove any redundant blocks
    # Rebuild cover_counts from scratch to be safe
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in current_blocks:
        cover_counts[cover_matrix[idx]] += 1
        
    final_blocks = []
    for idx in current_blocks:
        # Check if this block is necessary
        # If all its t-subsets are covered by other blocks, it's redundant
        # We can check if min(cover_counts[cover_matrix[idx]]) > 1
        # If yes, it's redundant.
        # But we need to be careful: if we remove it, does it break others?
        # No, if all its t-subsets are covered by others, removing it doesn't uncover anything.
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            # Redundant
            cover_counts[cover_matrix[idx]] -= 1
        else:
            final_blocks.append(idx)
            
    # We might have removed blocks that made others redundant. Iterate?
    # The SA should have handled most redundancy.
    # Just do one more pass of removal.
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for idx in final_blocks:
        cover_counts[cover_matrix[idx]] += 1
        
    truly_final_blocks = []
    for idx in final_blocks:
        if np.min(cover_counts[cover_matrix[idx]]) > 1:
            cover_counts[cover_matrix[idx]] -= 1
        else:
            truly_final_blocks.append(idx)

    return [list(all_k_subsets[idx]) for idx in truly_final_blocks]
