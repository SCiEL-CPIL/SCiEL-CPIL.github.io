import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Precomputation
    all_ts = list(itertools.combinations(range(v), t))
    ts_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    num_ts = len(all_ts)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block -> ts indices
    block_ts_indices = []
    for b in all_blocks:
        ts_list = [ts_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_ts_indices.append(ts_list)
        
    # Matrix M for greedy
    M = np.zeros((num_blocks, num_ts), dtype=np.uint8)
    for i, ts_list in enumerate(block_ts_indices):
        for j in ts_list:
            M[i, j] = 1

    def greedy_cover(uncovered_mask):
        """
        uncovered_mask: boolean array of length num_ts
        Returns list of block indices
        """
        selected = []
        unc = uncovered_mask.copy()
        while np.any(unc):
            gains = M @ unc
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            unc[M[best_idx, :] == 1] = False
        return selected

    # Initial Greedy
    initial_uncovered = np.ones(num_ts, dtype=bool)
    current_indices = greedy_cover(initial_uncovered)
    
    # Local Search
    def prune(indices):
        """Remove redundant blocks. Returns new indices and counts."""
        counts = np.zeros(num_ts, dtype=np.int32)
        for idx in indices:
            for ts_idx in block_ts_indices[idx]:
                counts[ts_idx] += 1
                
        new_indices = []
        for idx in indices:
            ts_list = block_ts_indices[idx]
            if len(ts_list) == 0:
                continue
            min_cov = np.min(counts[ts_list])
            if min_cov > 1:
                for ts_idx in ts_list:
                    counts[ts_idx] -= 1
            else:
                new_indices.append(idx)
        return new_indices, counts

    best_indices, best_counts = prune(current_indices)
    
    while time.time() < deadline:
        # Try to improve by removing a block and re-greedying
        # This is a "kick"
        
        # Pick a random block to remove
        if not best_indices:
            break
            
        remove_idx_pos = np.random.randint(len(best_indices))
        remove_idx = best_indices[remove_idx_pos]
        
        # Create new list without this block
        candidate_indices = [idx for i, idx in enumerate(best_indices) if i != remove_idx_pos]
        
        # Check which ts are now uncovered
        # We need to recompute counts or just check uncovered
        # Let's recompute counts for candidate_indices
        temp_counts = np.zeros(num_ts, dtype=np.int32)
        for idx in candidate_indices:
            for ts_idx in block_ts_indices[idx]:
                temp_counts[ts_idx] += 1
                
        # Uncovered are those with count 0
        uncovered_mask = (temp_counts == 0)
        
        if not np.any(uncovered_mask):
            # Still valid! We removed a block and it's still valid?
            # This means the block was redundant, but prune should have caught it.
            # Unless prune logic was flawed?
            # Prune removes if min_cov > 1.
            # If min_cov == 1 for all, it stays.
            # If we remove it, those ts become 0.
            # So this branch should rarely happen unless prune was incomplete.
            # If it happens, we accept the smaller set.
            best_indices = candidate_indices
            best_counts = temp_counts
            continue
            
        # Run greedy to cover the newly uncovered ts
        # We only need to cover the ones that became uncovered
        # But greedy_cover expects a mask.
        new_blocks = greedy_cover(uncovered_mask)
        
        # Add new blocks
        candidate_indices.extend(new_blocks)
        
        # Prune the result
        pruned_indices, pruned_counts = prune(candidate_indices)
        
        # Compare with best
        if len(pruned_indices) < len(best_indices):
            best_indices = pruned_indices
            best_counts = pruned_counts
        else:
            # No improvement, we could try a different kick or break
            # To avoid getting stuck, we just continue
            pass

    final_blocks = [list(all_blocks[idx]) for idx in best_indices]
    return final_blocks
