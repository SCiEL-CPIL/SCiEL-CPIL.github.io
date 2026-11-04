import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    def time_ok():
        return time.time() < deadline

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, store the array of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        idxs = np.array([t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)], dtype=np.int32)
        k_sub_t_idxs.append(idxs)

    # For each t-subset, store the list of k-subset indices that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    # Convert to numpy arrays for faster access
    t_sub_to_k_idxs_arr = [np.array(lst, dtype=np.int32) for lst in t_sub_to_k_idxs]

    def greedy_cover_fast(rng):
        """
        Fast greedy using numpy boolean mask for uncovered t-subsets.
        """
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        
        while uncovered.any() and time_ok():
            # Find indices of uncovered t-subsets
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            
            # Pick a random uncovered t-subset
            target_t_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
            
            # Get candidates (k-subsets that cover this t-subset)
            candidates = t_sub_to_k_idxs_arr[target_t_idx]
            
            # Shuffle candidates to break ties
            candidates = candidates.copy()
            rng.shuffle(candidates)
            
            best_block = -1
            best_gain = 0
            
            # Evaluate gain for each candidate
            for b_idx in candidates:
                # Gain is the number of currently uncovered t-subsets covered by this block
                # k_sub_t_idxs[b_idx] is an array of t-subset indices
                # uncovered is a boolean array
                gain = np.sum(uncovered[k_sub_t_idxs[b_idx]])
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            blocks.append(best_block)
            # Update uncovered mask
            uncovered[k_sub_t_idxs[best_block]] = False
        
        return blocks

    def reduce_blocks(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return []
            
        # Count coverage for each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        for b_idx in block_indices:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        current_blocks = list(block_indices)
        changed = True
        while changed and time_ok():
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                t_idxs = k_sub_t_idxs[b_idx]
                # If any t-subset in this block has cover_count == 1, it's essential
                if np.any(cover_count[t_idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    cover_count[t_idxs] -= 1
            
            current_blocks = new_blocks
        
        return current_blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run a few restarts. The fast greedy should allow more restarts within time limit.
    num_restarts = 10
    
    for i in range(num_restarts):
        if not time_ok():
            break
            
        rng = random.Random(i * 12345 + 67890)
        blocks = greedy_cover_fast(rng)
        
        if not blocks:
            continue
            
        reduced = reduce_blocks(blocks)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced
    
    if best_blocks is None:
        # Fallback
        rng = random.Random(42)
        blocks = greedy_cover_fast(rng)
        best_blocks = reduce_blocks(blocks)
        
    return [list(all_k_subs[idx]) for idx in best_blocks]
