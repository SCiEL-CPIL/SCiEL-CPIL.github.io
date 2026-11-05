import itertools
import time
import random
import sys
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_index = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # 2. All k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # 3. Map each block to the indices of t-subsets it covers
    block_to_t_indices = []
    for block in all_k_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_subset_index[sub])
        block_to_t_indices.append(indices)
        
    # 4. Map each t-subset to the list of block indices covering it
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)
            
    # 5. Build sparse incidence matrix for vectorized greedy
    # M[b_idx, t_idx] = 1 if block b_idx covers t_subset t_idx
    rows = []
    cols = []
    for b_idx, indices in enumerate(block_to_t_indices):
        for t_idx in indices:
            rows.append(b_idx)
            cols.append(t_idx)
    data = np.ones(len(rows))
    M = sparse.csr_matrix((data, (rows, cols)), shape=(num_k, num_t))

    def greedy_cover(seed):
        random.seed(seed)
        uncovered = np.ones(num_t, dtype=bool)
        cover = []
        
        # Use top-k sampling for diversity
        top_k = 20
        
        while uncovered.any():
            # Calculate marginal gain for each block
            # M is CSR, so M @ vector is efficient
            gains = M @ uncovered.astype(float)
            
            # Find indices of top k gains
            # argsort is O(N log N), N=75k, which is fast
            sorted_indices = np.argsort(gains)
            top_indices = sorted_indices[-top_k:]
            
            # Filter to only those with positive gain
            valid_top = [i for i in top_indices if gains[i] > 0]
            
            if not valid_top:
                break
            
            # Use a weighted random choice favoring higher gains
            # but with some randomness for diversity
            weights = [gains[i] for i in valid_top]
            max_w = max(weights)
            # Soft-max style weighting: exp(gain / temp)
            temp = max_w * 0.3 if max_w > 0 else 1.0
            exp_weights = [np.exp(w / temp) for w in weights]
            total = sum(exp_weights)
            r = random.random() * total
            cumsum = 0
            chosen_idx = 0
            for idx, ew in enumerate(exp_weights):
                cumsum += ew
                if r <= cumsum:
                    chosen_idx = idx
                    break
            
            best_block = valid_top[chosen_idx]
            
            cover.append(best_block)
            
            # Update uncovered
            nz_indices = M.indices[M.indptr[best_block]:M.indptr[best_block+1]]
            uncovered[nz_indices] = False
            
        return cover

    def local_search(cover, seed):
        random.seed(seed)
        # cover is a list of block indices
        # Initialize counts
        counts = [0] * num_t
        for b_idx in cover:
            for t_idx in block_to_t_indices[b_idx]:
                counts[t_idx] += 1
        
        # Helper to compute excess coverage: sum of max(0, count-1)
        def compute_excess():
            excess = 0
            for c in counts:
                if c > 1:
                    excess += c - 1
            return excess
        
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            # Shuffle order to avoid bias
            indices = list(range(len(cover)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cover): continue # Cover might have shrunk
                b_idx = cover[i]
                
                # Find essential t-subsets for b_idx
                essential = []
                for t_idx in block_to_t_indices[b_idx]:
                    if counts[t_idx] == 1:
                        essential.append(t_idx)
                        
                if not essential:
                    # Block is redundant
                    # Remove it
                    cover.pop(i)
                    for t_idx in block_to_t_indices[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    break # Restart loop
                    
                else:
                    # Try to find a replacement
                    # Candidates must cover all essential t-subsets
                    # Intersection of t_to_blocks[s] for s in essential
                    # Start with the smallest list
                    min_list_idx = min(essential, key=lambda s: len(t_to_blocks[s]))
                    candidates = set(t_to_blocks[min_list_idx])
                    
                    for s in essential:
                        if s != min_list_idx:
                            candidates &= set(t_to_blocks[s])
                        if not candidates:
                            break
                    
                    candidates.discard(b_idx)
                    
                    if candidates:
                        # Pick best candidate
                        # Score: minimize excess coverage after swap
                        # After removing b_idx and adding c_idx, compute new excess
                        # We want to minimize the increase in excess (or maximize decrease)
                        
                        best_cand = None
                        best_score = float('inf')
                        
                        essential_set = set(essential)
                        
                        for c_idx in candidates:
                            # Simulate the swap to compute excess change
                            # Remove b_idx: for each t_idx in block_to_t_indices[b_idx], counts[t_idx] -= 1
                            # Add c_idx: for each t_idx in block_to_t_indices[c_idx], counts[t_idx] += 1
                            
                            # Compute excess change efficiently
                            # For t_idx in b_idx but not in c_idx: count decreases by 1
                            # For t_idx in c_idx but not in b_idx: count increases by 1
                            # For t_idx in both: no change
                            
                            b_set = set(block_to_t_indices[b_idx])
                            c_set = set(block_to_t_indices[c_idx])
                            
                            delta_excess = 0
                            # For t_idx only in b_idx (removed)
                            for t_idx in b_set - c_set:
                                old_count = counts[t_idx]
                                new_count = old_count - 1
                                old_excess = max(0, old_count - 1)
                                new_excess = max(0, new_count - 1)
                                delta_excess += new_excess - old_excess
                            
                            # For t_idx only in c_idx (added)
                            for t_idx in c_set - b_set:
                                old_count = counts[t_idx]
                                new_count = old_count + 1
                                old_excess = max(0, old_count - 1)
                                new_excess = max(0, new_count - 1)
                                delta_excess += new_excess - old_excess
                            
                            # For t_idx in both: no change to count, so no change to excess
                            
                            # Also, we need to ensure that after the swap, all essential
                            # t-subsets are still covered. Since c_idx covers all essential,
                            # this is guaranteed.
                            
                            if delta_excess < best_score:
                                best_score = delta_excess
                                best_cand = c_idx
                            
                            # Early break if we find a swap that reduces excess
                            if best_score < 0:
                                break
                                
                        if best_cand is not None and best_score < 0:
                            # Perform swap
                            # Remove b_idx
                            for t_idx in block_to_t_indices[b_idx]:
                                counts[t_idx] -= 1
                            # Add best_cand
                            for t_idx in block_to_t_indices[best_cand]:
                                counts[t_idx] += 1
                            cover[i] = best_cand
                            improved = True
                            break # Restart loop
                            
        return cover

    best_cover = None
    best_len = float('inf')
    
    # Run multiple instances
    seed_counter = 0
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        seed = seed_counter
        seed_counter += 1
        
        # Initial greedy
        cover = greedy_cover(seed)
        if not cover:
            continue
            
        # Local search
        cover = local_search(cover, seed + 1000)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover.copy()
            
            # If we've found a good solution, do additional local search passes
            if best_len < 350 and time.time() < deadline - 15.0:
                # Do additional local search passes with different seeds
                for extra_seed in range(3):
                    if time.time() >= deadline - 15.0:
                        break
                    cover2 = local_search(cover.copy(), seed + 1000 + extra_seed * 100)
                    if len(cover2) < best_len:
                        best_len = len(cover2)
                        best_cover = cover2.copy()
        
    # Convert block indices back to point lists
    if best_cover is None:
        # Fallback: run one more greedy to ensure we have a valid cover
        cover = greedy_cover(seed_counter)
        if cover:
            best_cover = cover
        else:
            return []
        
    # Deduplicate blocks (in case local search introduced duplicates somehow)
    seen = set()
    unique_cover = []
    for b_idx in best_cover:
        if b_idx not in seen:
            seen.add(b_idx)
            unique_cover.append(b_idx)
    best_cover = unique_cover
    
    result = []
    for b_idx in best_cover:
        result.append(list(all_k_blocks[b_idx]))
        
    return result

# Test locally if needed
# if __name__ == "__main__":
#     import time
#     start = time.time()
#     blocks = construct(19, 8, 5)
#     end = time.time()
#     print(f"Time: {end - start}, Blocks: {len(blocks)}")
