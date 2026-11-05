import itertools
import numpy as np
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_choose_t = len(list(itertools.combinations(range(k), t)))

    # block_t_indices[i, j] is the index of the j-th t-subset of block i
    block_t_indices = np.empty((n_blocks, k_choose_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[subset]

    # Build sparse coverage matrix for fast gain calculation
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = block_t_indices.flatten()
    data = np.ones(n_blocks * k_choose_t, dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))
    
    # Inverse mapping: for each t-subset, list of blocks that cover it
    t_sub_blocks = {i: [] for i in range(n_t)}
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            t_sub_blocks[idx].append(i)

    def greedy_cover(seed=0, alpha=0.5):
        """
        GRASP construction phase.
        alpha controls randomness: 0 is pure greedy, 1 is fully random among top candidates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains: number of uncovered t-subsets covered by each block
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)
            gains[used] = -1

            max_gain = gains.max()
            if max_gain <= 0:
                break

            # Restricted Candidate List (RCL)
            # Threshold = max_gain - alpha * (max_gain - min_gain_among_positive)
            # Simpler: take top 10% or top 10 blocks
            candidates = np.where(gains >= max_gain * (1 - alpha))[0]
            
            # Filter out used blocks again just in case, though gains are -1
            candidates = candidates[gains[candidates] > 0]
            
            if len(candidates) == 0:
                candidates = np.where(gains == max_gain)[0]

            # Randomly select from RCL
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            
            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def remove_redundant(selected):
        """Remove blocks that do not cover any unique t-subset."""
        if not selected:
            return selected

        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected = new_selected

        return selected

    def local_search(selected, max_rounds=20):
        """
        1-opt and 2-opt local search.
        """
        if not selected:
            return selected

        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1

        improved = True
        rnd = 0
        rng = random.Random(42)

        while improved and rnd < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rnd += 1

            # 1. Remove redundant
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    improved = True
                else:
                    new_selected.append(b)
            selected = new_selected
            
            if not selected:
                break

            # 2. 1-Opt: Try to replace each block
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break

                b = selected[i]
                sub_idx = block_t_indices[b]
                essential = sub_idx[cover_count[sub_idx] == 1]

                if len(essential) == 0:
                    # Should have been removed in step 1, but just in case
                    cover_count[sub_idx] -= 1
                    selected.pop(i)
                    improved = True
                    break
                else:
                    # Find best replacement
                    best_replace = -1
                    best_score = -1
                    
                    # Candidates must cover the first essential subset
                    first_ess = essential[0]
                    candidates_list = t_sub_blocks.get(first_ess, [])
                    
                    # Sample candidates to keep it fast
                    if len(candidates_list) > 200:
                        candidates_list = rng.sample(candidates_list, 200)
                    
                    essential_set = set(essential)
                    for cand in candidates_list:
                        if cand == b or cand in selected:
                            continue
                        
                        cand_sub = block_t_indices[cand]
                        # Check if cand covers all essential subsets
                        # Optimization: check if any essential is NOT in cand_sub
                        # Using set intersection is fast enough for small sets
                        if not essential_set.issubset(set(cand_sub)):
                            continue
                        
                        # Score: 
                        # 1. Number of new subsets covered (count == 0)
                        # 2. Minus number of subsets that become over-covered (count == 1 -> 2)
                        new_covers = np.sum(cover_count[cand_sub] == 0)
                        # We don't strictly need to penalize over-coverage heavily, 
                        # but it helps keep the solution "tight" for future removals
                        score = new_covers
                        
                        if score > best_score:
                            best_score = score
                            best_replace = cand

                    if best_replace != -1:
                        old_sub = block_t_indices[b]
                        new_sub = block_t_indices[best_replace]
                        cover_count[old_sub] -= 1
                        cover_count[new_sub] += 1
                        selected[i] = best_replace
                        improved = True
                        break # Restart loop after a change

        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    selected = greedy_cover(0, alpha=0.0)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_rounds=10)
    if selected:
        best_blocks = selected
        best_len = len(selected)

    # 2. GRASP Restarts
    # We use different alpha values to balance greediness and randomness
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7]
    n_restarts_per_alpha = 5
    
    for alpha in alphas:
        for seed in range(n_restarts_per_alpha):
            if time.time() > deadline - 10.0:
                break
                
            selected = greedy_cover(seed + 100 * alpha, alpha=alpha)
            selected = remove_redundant(selected)
            selected = local_search(selected, max_rounds=15)
            
            if selected and len(selected) < best_len:
                best_blocks = selected
                best_len = len(selected)
                
            if time.time() > deadline - 10.0:
                break

    # 3. Final Polish on the best solution
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search(best_blocks, max_rounds=30)
        best_blocks = remove_redundant(best_blocks)

    # 4. Final Cleanup
    best_blocks = remove_redundant(best_blocks)

    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    return result_blocks
