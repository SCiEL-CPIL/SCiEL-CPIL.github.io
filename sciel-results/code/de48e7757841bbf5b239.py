import itertools
import random
import time
import numpy as np
import heapq


def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, the list of t-subset indices it covers
    block_tsubs_list = []
    for b in blocks:
        ts_idxs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs_list.append(ts_idxs)
    
    # For each t-subset, list of block indices that cover it
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsub_blocks[tsub_idx[ts]].append(bi)
    
    # Convert to numpy arrays
    block_tsubs_arr = [np.array(x, dtype=np.int32) for x in block_tsubs_list]
    tsub_blocks_arr = [np.array(x, dtype=np.int32) for x in tsub_blocks]
    
    return tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks


def _greedy_cover(v, k, t, rng, precomp):
    """Global greedy set cover with efficient gain updates."""
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=np.bool_)
    selected = []
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = np.sum(uncovered[block_tsubs_arr[bi]])
    
    while uncovered.any():
        best_bi = int(np.argmax(gain))
        if gain[best_bi] == 0:
            break
        
        selected.append(best_bi)
        idxs = block_tsubs_arr[best_bi]
        
        was_uncovered = uncovered[idxs]
        uncovered[idxs] = False
        
        # Update gains: only for blocks that cover newly covered t-subsets
        for tsi in idxs[was_uncovered]:
            for bi2 in tsub_blocks_arr[tsi]:
                if gain[bi2] > 0:
                    gain[bi2] -= 1
    
    return selected


def _greedy_randomized(v, k, t, rng, precomp, alpha=0.3):
    """Randomized greedy."""
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=np.bool_)
    selected = []
    
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = np.sum(uncovered[block_tsubs_arr[bi]])
    
    while uncovered.any():
        if rng.random() < alpha:
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_tsi = uncovered_indices[rng.randrange(len(uncovered_indices))]
            
            candidates = tsub_blocks_arr[target_tsi]
            if len(candidates) == 0:
                break
            
            best_bi = -1
            best_gain = -1
            for bi in candidates:
                g = int(gain[bi])
                if g > best_gain:
                    best_gain = g
                    best_bi = bi
            if best_bi == -1 or best_gain <= 0:
                break
        else:
            best_bi = int(np.argmax(gain))
            if gain[best_bi] == 0:
                break
        
        selected.append(best_bi)
        idxs = block_tsubs_arr[best_bi]
        was_uncovered = uncovered[idxs]
        uncovered[idxs] = False
        
        for tsi in idxs[was_uncovered]:
            for bi2 in tsub_blocks_arr[tsi]:
                if gain[bi2] > 0:
                    gain[bi2] -= 1
    
    return selected


def _local_search(v, k, t, block_indices, rng, precomp, time_limit=100.0):
    """Improved local search: deletion, single swap, 2-swap."""
    start_time = time.time()
    tsubs, tsub_idx, blocks_list, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    blocks = list(block_indices)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in blocks:
        cov[block_tsubs_arr[bi]] += 1
    
    # Precompute for faster swap search: for each t-subset, candidate blocks
    # Already have tsub_blocks_arr
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Aggressive deletion of redundant blocks
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            bi = blocks[i]
            idxs = block_tsubs_arr[bi]
            if np.all(cov[idxs] > 1):
                cov[idxs] -= 1
                blocks.pop(i)
                improved = True
                # Continue deleting in same pass
                indices = list(range(len(blocks)))
                rng.shuffle(indices)
        
        if improved:
            continue
        
        if len(blocks) == 0:
            break
        
        # Phase 2: Single-block swaps - try to replace a block with another that covers all its critical t-subsets
        found_swap = False
        num_tries = min(100, len(blocks))
        
        for _ in range(num_tries):
            if time.time() - start_time > time_limit:
                break
            
            i = rng.randrange(len(blocks))
            bi = blocks[i]
            idxs_bi = block_tsubs_arr[bi]
            
            critical_mask = cov[idxs_bi] == 1
            critical_idxs = idxs_bi[critical_mask]
            
            if len(critical_idxs) == 0:
                cov[idxs_bi] -= 1
                blocks.pop(i)
                improved = True
                found_swap = True
                break
            
            # Find replacement: must cover all critical t-subsets
            # Use the first critical t-subset to get candidates
            first_crit_tsi = int(critical_idxs[0])
            candidates = tsub_blocks_arr[first_crit_tsi]
            
            # Filter candidates that are not the same block
            candidates = [c for c in candidates if c != bi]
            
            if not candidates:
                continue
            
            # Try candidates, prefer those with high gain (cover many other uncovered-by-bi t-subsets)
            best_replacement = None
            best_score = -1
            
            # Limit candidates for speed
            if len(candidates) > 200:
                candidates = list(rng.sample(candidates, 200))
            
            for bi2 in candidates:
                idxs_bi2 = block_tsubs_arr[bi2]
                # Check if bi2 covers all critical t-subsets
                # Use set intersection for speed
                if not np.all(np.isin(critical_idxs, idxs_bi2)):
                    continue
                
                # Score: how many t-subsets does bi2 cover that bi doesn't?
                # This helps future deletions
                # Quick score: just accept first valid one, or try a few
                if best_replacement is None:
                    best_replacement = bi2
                    best_score = 1
                elif best_score < 2:
                    # Try to find a better one
                    overlap = np.sum(np.isin(idxs_bi2, idxs_bi))
                    new_score = len(idxs_bi2) - overlap
                    if new_score > best_score:
                        best_replacement = bi2
                        best_score = new_score
                    if best_score > 10:
                        break
            
            if best_replacement is not None:
                cov[idxs_bi] -= 1
                cov[block_tsubs_arr[best_replacement]] += 1
                blocks[i] = best_replacement
                improved = True
                found_swap = True
                break
        
        if improved:
            continue
        
        # Phase 3: 2-block removal + 2-block addition
        if len(blocks) >= 2:
            found_2swap = False
            num_2swap_tries = min(50, len(blocks) * 2)
            
            for _ in range(num_2swap_tries):
                if time.time() - start_time > time_limit:
                    break
                
                i1 = rng.randrange(len(blocks))
                i2 = rng.randrange(len(blocks))
                if i1 == i2:
                    continue
                
                bi1 = blocks[i1]
                bi2 = blocks[i2]
                idxs1 = block_tsubs_arr[bi1]
                idxs2 = block_tsubs_arr[bi2]
                
                # t-subsets that would become uncovered
                temp_cov = cov.copy()
                temp_cov[idxs1] -= 1
                temp_cov[idxs2] -= 1
                
                newly_uncovered = (temp_cov == 0)
                nu_count = int(np.sum(newly_uncovered))
                
                if nu_count == 0:
                    # Both redundant together
                    cov[idxs1] -= 1
                    cov[idxs2] -= 1
                    if i1 > i2:
                        blocks.pop(i1)
                        blocks.pop(i2)
                    else:
                        blocks.pop(i2)
                        blocks.pop(i1)
                    improved = True
                    found_2swap = True
                    break
                
                if nu_count > 12:
                    continue
                
                nu_idxs = np.where(newly_uncovered)[0]
                
                # Find two blocks that cover all newly uncovered
                first_nu = int(nu_idxs[0])
                cand1 = tsub_blocks_arr[first_nu]
                cand1 = [c for c in cand1 if c != bi1 and c != bi2]
                
                if not cand1:
                    continue
                
                if len(cand1) > 100:
                    cand1 = list(rng.sample(cand1, 100))
                
                for c1 in cand1[:50]:
                    idxs_c1 = block_tsubs_arr[c1]
                    remaining = newly_uncovered.copy()
                    remaining[idxs_c1] = False
                    rem_count = int(np.sum(remaining))
                    
                    if rem_count == 0:
                        # c1 alone covers all - we need a second block, use any valid one
                        # Actually, we're replacing 2 with 2, so we need exactly 2
                        # If c1 covers all, we can add any block that doesn't create issues
                        # For simplicity, skip this case or use a random valid block
                        continue
                    
                    if rem_count > 8:
                        continue
                    
                    second_nu = int(np.where(remaining)[0][0])
                    cand2 = tsub_blocks_arr[second_nu]
                    cand2 = [c for c in cand2 if c != bi1 and c != bi2 and c != c1]
                    
                    if not cand2:
                        continue
                    
                    if len(cand2) > 50:
                        cand2 = list(rng.sample(cand2, 50))
                    
                    for c2 in cand2[:30]:
                        idxs_c2 = block_tsubs_arr[c2]
                        all_covered = newly_uncovered.copy()
                        all_covered[idxs_c1] = False
                        all_covered[idxs_c2] = False
                        if not np.any(all_covered):
                            # Valid 2-swap
                            cov[idxs1] -= 1
                            cov[idxs2] -= 1
                            cov[idxs_c1] += 1
                            cov[idxs_c2] += 1
                            blocks[i1] = c1
                            blocks[i2] = c2
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
            
            if improved:
                continue
        
        # No improvement
        break
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy + local search."""
    deadline = time.time() + 145.0
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple greedy strategies
    strategies = [
        (0.0, 0),
        (0.3, 42),
        (0.5, 123),
        (0.7, 456),
        (0.0, 789),
        (0.4, 1000),
        (0.6, 2000),
    ]
    
    for alpha, seed in strategies:
        if time.time() > deadline - 80:
            break
        
        rng = random.Random(seed)
        if alpha == 0.0:
            blocks = _greedy_cover(v, k, t, rng, precomp)
        else:
            blocks = _greedy_randomized(v, k, t, rng, precomp, alpha=alpha)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
    
    # Additional greedy restarts
    for r in range(5):
        if time.time() > deadline - 80:
            break
        rng = random.Random(5000 + r * 137)
        blocks = _greedy_cover(v, k, t, rng, precomp)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
    
    # Local search on best solution
    if best_blocks is not None:
        if time.time() < deadline - 10:
            rng = random.Random(42)
            remaining = min(100.0, deadline - time.time() - 5.0)
            best_blocks = _local_search(v, k, t, best_blocks, rng, precomp, time_limit=remaining)
    
    # Second local search pass with different seed
    if best_blocks is not None and time.time() < deadline - 15:
        rng = random.Random(999)
        remaining = min(50.0, deadline - time.time() - 5.0)
        if remaining > 10:
            best_blocks = _local_search(v, k, t, best_blocks, rng, precomp, time_limit=remaining)
    
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, precomp)
    
    # Convert block indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    seen = set()
    final_blocks = []
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
