import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    # Deadline: 105 seconds to be safe (limit is 120, kill is 150)
    start_time = time.time()
    deadline = start_time + 105.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and the t-subsets they contain
    # This is the incidence structure
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the indices of the t-subsets it covers
    # We store this as a flat array and offsets for efficient slicing
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            indices.append(subset_to_idx[sub])
        k_sub_tsub_indices.append(indices)
        
    # Flatten for potential vectorization or just use list of lists
    # List of lists is fine for ~50k items of length 35
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    # The randomization here is in the tie-breaking and initial order if we were doing sequential,
    # but for full-scan greedy, the order is determined by gain.
    # To get diversity, we can shuffle the order in which we consider ties, 
    # or simply run the greedy multiple times with different random seeds for tie-breaking.
    
    for attempt in range(50): # Max 50 attempts, break if time runs out
        if time.time() > deadline - 15.0:
            break
            
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        used_blocks = set()
        
        # Greedy Set Cover
        while uncovered.any():
            if time.time() > deadline - 15.0:
                break
                
            best_gain = -1
            best_block_idx = -1
            best_tie_breaker = random.random()
            
            # To add randomness, we can shuffle the indices we iterate over?
            # No, that's expensive. Instead, we can add a small random noise to the gain.
            # Or, we can just pick randomly among ties.
            
            # Optimization: We don't need to check blocks that are already used?
            # Actually, in set cover, we don't reuse blocks. But here blocks are distinct k-subsets.
            # We should not add the same block twice.
            
            # Iterate over all k-subsets
            # We can skip if gain is 0, but we need to find max gain first.
            
            # To speed up, we can maintain a list of active candidates?
            # For now, scan all. 50k * 35 ops is fast.
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                    
                indices = k_sub_tsub_indices[i]
                # Calculate gain: number of uncovered t-subsets in this block
                # np.sum(uncovered[indices])
                # Since uncovered is boolean, this counts True values.
                gain = np.sum(uncovered[indices])
                
                if gain == 0:
                    continue
                    
                # Add small noise to break ties randomly
                gain += random.random() * 0.01
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                # No block covers any new subset? This should not happen if uncovered.any()
                break
                
            # Add the best block
            current_blocks.append(list(all_k_subsets[best_block_idx]))
            used_blocks.add(best_block_idx)
            
            # Update uncovered status
            indices = k_sub_tsub_indices[best_block_idx]
            uncovered[indices] = False
            
            # Early check
            if not uncovered.any():
                break
        
        # If the loop broke due to timeout, this run is invalid/incomplete
        if time.time() > deadline - 15.0:
            continue
            
        if not uncovered.any(): # Valid covering
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks[:]
                
                # If we hit the lower bound, stop
                if best_count <= 131:
                    break
        else:
            # This run didn't complete a covering, discard it
            pass

    if best_blocks is None:
        # Fallback: If no valid covering was found (should not happen), 
        # we must return *something* valid. 
        # We can run a deterministic greedy once more, ignoring the time limit slightly 
        # or just return a large valid covering.
        # Actually, the first greedy run should always produce a valid covering 
        # unless it times out. If it times out, we are in trouble.
        # Let's ensure the first run is deterministic and fast.
        
        # Re-run a single deterministic greedy without random noise, hoping it finishes
        uncovered = np.ones(num_subsets, dtype=bool)
        current_blocks = []
        used_blocks = set()
        
        while uncovered.any():
            best_gain = -1
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                indices = k_sub_tsub_indices[i]
                gain = np.sum(uncovered[indices])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            current_blocks.append(list(all_k_subsets[best_block_idx]))
            used_blocks.add(best_block_idx)
            indices = k_sub_tsub_indices[best_block_idx]
            uncovered[indices] = False
            
        best_blocks = current_blocks

    # Local Search: ILS with exact intersection repair
    if best_blocks is not None and time.time() < deadline - 10.0:
        best_blocks = local_search(best_blocks, v, k, t, subset_to_idx, deadline)
    
    # Spectral Pruning: remove blocks with low eigenvector centrality in the
    # block co-occurrence graph (edges weighted by shared t-subsets).
    if best_blocks is not None and time.time() < deadline - 5.0:
        best_blocks = spectral_prune(best_blocks, v, k, t, subset_to_idx, deadline)
    
    # Final redundant-block sweep (cheap, exact)
    if best_blocks is not None:
        best_blocks = final_sweep(best_blocks, v, k, t, subset_to_idx)
    
    return best_blocks

def spectral_prune(blocks, v, k, t, subset_to_idx, deadline):
    """
    Build a block co-occurrence graph where edge weight = number of shared
    t-subsets between two blocks.  Compute eigenvector centrality and prune
    blocks whose centrality falls below a dynamically adjusted threshold.
    A block is only removed if doing so does not break coverage.
    """
    n = len(blocks)
    if n <= 2:
        return blocks

    # Precompute the list of t-subset indices for each block
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_subs.append(subs)

    # Build the co-occurrence (shared t-subset) matrix incrementally.
    # For each t-subset, find all blocks covering it and add pairwise weights.
    # To keep memory/time bounded, cap the number of blocks per subset.
    from collections import defaultdict
    subset_to_blocks = defaultdict(list)
    for bi, subs in enumerate(block_subs):
        for si in subs:
            if len(subset_to_blocks[si]) < 12:  # cap to limit O(m^2) blow-up
                subset_to_blocks[si].append(bi)

    # Weighted adjacency: W[i][j] = # shared t-subsets
    # Use a dict of dicts for sparse storage
    W = defaultdict(lambda: defaultdict(int))
    for si, blks in subset_to_blocks.items():
        m = len(blks)
        for a in range(m):
            for b in range(a + 1, m):
                W[blks[a]][blks[b]] += 1
                W[blks[b]][blks[a]] += 1

    # Power iteration for eigenvector centrality
    x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        new_x = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i in W:
                s = 0.0
                for j, w in W[i].items():
                    s += w * x[j]
                new_x[i] = s
        norm = np.linalg.norm(new_x)
        if norm < 1e-12:
            break
        new_x /= norm
        if np.max(np.abs(new_x - x)) < 1e-7:
            x = new_x
            break
        x = new_x

    # Compute coverage counts
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    for subs in block_subs:
        for si in subs:
            coverage_count[si] += 1

    # Sort blocks by centrality ascending (least central first)
    order = np.argsort(x)  # ascending
    blocks = [b[:] for b in blocks]
    block_subs = [block_subs[i] for i in order]
    x_sorted = x[order]

    # Dynamically adjusted threshold: start at the 10th percentile, tighten
    # as we remove blocks.
    threshold = np.percentile(x_sorted, 10)

    removed = []
    for i in range(n):
        if time.time() > deadline - 5.0:
            break
        if x_sorted[i] >= threshold:
            # All remaining blocks have centrality >= threshold
            break
        # Try to remove block i
        safe = True
        for si in block_subs[i]:
            if coverage_count[si] <= 1:
                safe = False
                break
        if safe:
            for si in block_subs[i]:
                coverage_count[si] -= 1
            removed.append(i)
            # Slightly tighten threshold to keep removing more
            threshold = max(threshold, x_sorted[i] * 1.1)

    if not removed:
        return blocks

    # Rebuild the block list without removed blocks
    removed_set = set(removed)
    new_blocks = []
    for i in range(n):
        if i not in removed_set:
            new_blocks.append(blocks[i])
    return new_blocks

def final_sweep(blocks, v, k, t, subset_to_idx):
    """
    Exact redundant-block removal: repeatedly scan blocks in a fixed order
    and remove any block whose removal does not break coverage.
    """
    if not blocks:
        return blocks

    # Build coverage counts
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            si = subset_to_idx[sub]
            subs.append(si)
            coverage_count[si] += 1
        block_subs.append(subs)

    n = len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            # Check if block i is redundant
            safe = True
            for si in block_subs[i]:
                if coverage_count[si] <= 1:
                    safe = False
                    break
            if safe:
                for si in block_subs[i]:
                    coverage_count[si] -= 1
                blocks.pop(i)
                block_subs.pop(i)
                n -= 1
                changed = True
                break  # restart scan since indices shifted
    return blocks

def local_search(blocks, v, k, t, subset_to_idx, deadline):
    """
    ILS with exact intersection repair: try to remove blocks that are redundant,
    or swap blocks to maintain validity while reducing size.
    """
    blocks = [b[:] for b in blocks]
    current_count = len(blocks)
    
    # Precompute coverage count for each t-subset
    # coverage_count[i] = number of blocks that cover subset i
    coverage_count = np.zeros(len(subset_to_idx), dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            coverage_count[idx] += 1
    
    # Try to remove redundant blocks (those whose removal doesn't break coverage)
    # We can do this in a random order to escape local minima.
    # Use a list of (block, subs) to avoid recomputing combinations.
    block_subs = []
    for block in blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(subset_to_idx[sub])
        block_subs.append(subs)

    for _ in range(200):  # Max 200 removal attempts
        if time.time() > deadline - 5.0:
            break
        if current_count <= 131:
            break
        if not blocks:
            break

        # Pick a random block to try removing
        idx_to_try = random.randint(0, len(blocks) - 1)
        block_to_remove = blocks[idx_to_try]
        subs_to_remove = block_subs[idx_to_try]

        # Check if removing this block would leave any t-subset uncovered
        safe_to_remove = True
        for idx in subs_to_remove:
            if coverage_count[idx] <= 1:
                safe_to_remove = False
                break

        if safe_to_remove:
            # Remove the block and update coverage counts
            for idx in subs_to_remove:
                coverage_count[idx] -= 1
            blocks.pop(idx_to_try)
            block_subs.pop(idx_to_try)
            current_count = len(blocks)

    return blocks
