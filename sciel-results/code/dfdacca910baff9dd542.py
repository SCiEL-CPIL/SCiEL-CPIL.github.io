import itertools
import time
import sys
import numpy as np
from collections import Counter

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using optimized greedy with
    multiple randomized restarts and local search.
    """
    deadline = time.monotonic() + 110.0  # Safety margin before 120s kill
    points = list(range(v))

    # Precomputation: all t-subsets
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    # Precompute all k-blocks and their t-subset indices
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Store for each block, the list of t-subset indices it covers
    block_tsubs = []
    for b in all_kblocks:
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(idxs)

    # Helper to check coverage
    def is_complete(block_indices):
        if not block_indices:
            return num_tsubs == 0
        # Use a set to track covered t-subs
        covered = set()
        for bi in block_indices:
            covered.update(block_tsubs[bi])
        return len(covered) == num_tsubs

    # Helper to prune redundant blocks
    def prune(block_indices):
        if not block_indices:
            return block_indices
        counts = Counter()
        for bi in block_indices:
            for ti in block_tsubs[bi]:
                counts[ti] += 1
        
        changed = True
        while changed:
            if time.monotonic() > deadline - 10:
                break
            changed = False
            new_list = []
            for bi in block_indices:
                redundant = True
                for ti in block_tsubs[bi]:
                    if counts[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_tsubs[bi]:
                        counts[ti] -= 1
                    changed = True
                else:
                    new_list.append(bi)
            block_indices = new_list
        return block_indices

    # Fast Greedy
    def greedy(seed=None, noise_scale=0.0):
        rng = np.random.default_rng(seed)
        uncovered = set(range(num_tsubs))
        selected = []
        remaining_blocks = set(range(num_kblocks))
        
        # Precompute for each t-sub, which blocks cover it? 
        # This is expensive to build fully, but we can do it on the fly or use a different approach.
        # Given num_tsubs ~ 4k and num_blocks ~ 50k, building a reverse index is feasible.
        # tsub_to_blocks: list of lists, where tsub_to_blocks[i] contains block indices covering t-sub i
        
        if not hasattr(greedy, 'tsub_to_blocks'):
            greedy.tsub_to_blocks = [[] for _ in range(num_tsubs)]
            for bi, ts_list in enumerate(block_tsubs):
                for ti in ts_list:
                    greedy.tsub_to_blocks[ti].append(bi)

        while uncovered and time.monotonic() < deadline - 30:
            # Find block with max gain
            # Gain of block j = number of t-subs in uncovered that are in block_tsubs[j]
            # We can iterate over uncovered t-subs and count how many remaining blocks cover them?
            # No, that's O(|uncovered| * |blocks|).
            
            # Better: For each remaining block, calculate intersection size with uncovered.
            # To speed this up, we can maintain a 'gain' counter for each block.
            # But updating gains when we pick a block is tricky.
            
            # Simple approach: Iterate remaining blocks, compute gain.
            # Since |uncovered| decreases, we can filter blocks that intersect uncovered.
            
            best_block = -1
            best_gain = 0
            
            # Optimization: Only consider blocks that cover at least one uncovered t-sub
            # We can maintain a list of candidate blocks.
            # For simplicity and speed in Python, we'll just iterate all remaining blocks.
            # 50k blocks * ~100-200 iterations (greedy steps) = 5-10 million ops. Should be fast enough.
            
            for bi in remaining_blocks:
                # Count how many t-subs in block_tsubs[bi] are in uncovered
                gain = 0
                for ti in block_tsubs[bi]:
                    if ti in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we can't beat best_gain?
                            # Max possible gain is len(block_tsubs[bi]) = C(7,4)=35
                            # If best_gain is close to 35, we can break early for this block
                            pass
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    if best_gain == len(block_tsubs[bi]):
                        # Can't do better than this block
                        break
            
            if best_block == -1 or best_gain == 0:
                break
                
            selected.append(best_block)
            remaining_blocks.remove(best_block)
            
            # Update uncovered
            for ti in block_tsubs[best_block]:
                if ti in uncovered:
                    uncovered.remove(ti)
                    
            # Optional: Add noise?
            # For now, deterministic greedy is fast and robust.
            
        return selected

    # Local Search: Swap based
    def local_search(block_indices, max_iters=500):
        block_indices = list(block_indices)
        n = len(block_indices)
        if n == 0:
            return block_indices

        counts = Counter()
        for bi in block_indices:
            for ti in block_tsubs[bi]:
                counts[ti] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.monotonic() < deadline - 20:
            improved = False
            iters += 1

            # Try dropping
            for idx in range(len(block_indices)):
                bi = block_indices[idx]
                can_drop = True
                for ti in block_tsubs[bi]:
                    if counts[ti] == 1:
                        can_drop = False
                        break
                if can_drop:
                    for ti in block_tsubs[bi]:
                        counts[ti] -= 1
                    block_indices.pop(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping
            for i in range(len(block_indices)):
                bi = block_indices[i]
                critical = [ti for ti in block_tsubs[bi] if counts[ti] == 1]
                if not critical:
                    continue
                
                critical_set = set(critical)
                # Find a block that covers all critical t-subs
                # And is not in block_indices
                for j in range(num_kblocks):
                    if j in block_indices:
                        continue
                    
                    # Check if j covers all critical
                    covers = True
                    for ti in critical_set:
                        if ti not in block_tsubs[j]: # This check is slow if block_tsubs[j] is a list
                            # Precompute sets for blocks? Or just check intersection
                            covers = False
                            break
                    if covers:
                        # Perform swap
                        for ti in block_tsubs[bi]:
                            counts[ti] -= 1
                        for ti in block_tsubs[j]:
                            counts[ti] += 1
                        block_indices[i] = j
                        improved = True
                        break
                if improved:
                    break

        return block_indices

    # Precompute sets for blocks to speed up local search coverage check
    block_tsub_sets = [set(lst) for lst in block_tsubs]

    # Modify local search to use sets
    def local_search_fast(block_indices, max_iters=500):
        block_indices = list(block_indices)
        n = len(block_indices)
        if n == 0:
            return block_indices

        counts = Counter()
        for bi in block_indices:
            for ti in block_tsub_sets[bi]:
                counts[ti] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.monotonic() < deadline - 20:
            improved = False
            iters += 1

            # Try dropping
            for idx in range(len(block_indices)):
                bi = block_indices[idx]
                can_drop = True
                for ti in block_tsub_sets[bi]:
                    if counts[ti] == 1:
                        can_drop = False
                        break
                if can_drop:
                    for ti in block_tsub_sets[bi]:
                        counts[ti] -= 1
                    block_indices.pop(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping
            for i in range(len(block_indices)):
                bi = block_indices[i]
                critical = [ti for ti in block_tsub_sets[bi] if counts[ti] == 1]
                if not critical:
                    continue
                
                critical_set = set(critical)
                for j in range(num_kblocks):
                    if j in block_indices:
                        continue
                    
                    # Check if j covers all critical
                    if critical_set.issubset(block_tsub_sets[j]):
                        # Perform swap
                        for ti in block_tsub_sets[bi]:
                            counts[ti] -= 1
                        for ti in block_tsub_sets[j]:
                            counts[ti] += 1
                        block_indices[i] = j
                        improved = True
                        break
                if improved:
                    break

        return block_indices

    # Run multiple greedy restarts
    best_blocks = None
    best_count = float('inf')
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.monotonic() > deadline - 40:
            break
        
        seed = 42 + r * 7
        # Use deterministic greedy first, then maybe randomized if we add noise logic
        # For now, just run greedy. To get diversity, we could shuffle the order of blocks considered
        # or use a different tie-breaking. 
        # Simple diversity: skip some blocks in early iterations? No, greedy is greedy.
        # Let's just run the same greedy. It's deterministic.
        # To get different results, we can try "randomized greedy" where we pick from top 5 gains.
        
        sel = greedy(seed=seed)
        
        if not sel:
            continue
            
        sel = prune(sel)
        
        if is_complete(sel):
            # Local search
            sel_ls = local_search_fast(sel, max_iters=200)
            if is_complete(sel_ls):
                if len(sel_ls) < best_count:
                    best_count = len(sel_ls)
                    best_blocks = sel_ls
                    sys.stderr.write(f"Restart {r}: {best_count} blocks\n")
            else:
                # Fallback to pre-LS if LS broke it (shouldn't happen if LS is correct)
                if len(sel) < best_count:
                    best_count = len(sel)
                    best_blocks = sel

    if best_blocks is None:
        # Fallback
        sel = greedy(seed=0)
        sel = prune(sel)
        best_blocks = sel
        best_count = len(sel)

    return [list(all_kblocks[i]) for i in best_blocks]
