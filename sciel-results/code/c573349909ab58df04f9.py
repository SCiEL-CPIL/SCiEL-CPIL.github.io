import itertools
import random
import time
import sys
import numpy as np

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_triples[i] is a list of triple indices in block i
    block_triples = [[] for _ in range(n_blocks)]
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            block_triples[i].append(idx)
            blocks_containing[idx].append(i)
            
    # Precompute mask for quick coverage check? 
    # For v=24, k=7, t=3, n_triples=2024. 
    # Python ints are fine for 2024 bits.
    block_masks = [0] * n_blocks
    for i in range(n_blocks):
        mask = 0
        for tri_idx in block_triples[i]:
            mask |= (1 << tri_idx)
        block_masks[i] = mask
            
    _CACHE[key] = (all_blocks, block_masks, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, blocks_containing, n_blocks, n_triples, block_triples, seed):
    rng = random.Random(seed)
    uncovered = (1 << n_triples) - 1
    blocks_used = []
    active_blocks = set(range(n_blocks))
    
    while uncovered:
        best_block_idx = -1
        max_gain = 0
        # To speed up, we can sample if n_blocks is huge, but 346k is manageable for a few iterations
        # However, iterating 346k blocks * 76 times is ~26M ops, which is fast in Python if optimized?
        # bit_count is fast.
        
        # Optimization: Instead of scanning all active blocks, we can pick a random uncovered triple 
        # and only look at blocks containing it. This is the "Greedy by Rarest" heuristic.
        
        # Pick a random uncovered triple
        # Get an index of an uncovered bit
        # We can just pick the first one or random. Random is better for diversity.
        # But finding a random set bit is O(n_triples).
        # Let's just pick the first uncovered triple to be deterministic-ish but fast?
        # Or better: maintain a list of uncovered triples? 
        # Actually, scanning all blocks is O(N_blocks). 
        # N_blocks = 346,104.
        # 76 * 346k = 26 million iterations. In Python, this might take ~5-10 seconds.
        # Let's try to optimize by picking a random uncovered triple and only checking its blocks.
        
        # Find an uncovered triple index
        # We can iterate through uncovered bits?
        # Simpler: Just pick a random index from 0..n_triples-1 until we find one in uncovered?
        # Or just use the first one.
        
        # Let's use the "Rarest First" approach properly.
        # 1. Find an uncovered triple.
        # 2. Get all blocks containing it.
        # 3. Among those, pick the one with max gain.
        
        # How to find an uncovered triple quickly?
        # We can maintain a list `uncovered_list`.
        # Initially list(range(n_triples)).
        # When a block is added, remove covered triples from `uncovered_list`.
        
        # This requires updating `uncovered_list` efficiently.
        # Using a set `uncovered_set` is O(1) removal.
        
        uncovered_set = set(range(n_triples))
        
        while uncovered_set:
            # Pick a random uncovered triple to anchor our search
            tri_idx = next(iter(uncovered_set)) # Arbitrary pick, or random
            # To be more random:
            # tri_idx = random.choice(list(uncovered_set)) -- Too slow to convert to list every time
            
            # Let's just pick the first one for now, or keep a list.
            # Actually, the standard greedy scans all blocks. 
            # The "anchored" greedy is often better.
            
            candidates = blocks_containing[tri_idx]
            best_gain = 0
            best_candidate = -1
            
            for b_idx in candidates:
                if b_idx not in active_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = b_idx
            
            if best_candidate == -1:
                break # Should not happen if problem is solvable
            
            blocks_used.append(best_candidate)
            active_blocks.remove(best_candidate)
            
            # Update uncovered
            new_uncovered = uncovered & ~block_masks[best_candidate]
            # Remove covered triples from uncovered_set
            covered_in_block = block_masks[best_candidate] & uncovered
            # Iterate set bits in covered_in_block
            # This is a bit expensive to do bit by bit.
            # Alternative: Just recompute uncovered_set? No.
            # We can just update the mask `uncovered` and not maintain the set, 
            # and pick a random triple from the full list, checking if it's uncovered?
            # That's O(1) per check.
            
            # Let's stick to the simple scan-all-blocks greedy if it's fast enough, 
            # or use the anchored one.
            # Anchored is O(Deg(triple)) per step. Deg is ~ (24-3 choose 7-3) / (24 choose 7) * 346k?
            # Each triple is in C(21,4) = 5985 blocks.
            # 5985 * 76 = 450k ops. Much faster than 26M.
            
            # So we MUST use anchored greedy.
            # But we need to pick a "good" uncovered triple. 
            # Picking the first one in the set is fine.
            
            # Update uncovered mask
            uncovered = new_uncovered
            
            # Update uncovered_set
            # We need to remove all bits in (block_masks[best_candidate] & old_uncovered) from uncovered_set
            # Let's just rebuild uncovered_set if it gets too small? 
            # Or iterate through block_triples[best_candidate] and remove from set.
            for t_idx in block_triples[best_candidate]:
                if t_idx in uncovered_set:
                    uncovered_set.remove(t_idx)
            
            # Optimization: If the set is large, `next(iter(set))` is O(1).
            # But `set.remove` is O(1).
            # This looks good.

        return blocks_used

def _remove_redundant_blocks(selected_blocks, block_triples, n_triples, blocks_containing, block_masks, n_blocks):
    # Remove any block that has no unique triples
    # A block is redundant if for every triple t in it, there is another block in S containing t.
    # i.e., coverage_count[t] > 1 for all t in block.
    
    selected = set(selected_blocks)
    if not selected:
        return selected
    
    coverage_count = np.zeros(n_triples, dtype=np.int32)
    for b in selected:
        for t_idx in block_triples[b]:
            coverage_count[t_idx] += 1
            
    changed = True
    while changed:
        changed = False
        for b in list(selected):
            is_redundant = True
            for t_idx in block_triples[b]:
                if coverage_count[t_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove b
                for t_idx in block_triples[b]:
                    coverage_count[t_idx] -= 1
                selected.remove(b)
                changed = True
                
    return selected

def _local_search_2_1_swap(selected_blocks, block_triples, n_triples, blocks_containing, block_masks, n_blocks, seed, deadline):
    rng = random.Random(seed)
    selected = set(selected_blocks)
    
    # Compute coverage counts
    coverage_count = np.zeros(n_triples, dtype=np.int32)
    for b in selected:
        for t_idx in block_triples[b]:
            coverage_count[t_idx] += 1
            
    best_size = len(selected)
    best_selected = set(selected)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try 2-1 swaps
        # Pick a random pair of blocks
        sel_list = list(selected)
        if len(sel_list) < 2:
            break
            
        # Shuffle to avoid bias
        rng.shuffle(sel_list)
        
        found_swap = False
        for i in range(len(sel_list)):
            if time.time() > deadline:
                break
            b1 = sel_list[i]
            
            # Find unique triples in b1
            unique_b1 = [t for t in block_triples[b1] if coverage_count[t] == 1]
            
            for j in range(i + 1, len(sel_list)):
                if time.time() > deadline:
                    break
                b2 = sel_list[j]
                
                # Find unique triples in b2 (considering b1 is still there)
                unique_b2 = [t for t in block_triples[b2] if coverage_count[t] == 1]
                
                # Combined unique triples
                all_unique = set(unique_b1) | set(unique_b2)
                
                if not all_unique:
                    # Both are redundant?
                    # Just remove one?
                    # If all_unique is empty, it means no triple is uniquely covered by b1 or b2.
                    # So we can remove one of them safely.
                    # Let's remove b1.
                    for t in block_triples[b1]:
                        coverage_count[t] -= 1
                    selected.remove(b1)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    found_swap = True
                    break
                    
                # We need a block b_new that covers all triples in all_unique
                # Pick the triple in all_unique with the fewest candidate blocks
                # to minimize search space
                
                # Get candidates for the first triple
                # Better: pick the rarest triple in all_unique
                min_deg = float('inf')
                rarest_tri = None
                for t in all_unique:
                    deg = len(blocks_containing[t])
                    if deg < min_deg:
                        min_deg = deg
                        rarest_tri = t
                        if deg == 1:
                            break
                
                if rarest_tri is None:
                    continue
                    
                candidates = blocks_containing[rarest_tri]
                
                # Shuffle candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                # Check candidates
                # Limit check to speed up
                for b_new in cand_list[:100]: 
                    if b_new in selected:
                        continue
                        
                    # Check if b_new covers all_unique
                    covers_all = True
                    for t in all_unique:
                        # Check if t is in b_new
                        # Using mask is faster
                        if not (block_masks[b_new] & (1 << t)):
                            covers_all = False
                            break
                            
                    if not covers_all:
                        continue
                        
                    # Perform swap
                    # Remove b1, b2
                    for t in block_triples[b1]:
                        coverage_count[t] -= 1
                    for t in block_triples[b2]:
                        coverage_count[t] -= 1
                    selected.remove(b1)
                    selected.remove(b2)
                    
                    # Add b_new
                    for t in block_triples[b_new]:
                        coverage_count[t] += 1
                    selected.add(b_new)
                    
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    found_swap = True
                    break
                    
                if found_swap:
                    break
            
            if found_swap:
                break
        
        if not improved:
            # Try 1-1 swaps to escape local minima (Simulated Annealing style)
            # Or just restart?
            # Let's try a few random 1-1 swaps that maintain validity (or slightly degrade)
            # Actually, if we are stuck, we might need to break out.
            # For now, if no 2-1 swap found, break and let construct() restart with new seed.
            break
            
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    all_blocks, block_masks, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    while time.time() < deadline - 5:
        # 1. Greedy
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, blocks_containing, n_blocks, n_triples, block_triples, seed)
        
        if not initial_blocks:
            seed += 1
            continue
            
        # 2. Remove redundant
        minimal_blocks = _remove_redundant_blocks(initial_blocks, block_triples, n_triples, blocks_containing, block_masks, n_blocks)
        
        if len(minimal_blocks) < best_len:
            best_len = len(minimal_blocks)
            best_blocks = minimal_blocks
            
        # 3. Local Search (2-1 swaps)
        remaining = deadline - time.time()
        # Give local search some time
        ls_deadline = min(time.time() + min(10.0, remaining - 2.0), deadline - 3.0)
        
        if ls_deadline > time.time():
            optimized_blocks = _local_search_2_1_swap(minimal_blocks, block_triples, n_triples, blocks_containing, block_masks, n_blocks, seed, ls_deadline)
            
            if len(optimized_blocks) < best_len:
                best_len = len(optimized_blocks)
                best_blocks = optimized_blocks
                print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
        
        # If we hit the known best, we might want to spend more time
        if best_len <= 76:
             # Spend more time on the last few seconds
             pass

    if best_blocks is None:
        # Fallback
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, blocks_containing, n_blocks, n_triples, block_triples, 999)
        best_blocks = _remove_redundant_blocks(initial_blocks, block_triples, n_triples, blocks_containing, block_masks, n_blocks)
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
