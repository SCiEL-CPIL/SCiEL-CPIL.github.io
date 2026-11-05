import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118  # Safety margin for 120s limit
    
    # Precompute all t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Represent coverage as bitsets.
    # n_subsets = 3876. 3876 / 64 = 60.56 -> 61 words of 64 bits.
    n_words = (n_subsets + 63) // 64
    
    # block_masks: shape (n_blocks, n_words) of uint64
    # Each entry is a 64-bit mask of which t-subsets are covered by that block.
    # Note: A block of size k covers C(k, t) subsets.
    # For v=19, k=7, t=4: C(7,4) = 35 subsets per block.
    
    block_masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    # Fill block_masks
    # We can do this faster by iterating blocks and setting bits.
    for bi, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            word_idx = si // 64
            bit_idx = si % 64
            block_masks[bi, word_idx] |= (1 << bit_idx)
            
    # Helper to convert a list of block indices to a coverage count array
    # counts[w] = number of blocks covering the subset corresponding to bit in word w?
    # No, we need counts per subset.
    # Let's maintain `counts` as an array of shape (n_words, 64) or just flatten?
    # Actually, maintaining counts per subset is hard with bitsets directly in numpy without unpacking.
    # Alternative: Maintain `uncovered_mask` (bitsets of uncovered subsets).
    # And `coverage_counts` is hard to vectorize easily for "is count 1?".
    
    # Let's use a different representation for the local search state:
    # `counts`: np.array of shape (n_subsets,) int32.
    # This is 3876 * 4 bytes = ~15KB. Very small.
    # Updating counts when adding/removing a block:
    # We need the list of subsets covered by a block.
    
    # Precompute subset lists for each block for fast count updates?
    # block_subsets[bi] = list of subset indices covered by block bi.
    # Total entries: n_blocks * C(k,t) = 50388 * 35 ~ 1.7M integers.
    # This is feasible.
    
    block_subsets = []
    for bi in range(n_blocks):
        subs = []
        for sub in itertools.combinations(all_blocks[bi], t):
            subs.append(sub_idx[sub])
        block_subsets.append(np.array(subs, dtype=np.int32))
        
    # Convert to a flat array with offsets for faster access?
    # Or just keep list of arrays.
    
    def greedy_init(rng):
        # Randomized greedy: at each step, pick block with max gain, 
        # but among ties, pick randomly.
        uncovered_mask = np.ones(n_words, dtype=np.uint64)
        # Initialize uncovered_mask to all 1s.
        # Last word might have extra bits, mask them out.
        last_word_bits = n_subsets % 64
        if last_word_bits == 0:
             last_word_bits = 64
        if n_words > 0:
            # Mask out bits beyond n_subsets in the last word
            mask_last = (1 << last_word_bits) - 1
            uncovered_mask[-1] = mask_last
        
        chosen = []
        chosen_set = set()
        
        # Precompute gains? No, dynamic.
        # To speed up, we can maintain a priority queue or just scan.
        # Scanning 50k blocks * 61 words is 3M ops per iteration.
        # Greedy takes ~150 iterations. 150 * 3M = 450M ops.
        # In Python, this is too slow.
        
        # We need a faster greedy.
        # Vectorized gain calculation:
        # gain[bi] = popcount(block_masks[bi] & uncovered_mask)
        
        while True:
            if time.time() > deadline - 10:
                break
            if not any(uncovered_mask):
                break
                
            # Calculate gains for all blocks
            # uncovered_mask is shape (n_words,)
            # block_masks is shape (n_blocks, n_words)
            # intersection = block_masks & uncovered_mask[None, :]
            # popcount = np.bitwise_count(intersection).sum(axis=1)
            
            # np.bitwise_count is available in numpy 1.17+? 
            # Actually, np.bitwise_count is not in standard numpy for arrays of uint64 directly in older versions?
            # Wait, numpy has `np.bitwise_count` for integers, but for arrays?
            # Let's check. `np.bitwise_count` works on arrays.
            # If not, we can use a lookup table or loop.
            # Assuming numpy >= 1.17.
            
            # To handle the last word masking correctly in gains:
            # The bits in the last word beyond n_subsets are 0 in block_masks (since they don't correspond to real subsets).
            # So intersection is safe.
            
            try:
                # This creates a (n_blocks, n_words) array
                inter = block_masks & uncovered_mask[None, :]
                gains = np.bitwise_count(inter).sum(axis=1)
            except AttributeError:
                # Fallback for older numpy
                gains = np.zeros(n_blocks, dtype=np.int32)
                for w in range(n_words):
                    if uncovered_mask[w] != 0:
                        # Count bits in block_masks[:, w] & uncovered_mask[w]
                        # This is slow in python loop. 
                        # Let's assume modern numpy.
                        pass

            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max_gain
            candidates = np.where(gains == max_gain)[0]
            # Pick randomly from candidates
            bi = candidates[rng.randint(0, len(candidates)-1)]
            
            chosen.append(bi)
            chosen_set.add(bi)
            
            # Update uncovered_mask
            uncovered_mask &= ~block_masks[bi]
            
        return chosen

    def prune(blocks_list):
        # Remove redundant blocks
        if not blocks_list:
            return []
            
        chosen = list(blocks_list)
        counts = np.zeros(n_subsets, dtype=np.int32)
        
        for bi in chosen:
            counts[block_subsets[bi]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                bi = chosen[i]
                subs = block_subsets[bi]
                # Check if any subset has count 1
                if np.any(counts[subs] == 1):
                    continue
                # Redundant
                counts[subs] -= 1
                chosen.pop(i)
                changed = True
                break
                
        return chosen

    def local_search(blocks_list, time_budget):
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in current_blocks:
            counts[block_subsets[bi]] += 1
            
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        t_start = time.time()
        
        # Precompute for each block, the list of subsets it covers (already have block_subsets)
        # To speed up swap search, we need to know which blocks cover specific subsets.
        # subset_to_blocks: list of lists.
        # subset_to_blocks[si] = list of block indices covering si.
        # Size: sum(C(k,t)) = 1.7M entries.
        # Building this takes time but enables fast lookup.
        
        # Let's build it once globally? No, inside function to keep scope clean, 
        # but it's expensive. Let's do it outside if possible, or cache.
        # For simplicity and given 120s limit, let's build it here if time permits.
        
        if time.time() - t_start > time_budget:
            return best_blocks
            
        # Build reverse index
        subset_to_blocks = [[] for _ in range(n_subsets)]
        for bi in range(n_blocks):
            for si in block_subsets[bi]:
                subset_to_blocks[si].append(bi)
                
        # Convert to numpy arrays for faster iteration?
        # subset_to_blocks_np = [np.array(lst) for lst in subset_to_blocks]
        
        while time.time() - t_start < time_budget:
            # 1. Try random removals
            removed_any = False
            # Try a few random removals
            for _ in range(20):
                if not current_blocks:
                    break
                idx = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx]
                subs = block_subsets[bi_out]
                
                # Check if any count is 1
                if np.any(counts[subs] == 1):
                    continue
                    
                # Valid removal
                counts[subs] -= 1
                current_blocks.pop(idx)
                current_set.discard(bi_out)
                removed_any = True
                break
                
            if removed_any:
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
                
            # 2. Try swaps
            # Pick a random block to remove
            if not current_blocks:
                break
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            subs_out = block_subsets[bi_out]
            
            # Find critical subsets (count 1)
            critical_mask = (counts[subs_out] == 1)
            if not np.any(critical_mask):
                # Should have been removed
                continue
                
            critical_subs = subs_out[critical_mask]
            
            # We need a block B that covers ALL critical_subs
            # Intersection of subset_to_blocks for each critical sub
            # Start with all blocks, intersect with those covering each critical sub
            
            # Optimization: Pick the critical sub with the fewest covering blocks
            min_len = float('inf')
            min_sub = -1
            for si in critical_subs:
                l = len(subset_to_blocks[si])
                if l < min_len:
                    min_len = l
                    min_sub = si
                    
            if min_sub == -1:
                continue
                
            candidates = set(subset_to_blocks[min_sub])
            
            for si in critical_subs:
                if si == min_sub:
                    continue
                candidates &= set(subset_to_blocks[si])
                if not candidates:
                    break
                    
            if not candidates:
                continue
                
            # Filter candidates: must not be in current_set
            candidates -= current_set
            
            if not candidates:
                continue
                
            # Score candidates: 
            # We want to minimize the "cost" of the swap.
            # Cost = Number of subsets in B that are currently covered by exactly 1 block (other than A)?
            # Or simply: prefer B that covers many subsets with count > 1 (redundant), 
            # so that removing A doesn't create new criticals, and B adds redundancy.
            
            best_b = -1
            best_score = -1
            
            # Limit candidates to check for speed
            cand_list = list(candidates)
            if len(cand_list) > 50:
                cand_list = rng.sample(cand_list, 50)
                
            for bi_in in cand_list:
                subs_in = block_subsets[bi_in]
                # Check if it covers all criticals (already done by intersection, but double check?)
                # It does by construction.
                
                # Score: number of subsets in subs_in where count > 1
                # Note: counts[subs_in] reflects current state (before swap).
                # If we remove A, counts for subs_out drop by 1.
                # If we add B, counts for subs_in rise by 1.
                
                # Simple heuristic: Maximize overlap with already redundant subsets.
                score = np.sum(counts[subs_in] > 1)
                
                if score > best_score:
                    best_score = score
                    best_b = bi_in
                    
            if best_b != -1:
                # Perform swap
                counts[subs_out] -= 1
                current_set.discard(bi_out)
                
                subs_in = block_subsets[best_b]
                counts[subs_in] += 1
                current_set.add(best_b)
                
                current_blocks[idx] = best_b
                
        return best_blocks

    # Main Search
    # We will run multiple iterations of ILS
    
    best_solution = []
    best_size = float('inf')
    
    # Pre-calculate block_subsets if not done? 
    # It's done inside local_search currently. Let's move it out to avoid rebuilding.
    # Actually, block_subsets is built in the main scope? No, I defined it inside local_search.
    # Let's move block_subsets creation to main scope.
    
    # Re-create block_subsets in main scope for efficiency
    block_subsets_main = []
    for bi in range(n_blocks):
        subs = []
        for sub in itertools.combinations(all_blocks[bi], t):
            subs.append(sub_idx[sub])
        block_subsets_main.append(np.array(subs, dtype=np.int32))
        
    # And subset_to_blocks
    subset_to_blocks_main = [[] for _ in range(n_subsets)]
    for bi in range(n_blocks):
        for si in block_subsets_main[bi]:
            subset_to_blocks_main[si].append(bi)
            
    # Update local_search to use these global-like variables
    # Since Python closures capture by reference, we can just define local_search to use them.
    # But I defined local_search inside construct. It can access block_subsets_main if I define it before.
    
    # Let's redefine the flow:
    
    # 1. Greedy Init
    init_blocks = greedy_init(rng)
    pruned_init = prune(init_blocks)
    best_solution = pruned_init
    best_size = len(pruned_init)
    
    # 2. ILS Loop
    # We have time for multiple restarts.
    # Each restart: Greedy (fast) -> Local Search (fast)
    
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        # Randomize greedy by using a different seed or just the same rng
        # Greedy is fast enough to run multiple times?
        # Greedy takes ~0.5s?
        # Let's check.
        
        curr_init = greedy_init(rng)
        curr_pruned = prune(curr_init)
        
        # Local Search
        # Pass the precomputed structures
        ls_result = local_search(curr_pruned, time_budget=min(5, deadline - time.time() - 2))
        ls_pruned = prune(ls_result)
        
        if len(ls_pruned) < best_size:
            best_size = len(ls_pruned)
            best_solution = ls_pruned
            
        # If we find a very good solution, maybe stop?
        # Best known is 151. If we hit 151, we can keep trying for 150.
        
    return [list(all_blocks[bi]) for bi in best_solution]

# We need to fix the scope of block_subsets and subset_to_blocks for local_search.
# The current implementation of local_search rebuilds them or assumes they exist.
# I will refactor to ensure they are passed or available.

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    
    rng = random.Random(1234567890)

    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    n_words = (n_subsets + 63) // 64
    block_masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    for bi, blk in enumerate(all_blocks):
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            word_idx = si // 64
            bit_idx = si % 64
            block_masks[bi, word_idx] |= (1 << bit_idx)
            
    block_subsets = []
    for bi in range(n_blocks):
        subs = []
        for sub in itertools.combinations(all_blocks[bi], t):
            subs.append(sub_idx[sub])
        block_subsets.append(np.array(subs, dtype=np.int32))
        
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for bi in range(n_blocks):
        for si in block_subsets[bi]:
            subset_to_blocks[si].append(bi)
            
    def greedy_init():
        uncovered_mask = np.ones(n_words, dtype=np.uint64)
        last_word_bits = n_subsets % 64
        if last_word_bits == 0:
            last_word_bits = 64
        if n_words > 0:
            mask_last = (1 << last_word_bits) - 1
            uncovered_mask[-1] = mask_last
            
        chosen = []
        
        while True:
            if time.time() > deadline - 10:
                break
            if not any(uncovered_mask):
                break
                
            try:
                inter = block_masks & uncovered_mask[None, :]
                gains = np.bitwise_count(inter).sum(axis=1)
            except AttributeError:
                # Fallback if np.bitwise_count not available
                gains = np.zeros(n_blocks, dtype=np.int32)
                # This is slow, but hopefully not needed
                for bi in range(n_blocks):
                    g = 0
                    for w in range(n_words):
                        if uncovered_mask[w]:
                            g += (block_masks[bi, w] & uncovered_mask[w]).bit_count()
                    gains[bi] = g

            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            bi = candidates[rng.randint(0, len(candidates)-1)]
            
            chosen.append(bi)
            uncovered_mask &= ~block_masks[bi]
            
        return chosen

    def prune(blocks_list):
        if not blocks_list:
            return []
        chosen = list(blocks_list)
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in chosen:
            counts[block_subsets[bi]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                bi = chosen[i]
                subs = block_subsets[bi]
                if np.any(counts[subs] == 1):
                    continue
                counts[subs] -= 1
                chosen.pop(i)
                changed = True
                break
        return chosen

    def local_search(blocks_list, time_budget):
        current_blocks = list(blocks_list)
        current_set = set(current_blocks)
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in current_blocks:
            counts[block_subsets[bi]] += 1
            
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            removed_any = False
            for _ in range(20):
                if not current_blocks:
                    break
                idx = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx]
                subs = block_subsets[bi_out]
                
                if np.any(counts[subs] == 1):
                    continue
                    
                counts[subs] -= 1
                current_blocks.pop(idx)
                current_set.discard(bi_out)
                removed_any = True
                break
                
            if removed_any:
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
                
            if not current_blocks:
                break
            idx = rng.randrange(len(current_blocks))
            bi_out = current_blocks[idx]
            subs_out = block_subsets[bi_out]
            
            critical_mask = (counts[subs_out] == 1)
            if not np.any(critical_mask):
                continue
                
            critical_subs = subs_out[critical_mask]
            
            min_len = float('inf')
            min_sub = -1
            for si in critical_subs:
                l = len(subset_to_blocks[si])
                if l < min_len:
                    min_len = l
                    min_sub = si
                    
            if min_sub == -1:
                continue
                
            candidates = set(subset_to_blocks[min_sub])
            
            for si in critical_subs:
                if si == min_sub:
                    continue
                candidates &= set(subset_to_blocks[si])
                if not candidates:
                    break
                    
            if not candidates:
                continue
                
            candidates -= current_set
            
            if not candidates:
                continue
                
            best_b = -1
            best_score = -1
            
            cand_list = list(candidates)
            if len(cand_list) > 50:
                cand_list = rng.sample(cand_list, 50)
                
            for bi_in in cand_list:
                subs_in = block_subsets[bi_in]
                score = np.sum(counts[subs_in] > 1)
                
                if score > best_score:
                    best_score = score
                    best_b = bi_in
                    
            if best_b != -1:
                counts[subs_out] -= 1
                current_set.discard(bi_out)
                
                subs_in = block_subsets[best_b]
                counts[subs_in] += 1
                current_set.add(best_b)
                
                current_blocks[idx] = best_b
                
        return best_blocks

    best_solution = []
    best_size = float('inf')
    
    # Initial Greedy
    init_blocks = greedy_init()
    pruned_init = prune(init_blocks)
    best_solution = pruned_init
    best_size = len(pruned_init)
    
    # ILS Loop
    while time.time() < deadline - 2:
        curr_init = greedy_init()
        curr_pruned = prune(curr_init)
        
        # Allocate remaining time to local search, min 1s, max 5s per LS
        time_left = deadline - time.time() - 2
        ls_budget = min(5, time_left)
        
        ls_result = local_search(curr_pruned, time_budget=ls_budget)
        ls_pruned = prune(ls_result)
        
        if len(ls_pruned) < best_size:
            best_size = len(ls_pruned)
            best_solution = ls_pruned
            
    return [list(all_blocks[bi]) for bi in best_solution]
