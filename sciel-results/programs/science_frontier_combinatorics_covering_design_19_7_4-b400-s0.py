import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered_tsubs

    def greedy_init():
        # Standard Greedy Set Cover
        uncovered = [1] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        while n_uncovered > 0:
            best_block = None
            best_gain = -1
            
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None or best_gain == 0:
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return blocks

    def local_search(blocks):
        # Returns improved blocks
        current_blocks = list(blocks)
        current_block_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        def is_valid():
            # Check if any coverage is 0
            # This is O(n_tsubsets)
            for c in coverage:
                if c == 0:
                    return False
            return True

        def is_redundant(b):
            # A block is redundant if every t-subset it covers is covered by >1 block
            for ts_idx in block_to_tsubs[b]:
                if coverage[ts_idx] == 1:
                    return False
            return True

        def remove_block(b):
            current_block_set.remove(b)
            current_blocks.remove(b)
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] -= 1

        def add_block(b):
            current_block_set.add(b)
            current_blocks.append(b)
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1

        # Phase 1: Remove redundants
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal during iteration? 
            # Or just make a list of candidates.
            candidates = list(current_blocks)
            for b in candidates:
                if is_redundant(b):
                    remove_block(b)
                    changed = True
                    if (time.time() - start_time) > time_limit:
                        return current_blocks

        # Phase 2: 1-Exchange (Swap)
        # Try to swap each block with a better one
        # "Better" means: after swap, we can remove more redundants?
        # Or simply: after swap, the set is valid, and we run redundancy removal.
        # If the size decreases, accept.
        
        # To speed up, we can limit swaps.
        # For each block b in current_blocks:
        #   Find a block b_new not in current_block_set
        #   Swap b with b_new
        #   If valid, run redundancy removal
        #   If size < current_size, keep the change
        #   Else revert
        
        # This is O(N * M * Cost(Redundancy))
        # N ~ 150, M ~ 50000. Too slow if we try all M.
        # We need a heuristic for b_new.
        # Pick b_new that covers many t-subsets that are currently "critical" (covered only by b)?
        # Or just random sampling?
        
        # Let's try a limited 1-exchange:
        # For each block b:
        #   Identify t-subsets covered ONLY by b (critical for b).
        #   b_new must cover all these critical t-subsets? No, b_new replaces b, so b_new must cover all t-subsets that b covered AND are not covered by others?
        #   Actually, if we remove b, the critical t-subsets become uncovered.
        #   So b_new MUST cover all critical t-subsets of b.
        #   If no such b_new exists, we can't swap b directly to a valid state without adding another block?
        #   Wait, the set must remain valid.
        #   So b_new must cover all ts in (S_b \ S_others).
        
        # Let's implement this constraint.
        for _ in range(10): # Repeat passes
            if (time.time() - start_time) > time_limit:
                break
            improved = False
            # Shuffle to avoid bias
            blocks_to_try = current_blocks[:]
            random.shuffle(blocks_to_try)
            
            for b in blocks_to_try:
                if b not in current_block_set:
                    continue
                    
                # Find critical t-subsets for b
                critical_ts = []
                for ts_idx in block_to_tsubs[b]:
                    if coverage[ts_idx] == 1:
                        critical_ts.append(ts_idx)
                
                # If no critical ts, b is redundant (should have been removed in Phase 1)
                # But maybe it became critical after other swaps?
                # Phase 1 ran before. So if we are here, b has at least one critical ts?
                # Not necessarily. If we swapped other blocks, coverage counts changed.
                # So we must check.
                
                if not critical_ts:
                    # b is redundant now
                    remove_block(b)
                    improved = True
                    # Re-run redundancy removal? Or just continue.
                    # Let's break and restart Phase 1? 
                    # For simplicity, just mark improved and continue loop.
                    # But we should probably re-run Phase 1 to catch cascading redundancies.
                    # Let's just break out and restart the whole local_search?
                    # That might be slow.
                    # Let's just remove it and continue.
                    pass
                else:
                    # Find a candidate b_new that covers all critical_ts
                    # To avoid O(M) scan, we can intersect tsub_to_blocks for each critical_ts
                    # Candidates = intersection of tsub_to_blocks[ts] for ts in critical_ts
                    # Exclude current_block_set
                    
                    candidates = None
                    for ts_idx in critical_ts:
                        blocks_covering = tsub_to_blocks[ts_idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                    
                    if candidates:
                        candidates -= current_block_set
                        if candidates:
                            # Pick a random candidate
                            b_new = random.choice(list(candidates))
                            
                            # Try swap
                            # Save state? 
                            # Just do it, check validity (it should be valid by construction 
                            # if b_new covers all critical ts of b, and other ts of b are covered by others?
                            # Wait. If ts is in b but NOT critical, coverage[ts] >= 2.
                            # If we remove b, coverage[ts] becomes >= 1. So still covered.
                            # So yes, swapping b with b_new (which covers critical_ts) results in a valid set.
                            
                            # Perform swap
                            remove_block(b)
                            add_block(b_new)
                            
                            # Try to remove redundants
                            # Quick check: run Phase 1 logic
                            changed = True
                            while changed:
                                changed = False
                                for bb in list(current_blocks):
                                    if is_redundant(bb):
                                        remove_block(bb)
                                        changed = True
                                        
                            improved = True
                            # If size didn't decrease, we might want to revert?
                            # The swap itself doesn't change size.
                            # But redundancy removal might.
                            # If no redundancy was removed, size is same.
                            # We might be stuck in a local minimum of size.
                            # But we accepted the swap.
                            # This is a valid move.
                            
            if not improved:
                break
                
        return current_blocks

    # Run ILS
    best_blocks = greedy_init()
    best_size = len(best_blocks)
    
    # Local search on initial
    current_blocks = local_search(best_blocks)
    if len(current_blocks) < best_size:
        best_blocks = current_blocks
        best_size = len(current_blocks)
        
    # ILS Loop
    while (time.time() - start_time) < time_limit:
        # Perturbation
        # Pick a random block b
        # Find critical ts of b
        # Find b_new covering critical ts
        # Swap
        # If valid, local search
        
        b = random.choice(current_blocks)
        
        # Find critical ts
        coverage = [0] * n_tsubsets
        for bb in current_blocks:
            for ts_idx in block_to_tsubs[bb]:
                coverage[ts_idx] += 1
        
        critical_ts = []
        for ts_idx in block_to_tsubs[b]:
            if coverage[ts_idx] == 1:
                critical_ts.append(ts_idx)
        
        if not critical_ts:
            # b is redundant, remove it
            current_blocks.remove(b)
            # Local search
            new_blocks = local_search(current_blocks)
            if len(new_blocks) < best_size:
                best_blocks = new_blocks
                best_size = len(new_blocks)
                current_blocks = new_blocks
            continue
            
        # Find candidates for b_new
        candidates = None
        for ts_idx in critical_ts:
            blocks_covering = tsub_to_blocks[ts_idx]
            if candidates is None:
                candidates = set(blocks_covering)
            else:
                candidates.intersection_update(blocks_covering)
        
        if not candidates:
            # Cannot swap b directly
            # Try removing b and adding 2 blocks? 
            # Or just skip perturbation for this b
            continue
            
        candidates -= set(current_blocks)
        if not candidates:
            continue
            
        b_new = random.choice(list(candidates))
        
        # Perform swap on a copy
        temp_blocks = list(current_blocks)
        temp_blocks.remove(b)
        temp_blocks.append(b_new)
        
        # Local search
        new_blocks = local_search(temp_blocks)
        
        if len(new_blocks) < best_size:
            best_blocks = new_blocks
            best_size = len(new_blocks)
            current_blocks = new_blocks
        else:
            # Even if not best, we can accept if it's the same size?
            # To explore, maybe accept same size with some probability?
            # Or just keep current?
            # Let's keep current to maintain momentum, or accept same size to diversify.
            # Let's accept same size to allow further perturbations to find a dip.
            if len(new_blocks) <= best_size:
                 current_blocks = new_blocks

    return [list(b) for b in best_blocks]
