"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    # Pre-compute all t-subsets of each possible k-block? Too many.
    # Instead, use a heuristic: pick a random uncovered subset, extend it greedily.
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block_set = set(seed)
        
        # Greedily add points to maximize coverage of *uncovered* subsets
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # Check all (t-1) subsets of current block
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                # Tie-breaking: random
                if gain > best_gain or (gain == best_gain and best_x is not None and rng.random() < 0.5):
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in blocks:
            blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    total_t = len(list(itertools.combinations(range(v), t)))
    uncovered = total_t - len(counts)
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = frozenset(fs)
            old_c = counts.get(f, 0)
            new_c = old_c + sign
            if new_c == 0:
                if f in counts:
                    del counts[f]
                if old_c == 1:
                    delta_uncovered += 1
                if old_c == 0:
                    delta_uncovered -= 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[f] = new_c
        return delta_uncovered

    def try_rebuild_block(i):
        """
        Try to replace block i with a new block that covers all critical subsets
        of the old block and potentially more uncovered subsets.
        Returns True if successful (valid covering maintained).
        """
        old_block = blocks[i]
        
        # Identify critical subsets (those covered only by this block)
        critical = []
        for fs in itertools.combinations(old_block, t):
            f = frozenset(fs)
            if counts.get(f, 0) == 1:
                critical.append(f)
        
        # If no critical subsets, the block is redundant (handled by removal step usually)
        # But we can still try to improve it.
        
        # Temporarily remove old block
        delta_u_remove = apply_block_change(old_block, -1)
        # uncovered is updated by apply_block_change? No, apply_block_change returns delta.
        # We need to track uncovered manually or use the returned delta.
        # Let's assume apply_block_change updates counts but we track 'uncovered' variable externally.
        # In the current code, 'uncovered' is a local variable in construct.
        # apply_block_change does NOT update the 'uncovered' variable in the scope of construct directly, 
        # it returns the delta. The caller must update 'uncovered'.
        
        # Wait, looking at the original code:
        # delta_u = apply_block_change(b, -1)
        # uncovered += delta_u
        # So apply_block_change only modifies 'counts'.
        
        # So:
        # 1. Apply remove
        # 2. Check if uncovered == 0 (if so, we can just delete, but this function is for replacement)
        # 3. If not, try to build new block covering all newly uncovered subsets.
        
        # The newly uncovered subsets are exactly those with count 0 now.
        # But we specifically care about the ones that were covered by old_block and are now 0.
        # These are the 'critical' ones.
        
        # Let's find all currently uncovered subsets (count 0)
        # This is expensive if we scan all C(v,t). 
        # Better: The sets that became uncovered are those in old_block that had count 1.
        # Let's call this set 'must_cover'.
        must_cover = set(critical)
        
        # If must_cover is empty, we can just delete? 
        # If must_cover is empty, delta_u_remove would be 0 (or negative if we uncovered something? No, removing a block only increases uncovered).
        # If must_cover is empty, it means all subsets of old_block had count > 1. So removing it leaves no new uncovered subsets.
        # In that case, we can just delete it.
        if not must_cover:
             # We can just delete
             # But this function is "rebuild". If we can delete, that's better.
             # However, the main loop handles deletion. 
             # If we get here, it means deletion wasn't tried or failed?
             # Actually, if must_cover is empty, deletion is valid.
             # Let's just return False to let the deletion step handle it, or handle it here?
             # Let's handle it: if we can delete, we should.
             # But the caller expects a "replacement" or "failure".
             # If we delete, we change len(blocks).
             # Let's stick to replacement to keep logic simple.
             # If must_cover is empty, we can replace with ANY block? No, that adds a block.
             # If must_cover is empty, we should just delete.
             # Let's signal the caller to delete.
             # For now, let's just try to find a replacement that covers must_cover.
             # If must_cover is empty, any block works? No, we want to minimize blocks.
             # If must_cover is empty, we should delete.
             # Let's add a flag or handle deletion in the main loop.
             # To avoid complexity, let's only try to rebuild if must_cover is not empty.
             # If must_cover is empty, we revert and let the deletion step catch it.
             apply_block_change(old_block, 1)
             return False

        # Greedily construct a new block to cover must_cover
        # Start with a random subset from must_cover
        seed = rng.choice(list(must_cover))
        new_block_set = set(seed)
        
        # We need to cover all sets in must_cover.
        # Greedy: at each step, add the point that covers the most uncovered sets in must_cover 
        # (and potentially other uncovered sets).
        
        # Optimization: Precompute which points are in which critical sets?
        # Since must_cover is small (usually), we can just iterate.
        
        while len(new_block_set) < k:
            best_x = None
            best_score = -1
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            for x in cands:
                # Count how many sets in must_cover are covered by adding x
                gain_must = 0
                # Also count gain in total uncovered (optional, but good for future removals)
                gain_other = 0
                
                for rest in itertools.combinations(new_block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in must_cover:
                        gain_must += 1
                    # Check if it covers other uncovered subsets (count 0)
                    # This is expensive. We can skip or sample.
                    # Let's just focus on must_cover for now to ensure validity.
                
                # Tie-breaking: random
                if gain_must > best_score or (gain_must == best_score and best_x is not None and rng.random() < 0.5):
                    best_score = gain_must
                    best_x = x
            
            if best_x is None:
                break
            new_block_set.add(best_x)
        
        new_block = tuple(sorted(new_block_set))
        
        # Check if new_block covers all must_cover
        # And check if new_block is not the same as old_block
        if new_block == old_block:
            apply_block_change(old_block, 1)
            return False
            
        # Verify coverage of must_cover
        covered_must = set()
        for fs in itertools.combinations(new_block, t):
            f = frozenset(fs)
            if f in must_cover:
                covered_must.add(f)
        
        if covered_must != must_cover:
            # Failed to cover all critical subsets
            apply_block_change(old_block, 1)
            return False
        
        # If we reach here, new_block covers all critical subsets.
        # We can replace old_block with new_block.
        # Apply add for new_block
        apply_block_change(new_block, 1)
        
        # Update blocks list
        blocks[i] = new_block
        
        return True

    no_improve_iter = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        
        if removed_any:
            continue
            
        # 2. Try Rebuild (Replacement)
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            # Only try rebuild if the block has critical subsets
            # (If no critical subsets, it's redundant, handled above)
            has_critical = False
            for fs in itertools.combinations(blocks[i], t):
                if counts.get(frozenset(fs), 0) == 1:
                    has_critical = True
                    break
            
            if has_critical:
                success = try_rebuild_block(i)
                if success:
                    # Recalculate uncovered? 
                    # try_rebuild_block maintains validity, so uncovered should remain 0?
                    # No, uncovered is the count of t-subsets with count 0.
                    # Since we maintained validity, uncovered remains 0?
                    # Wait, 'uncovered' variable tracks the number of uncovered subsets.
                    # If the covering was valid before (uncovered == 0), and we maintain validity, uncovered remains 0.
                    # So we don't need to update 'uncovered' if it was 0.
                    # But what if 'uncovered' was not 0? (Shouldn't happen if we always maintain validity).
                    # We start with a valid covering. Removals and Rebuilds maintain validity.
                    # So 'uncovered' should always be 0.
                    # Let's verify:
                    # Initial: valid -> uncovered = 0.
                    # Removal: only if valid -> uncovered = 0.
                    # Rebuild: only if valid -> uncovered = 0.
                    # So 'uncovered' is always 0.
                    # We can remove the 'uncovered' variable and just check counts if needed.
                    # But the code uses it. Let's keep it consistent.
                    no_improve_iter = 0
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                else:
                    no_improve_iter += 1
            else:
                no_improve_iter += 1

        # 3. Perturbation if stuck (Restart-like)
        if no_improve_iter > max_no_improve:
            # Try a more aggressive perturbation: remove 2 blocks and rebuild
            # Or just reset to best and try a different random seed?
            # Let's try removing a random block and seeing if we can rebuild it to be "better" 
            # (e.g., covering more redundant subsets).
            # For now, just increment no_improve_iter to allow the loop to continue.
            # A simple restart:
            # blocks = list(best_blocks)
            # counts = compute_counts(blocks)
            # uncovered = 0
            # no_improve_iter = 0
            
            # Let's try a simple "kick": swap a random block with a random new block 
            # that covers its critical sets (if possible).
            # This is essentially what try_rebuild_block does.
            # If we are stuck, it means try_rebuild_block is failing.
            # Why? Maybe the critical sets cannot be covered by any k-block.
            # In that case, we are in a local minimum.
            # To escape, we might need to allow temporary invalidity (Simulated Annealing).
            # But for now, let's just keep searching.
            
            no_improve_iter += 1
        else:
            no_improve_iter += 1

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
