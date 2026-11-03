"""Simulated Annealing for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    # Precompute all t-subsets as tuples for faster hashing? 
    # frozenset is slower than tuple for comparison? 
    # Actually, for v=24, t=3, there are 2024 subsets.
    # We can represent a t-subset as an integer bitmask or a sorted tuple.
    # Sorted tuple is fine.
    
    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in all_points if x not in block]
            rng.shuffle(cands)
            
            # Optimize gain calculation
            # For each candidate x, count how many uncovered t-subsets are formed by x + (t-1) from block
            # Since t=3, we look at pairs from block.
            # Precompute pairs of current block?
            if t == 3:
                pairs = list(itertools.combinations(block, 2))
                for x in cands:
                    gain = 0
                    for p in pairs:
                        # Create the 3-set
                        # Sort to match canonical form
                        s = tuple(sorted(p + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            else:
                # General case
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            # Update uncovered
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))
                
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
                f = tuple(sorted(fs))
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Precompute total t-subsets for quick check
    # Not strictly needed if we track uncovered count
    
    uncovered_count = 0 # Should be 0
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = tuple(sorted(fs))
            old_c = counts.get(f, 0)
            new_c = old_c + sign
            if new_c == 0:
                if f in counts:
                    del counts[f]
                if old_c == 1:
                    delta_uncovered += 1
                elif old_c == 0:
                    delta_uncovered -= 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[f] = new_c
        return delta_uncovered

    def try_remove_block(index):
        """
        Try to remove block at index. Returns True if successful (improvement).
        """
        b = blocks[index]
        redundant = True
        for fs in itertools.combinations(b, t):
            f = tuple(sorted(fs))
            if counts.get(f, 0) <= 1:
                redundant = False
                break
        
        if redundant:
            delta_u = apply_block_change(b, -1)
            # delta_u should be 0 if redundant
            if delta_u == 0:
                blocks.pop(index)
                return True
            else:
                apply_block_change(b, 1)
        return False

    def try_swap_block(index, new_block):
        """
        Try to replace block at index with new_block.
        Returns True if successful (valid swap).
        """
        old_block = blocks[index]
        if old_block == new_block:
            return False
        
        # Check if new_block is already in blocks
        # This check is expensive O(N). We can skip it or do it carefully.
        # If we swap to a duplicate, we have a multiset of blocks? 
        # The problem says "blocks must be distinct".
        # So we must ensure new_block is not in blocks (except at index).
        
        # Quick check: if new_block is in blocks and not at index, reject.
        # This is O(N). With N~76, it's fast.
        for i, b in enumerate(blocks):
            if i != index and b == new_block:
                return False

        delta_remove = apply_block_change(old_block, -1)
        # If delta_remove > 0, we have uncovered sets.
        # We must add new_block and see if it covers them.
        
        delta_add = apply_block_change(new_block, 1)
        
        total_delta = delta_remove + delta_add
        
        if total_delta == 0:
            # Valid swap
            blocks[index] = new_block
            return True
        else:
            # Revert
            apply_block_change(new_block, -1)
            apply_block_change(old_block, 1)
            return False

    no_improve_iter = 0
    max_no_improve = 200
    temp = 1.0
    cooling = 0.999

    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        # Pick a few random indices
        for _ in range(5):
            if not blocks: break
            i = rng.randrange(len(blocks))
            if try_remove_block(i):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve_iter = 0
                break
        
        if no_improve_iter == 0:
            continue

        # 2. Try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Generate a new block similar to old_block
            # Keep some points, change others
            # Or just random?
            # Let's try a "local" swap: change 1-3 points.
            
            # Strategy: Keep t-1 points from old block? No, that forces structure.
            # Let's pick a random subset of points from old_block to keep.
            keep_size = rng.randint(2, k-1)
            keep_points = rng.sample(old_block, keep_size)
            
            new_block_set = set(keep_points)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Fill rest randomly
            while len(new_block_set) < k:
                new_block_set.add(cands.pop())
            
            new_block = tuple(sorted(new_block_set))
            
            if try_swap_block(i, new_block):
                # Accept swap
                # This is a neutral move (count same). 
                # We accept it to explore state space.
                # To avoid cycling, we might want to accept with prob exp(-0/T) = 1? 
                # Yes, neutral moves are always good for mixing.
                no_improve_iter += 1 # Count as no improvement in count, but reset "stuck" counter?
                # Actually, if we swap, we might enable a removal later.
                # Let's reset no_improve_iter to a smaller value or keep it?
                # If we keep incrementing, we'll trigger perturbation soon.
                # Let's reset to 0 to allow more swaps.
                no_improve_iter = 0
                continue

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            # Destroy and Repair
            old_blocks = list(blocks)
            old_counts = dict(counts)
            
            # Remove a random block
            i = rng.randrange(len(blocks))
            b = blocks[i]
            apply_block_change(b, -1)
            blocks.pop(i)
            
            # Find uncovered sets
            unc_list = []
            for f, c in counts.items():
                if c == 0:
                    unc_list.append(f)
            
            # Greedily cover uncovered sets
            # This is a mini-greedy
            # We want to add as few blocks as possible.
            # If we add 1 block, count is same. If 0, improvement. If >1, worse.
            # So we only accept if we add <= 1 block? 
            # Or we can add more if it leads to a better structure? 
            # Given time, let's just try to cover with 1 block.
            
            if len(unc_list) > 0:
                # Find a block that covers all unc_list?
                # Or covers the most?
                # Let's try to find a block that covers all unc_list.
                # If unc_list is small, we can intersect constraints.
                # For each unc_set in unc_list, the new block must be a superset.
                # Intersection of supersets?
                
                # Simpler: Pick one unc_set, build a block around it, check if it covers all others.
                seed = rng.choice(unc_list)
                block = list(seed)
                rng.shuffle(block)
                
                # Greedily extend to cover all unc_list
                # We need to cover all sets in unc_list.
                # Current block covers some.
                # Let's just run a few greedy steps.
                
                # Actually, let's just try to add 1 random block and see if it fixes it.
                # If not, revert.
                
                # Better: Greedily construct a block to cover unc_list.
                # Start with seed.
                # While unc_list not empty:
                #   Pick x that covers most unc_list.
                
                # This might be slow if unc_list is large.
                # But usually removal of 1 block uncovers at most C(7,3)=35 sets.
                
                current_unc = set(unc_list)
                new_block = list(seed)
                rng.shuffle(new_block)
                
                while len(new_block) < k and current_unc:
                    best_x = None
                    best_gain = -1
                    cands = [x for x in range(v) if x not in new_block]
                    rng.shuffle(cands)
                    
                    for x in cands:
                        gain = 0
                        for rest in itertools.combinations(new_block, t-1):
                            fs = tuple(sorted(rest + (x,)))
                            if fs in current_unc:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x is None or best_gain == 0:
                        break
                    new_block.append(best_x)
                    # Update current_unc
                    for rest in itertools.combinations(new_block, t):
                        current_unc.discard(tuple(sorted(rest)))
                
                new_block.sort()
                new_block = tuple(new_block)
                
                # Check if it covers all original unc_list
                # Re-add and check
                apply_block_change(new_block, 1)
                
                # Check if any uncovered remain
                still_unc = 0
                for f, c in counts.items():
                    if c == 0:
                        still_unc += 1
                        if still_unc > 0: break # just need to know if >0
                
                if still_unc == 0:
                    # Success! We added 1 block to cover what was uncovered by removing 1.
                    # Count is same.
                    # But structure changed.
                    blocks.append(new_block)
                    no_improve_iter = 0
                else:
                    # Failed, revert
                    apply_block_change(new_block, -1)
                    blocks = old_blocks
                    counts = old_counts
                    no_improve_iter += 1
            else:
                # No uncovered sets? That means the removed block was redundant.
                # But try_remove_block should have caught that.
                # Unless counts were stale?
                # If unc_list is empty, we just removed a block and it's valid.
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve_iter = 0

        else:
            no_improve_iter += 1

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
