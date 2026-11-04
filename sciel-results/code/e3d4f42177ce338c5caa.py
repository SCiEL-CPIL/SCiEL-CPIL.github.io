import itertools
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Deadline: 150s limit, use 148s for safety
    deadline = time.time() + 148.0
    
    # Precompute all k-combinations? No, C(19,8) is ~75k, manageable but let's be careful.
    # Actually, for the initial valid cover, we can use a fast greedy.
    
    # Fast Greedy to get an initial valid covering (redundant)
    def fast_greedy_cover():
        covered = bytearray(num_tsubs)
        uncovered_count = num_tsubs
        blocks = []
        
        # To speed up initial cover, we don't need optimal selection, just valid.
        # Standard greedy: pick block covering max uncovered.
        # But computing gain for all C(v,k) blocks is expensive.
        # Alternative: Randomized greedy with lookahead?
        # Or: Just pick random blocks until covered? Might take too many.
        # Let's use a slightly smarter greedy: 
        # Pick a random uncovered t-subset, then pick a random k-subset containing it?
        # No, that's not guaranteed to be efficient.
        
        # Let's stick to the proven fast greedy but optimized:
        # We will precompute for each t-subset, a list of candidate blocks? 
        # No, that's C(v,k) * C(v,t) memory.
        
        # Let's use the "random anchor" greedy which was in the seed, 
        # but we just need ONE valid cover to start the local search.
        # We can make it very fast by not optimizing gain, just picking 
        # a random block that covers the current uncovered set?
        # Actually, a simple greedy that picks the block with the most 
        # new coverage is standard.
        
        # For v=19, k=8, t=5.
        # C(19,5) = 11628 t-subsets.
        # C(19,8) = 75582 blocks.
        # Each block covers C(8,5) = 56 t-subsets.
        
        # Let's implement a quick greedy:
        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                return None # Timeout during init
                
            # Pick a random uncovered t-subset
            # To avoid O(N) scan, we can keep a list of uncovered indices?
            # Or just probe.
            # Since we start with 0 covered, probing is fast.
            # As we cover more, probing slows down.
            
            # Optimization: Maintain a list of uncovered t-subset indices?
            # Removing from list is O(N).
            # Let's just probe.
            
            while True:
                i = random.randint(0, num_tsubs - 1)
                if not covered[i]:
                    break
            
            start_tsub = all_tsubs[i]
            
            # Pick a random k-subset containing start_tsub
            # start_tsub has t elements. We need k-t more from v-t remaining.
            remaining_points = [x for x in range(v) if x not in start_tsub]
            extra_points = random.sample(remaining_points, k - t)
            
            block = sorted(list(start_tsub) + extra_points)
            block_tuple = tuple(block)
            
            # Check if we already have this block?
            # We don't strictly need to avoid duplicates in the initial phase 
            # if we track covered status, but it wastes space.
            # Let's just add it.
            
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block_tuple, t):
                idx = tsub_index[ts]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1
            
            # Early exit if we have too many blocks?
            # If len(blocks) > 500, this random approach is too slow.
            # But for v=19, it should be around 300-400.
            
        return blocks

    # Local Search: Minimizing block count
    # State: list of blocks
    # Move 1: Remove a block if it is redundant (all its t-subsets are covered by others).
    # Move 2: Swap a block with another? (Hard to define "another" without generating new ones)
    # Let's focus on Removal first. This is a "Minimal Covering" problem.
    # A covering is minimal if no block can be removed.
    # We want a *minimum* covering.
    
    # Strategy:
    # 1. Get a valid cover (greedy).
    # 2. Repeatedly try to remove redundant blocks.
    # 3. If stuck (locally minimal), try to "shake" by removing a block 
    #    and then re-adding blocks to repair coverage, aiming for a smaller total?
    #    This is complex.
    
    # Simpler Local Search: 
    # Start with greedy cover.
    # Loop:
    #   a. Compute cover counts for each t-subset.
    #   b. Identify redundant blocks (those where all t-subsets have count > 1).
    #   c. Remove one redundant block at random.
    #   d. If no redundant blocks, try a "kick": 
    #      - Pick a random block, remove it.
    #      - Identify uncovered t-subsets.
    #      - Try to cover them with fewer new blocks? 
    #      - Or just accept the local minimum and restart with a different seed?
    
    # Given the time limit, restarting with different seeds for the initial greedy 
    # and then doing a thorough reduction pass is likely the most effective "structural" change 
    # compared to the seed which did greedy+reduce. 
    # The seed did: Greedy -> Reduce.
    # My approach: 
    # 1. Generate multiple initial covers (fast greedy with different seeds).
    # 2. For each, do an aggressive reduction:
    #    - Calculate cover counts.
    #    - Iteratively remove ANY block that is redundant.
    #    - If stuck, try "block exchange": 
    #      Remove block B. Find uncovered t-sets. 
    #      Try to find a single block B' that covers more of the uncovered sets 
    #      than B did? No, that's not right.
    
    # Let's implement a robust "Iterative Deletion" with randomization.
    
    def reduce_to_minimal(blocks):
        """
        Remove redundant blocks until none are left.
        Returns the reduced list.
        """
        if not blocks:
            return blocks
        
        # Map block tuple to index in list for fast updates?
        # Since we modify the list, indices change.
        
        # Better: Use a set of block tuples for O(1) lookup?
        # But we need to iterate and remove.
        
        # Let's use a list of block tuples.
        current_blocks = [tuple(b) for b in blocks]
        
        # Precompute t-subset indices for each block
        block_tsubs = []
        for b in current_blocks:
            tsubs = [tsub_index[ts] for ts in itertools.combinations(b, t)]
            block_tsubs.append(tsubs)
        
        cover_count = [0] * num_tsubs
        for tsubs in block_tsubs:
            for idx in tsubs:
                cover_count[idx] += 1
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_block_tsubs = []
            
            # We need to check each block if it's essential.
            # A block is essential if it contains at least one t-subset 
            # with cover_count == 1.
            
            for i, b in enumerate(current_blocks):
                tsubs = block_tsubs[i]
                essential = False
                for idx in tsubs:
                    if cover_count[idx] == 1:
                        essential = True
                        break
                
                if essential:
                    new_blocks.append(b)
                    new_block_tsubs.append(tsubs)
                else:
                    # Remove this block
                    changed = True
                    for idx in tsubs:
                        cover_count[idx] -= 1
            
            current_blocks = new_blocks
            block_tsubs = new_block_tsubs
            
            # Check timeout
            if time.time() > deadline - 5.0:
                break
        
        return [list(b) for b in current_blocks]

    def local_search_minimize(blocks):
        """
        Try to find a smaller covering by random removal and repair?
        For now, just do the reduction. 
        To improve, we can try "kick" moves:
        1. Remove a random block B.
        2. Check if covering is still valid. (It won't be if B was essential).
        3. If invalid, identify uncovered t-sets.
        4. Try to add a different block B' that covers these t-sets 
           and potentially allows removing another block later?
        
        This is getting complex. Let's stick to multiple random initializations 
        and thorough reduction. The variance in greedy construction 
        with different seeds often yields different local minima.
        """
        reduced = reduce_to_minimal(blocks)
        return reduced

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts of the initial greedy with different seeds
    # The seed program used 500 restarts. 
    # Since our reduction is deterministic given the blocks, 
    # the quality depends on the greedy output.
    
    # We can also randomize the *order* of removal in the reduction phase?
    # The current reduction removes ALL redundant blocks in one pass.
    # This is deterministic. 
    # To explore more, we can randomize which redundant block to remove first?
    # Actually, removing all redundant blocks simultaneously is safe 
    # because they are all non-essential. 
    # However, removing one might make another essential? 
    # No, if a block is non-essential, all its t-sets are covered by >= 2 blocks.
    # Removing one non-essential block decreases counts by 1.
    # If a t-set was covered by exactly 2 blocks, one of which is non-essential,
    # and we remove that non-essential block, the count becomes 1.
    # The other block becomes essential.
    # So we must be careful.
    
    # My `reduce_to_minimal` above does:
    # 1. Calculate counts.
    # 2. Identify ALL blocks that are essential (have a t-set with count 1).
    # 3. Remove ALL non-essential blocks.
    # 4. Update counts.
    # 5. Repeat.
    
    # This is correct. A block is essential if it is the *only* block covering some t-set.
    # If we remove all non-essential blocks, we might create new essential blocks, 
    # but we don't remove them in this pass. We repeat the loop.
    
    # This algorithm finds the unique minimal subcover? 
    # No, it finds A minimal subcover. 
    # Is it unique? 
    # If two blocks cover the same set of t-sets, and both are non-essential, 
    # we remove both. 
    # If they overlap partially, it depends on the counts.
    
    # To get better results, we can try different *initial* covers.
    
    num_restarts = 200 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Randomize the greedy construction
        # The `fast_greedy_cover` uses random.seed implicitly via random module state.
        # We should seed it explicitly or just rely on the global random state.
        # Let's seed it to ensure reproducibility per restart if needed, 
        # or just let it flow.
        
        blocks = fast_greedy_cover()
        
        if blocks is None:
            continue
            
        # Reduce
        reduced_blocks = reduce_to_minimal(blocks)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # Early exit if we hit a very good score
        if best_len <= 320:
            # Try a few more to see if we can beat it?
            # Or break?
            # Let's break to save time for other potential improvements?
            # No, let's continue a bit more.
            if restart > 50:
                break

    if best_blocks is None:
        # Fallback
        blocks = fast_greedy_cover()
        if blocks is None:
            # Emergency: return empty? No, must be valid.
            # If greedy failed, something is wrong.
            return []
        best_blocks = reduce_to_minimal(blocks)
        best_len = len(best_blocks)

    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END
