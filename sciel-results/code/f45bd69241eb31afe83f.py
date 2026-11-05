import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    T = len(all_triples)
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    B = len(all_blocks)
    
    # Map block to bitmask of triples it covers
    # Using Python integers for bitmasks
    block_masks = [0] * B
    # Also map triple to list of blocks containing it
    triple_to_blocks = [[] for _ in range(T)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_index[triple]
            mask |= (1 << idx)
            triple_to_blocks[idx].append(i)
        block_masks[i] = mask

    # Helper to check if a block is redundant given a set of other blocks
    # Returns True if block is redundant
    def is_redundant(block_idx, current_blocks_set):
        # Get mask of current block
        m = block_masks[block_idx]
        # Check if all triples in m are covered by other blocks
        # We can sum masks of other blocks? No, too slow.
        # Instead, for each triple in m, check if any other block in current_blocks_set covers it.
        # Optimization: If we maintain a coverage count array, this is O(1) per triple.
        # Let's maintain a global coverage count array `cover_count` of size T.
        # This is better for the main loop.
        pass # Will implement with cover_count in main loop

    # State:
    # current_blocks: set of block indices
    # cover_count: array of size T, count of how many blocks in current_blocks cover each triple
    
    def greedy_init():
        uncovered = (1 << T) - 1
        blocks_set = set()
        cover_count = [0] * T
        while uncovered:
            # Pick a random uncovered triple
            # Find an index where uncovered has a 1
            # This is inefficient if we do it naively.
            # Better: Pick a random triple from all_triples, if uncovered, use it.
            # Or, maintain a list of uncovered triples?
            # Let's just pick a random triple index.
            while True:
                idx = random.randint(0, T - 1)
                if (uncovered >> idx) & 1:
                    break
            
            # Pick a random block covering this triple
            candidates = triple_to_blocks[idx]
            if not candidates:
                # Should not happen for v=24, k=7, t=3
                break
            b_idx = random.choice(candidates)
            
            if b_idx in blocks_set:
                continue
                
            blocks_set.add(b_idx)
            # Update cover_count and uncovered
            m = block_masks[b_idx]
            # Iterate through set bits in m
            mm = m
            while mm:
                lsb = mm & -mm
                idx2 = lsb.bit_length() - 1
                cover_count[idx2] += 1
                if cover_count[idx2] == 1:
                    uncovered &= ~(1 << idx2)
                mm ^= lsb
                
        return blocks_set, cover_count, uncovered

    # Initial solution
    blocks_set, cover_count, uncovered = greedy_init()
    best_blocks = set(blocks_set)
    best_size = len(blocks_set)
    
    # Local Search
    # We will try to remove a block, see if it's redundant.
    # If not, try to replace it with a better one that might allow further reduction.
    
    def try_remove_and_fix(blocks_set, cover_count, uncovered):
        # Returns (new_blocks_set, new_cover_count, new_uncovered, improved)
        if not blocks_set:
            return blocks_set, cover_count, uncovered, False
            
        # Pick a random block to remove
        b_remove = random.choice(list(blocks_set))
        
        # Remove it
        blocks_set.remove(b_remove)
        m = block_masks[b_remove]
        newly_uncovered = 0
        mm = m
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                newly_uncovered |= (1 << idx)
            mm ^= lsb
            
        uncovered |= newly_uncovered
        
        if not newly_uncovered:
            # Block was redundant, removed successfully
            return blocks_set, cover_count, uncovered, True
        
        # Try to fix by adding a block
        # Find a block that covers the most newly_uncovered triples
        # If newly_uncovered is small, use triple_to_blocks intersection
        # Else, sample
        
        # Get list of set bits in newly_uncovered
        u_list = []
        mm = newly_uncovered
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            u_list.append(idx)
            mm ^= lsb
            
        best_gain = -1
        best_block = -1
        
        if len(u_list) <= 3:
            # Intersect block lists
            candidates = set(triple_to_blocks[u_list[0]])
            for idx in u_list[1:]:
                candidates &= set(triple_to_blocks[idx])
                if not candidates:
                    break
            if candidates:
                for b in candidates:
                    if b in blocks_set:
                        continue
                    # Check gain (should be len(u_list) if it covers all)
                    # Actually, we want to cover *all* newly_uncovered to restore validity
                    # But we might accept a block that covers *some* and rely on other blocks?
                    # No, we removed b_remove, so only newly_uncovered are the problem.
                    # If we add b_new, it must cover newly_uncovered?
                    # Not necessarily. It covers some, and maybe other existing blocks cover the rest?
                    # No, cover_count was 0 for those. So b_new MUST cover them.
                    # So we need b_new such that newly_uncovered is a subset of block_masks[b_new].
                    gain = 0
                    mm2 = newly_uncovered
                    while mm2:
                        lsb = mm2 & -mm2
                        idx2 = lsb.bit_length() - 1
                        if (block_masks[b] >> idx2) & 1:
                            gain += 1
                        mm2 ^= lsb
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
        else:
            # Sample some blocks
            # We can sample from the union of triple_to_blocks for a few random triples in u_list
            # Or just random sample from all_blocks
            # Let's sample from candidates of the first triple
            candidates = triple_to_blocks[u_list[0]]
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)
            for b in candidates:
                if b in blocks_set:
                    continue
                gain = 0
                mm2 = newly_uncovered
                while mm2:
                    lsb = mm2 & -mm2
                    idx2 = lsb.bit_length() - 1
                    if (block_masks[b] >> idx2) & 1:
                        gain += 1
                    mm2 ^= lsb
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            # If we didn't find a good one, try a few more random blocks
            for _ in range(10):
                b = random.randint(0, B - 1)
                if b in blocks_set:
                    continue
                gain = 0
                mm2 = newly_uncovered
                while mm2:
                    lsb = mm2 & -mm2
                    idx2 = lsb.bit_length() - 1
                    if (block_masks[b] >> idx2) & 1:
                        gain += 1
                    mm2 ^= lsb
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
        
        if best_block == -1 or best_gain < len(u_list):
            # Couldn't fully cover, revert
            blocks_set.add(b_remove)
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered &= ~(1 << idx)
                mm ^= lsb
            return blocks_set, cover_count, uncovered, False
        
        # Add best_block
        blocks_set.add(best_block)
        m_new = block_masks[best_block]
        mm = m_new
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered &= ~(1 << idx)
            mm ^= lsb
        
        # Now check if any block is redundant
        # Iterate through blocks in blocks_set
        # If a block's triples are all covered by others, remove it
        # This is O(N_blocks * T) in worst case, but T=2024, N_blocks~76.
        # 76 * 2024 * (cost of check)
        # Cost of check: for each triple in block, check if cover_count > 1?
        # No, cover_count includes the block itself.
        # So if for all triples in block b, cover_count[triple] > 1, then b is redundant.
        
        removed_any = False
        for b in list(blocks_set):
            m_b = block_masks[b]
            redundant = True
            mm = m_b
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                mm ^= lsb
            if redundant:
                # Remove b
                blocks_set.remove(b)
                mm = m_b
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    if cover_count[idx] == 0:
                        uncovered |= (1 << idx)
                    mm ^= lsb
                removed_any = True
                # If we removed one, we might have made others redundant?
                # Yes, so continue loop.
                
        if removed_any:
            return blocks_set, cover_count, uncovered, True
        else:
            # We swapped, but didn't reduce count.
            # Accept this move?
            # If we accept neutral moves, we explore.
            # But we must ensure we don't cycle.
            # Let's accept neutral moves with some probability or just accept them.
            # Since we want to minimize, and this move didn't increase count, it's safe.
            # But it might lead to a dead end.
            # Let's accept it.
            return blocks_set, cover_count, uncovered, True

    # Main Loop
    while time.time() < deadline:
        # Make a copy of current state?
        # To allow backtracking if a move is bad?
        # Since we only accept if count doesn't increase, we can just do it in place.
        # But if we swap and don't reduce, we might want to revert if we get stuck?
        # Let's just do in-place.
        
        # Check if current is better than best
        if len(blocks_set) < best_size:
            best_size = len(blocks_set)
            best_blocks = set(blocks_set)
            
        # Try a move
        new_blocks, new_cover, new_uncovered, improved = try_remove_and_fix(blocks_set, cover_count, uncovered)
        blocks_set, cover_count, uncovered = new_blocks, new_cover, new_uncovered
        
        # If we improved, continue
        # If not, we might want to randomize more
        # The randomization is inside try_remove_and_fix
        
    # Return the best found
    # Convert set of indices to list of blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
