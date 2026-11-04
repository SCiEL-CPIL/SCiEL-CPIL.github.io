import itertools
import random
import time

def _remove_redundant_blocks(v, k, t, blocks, counts=None):
    """
    Remove blocks that do not uniquely cover any t-subset.
    """
    if not blocks:
        return [], {}
    
    if counts is None:
        counts = {}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            
    redundant_indices = set()
    for i, b in enumerate(blocks):
        is_redundant = True
        for tsub in itertools.combinations(b, t):
            if counts.get(tsub, 0) <= 1:
                is_redundant = False
                break
        if is_redundant:
            redundant_indices.add(i)
            
    if not redundant_indices:
        return blocks, counts
        
    new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
    
    for i in redundant_indices:
        b = blocks[i]
        for tsub in itertools.combinations(b, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
                
    while True:
        if not new_blocks:
            return [], counts
            
        redundant_indices = set()
        for i, b in enumerate(new_blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                if counts.get(tsub, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.add(i)
                
        if not redundant_indices:
            break
            
        for i in redundant_indices:
            b = new_blocks[i]
            for tsub in itertools.combinations(b, t):
                counts[tsub] -= 1
                if counts[tsub] == 0:
                    del counts[tsub]
                    
        new_blocks = [b for i, b in enumerate(new_blocks) if i not in redundant_indices]
        
    return new_blocks, counts

def _build_greedy(v, k, t, all_tsubs, rng, deadline):
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset to anchor the new block
        # This ensures we cover something new
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = None
            t_minus_1 = t - 1
            block_t1 = list(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in block_t1:
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
                
            block.append(best_point)
            block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tuple(tsub))
            
    return blocks

def _local_search(v, k, t, blocks, rng, deadline):
    # Initialize counts
    counts = {}
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            counts[tsub] = counts.get(tsub, 0) + 1
            
    # Precompute all possible blocks? No, too many (77k).
    # But we can precompute for each point, which blocks contain it? No.
    
    # We will use a "Remove and Repair" strategy.
    # 1. Remove a random block B.
    # 2. Find uncovered subsets U.
    # 3. If U is empty, accept.
    # 4. If U is not empty, try to find a single block B' that covers all of U.
    #    If found, add B'. If len(blocks) decreased (removed 1, added 1 -> same size, but maybe better structure) 
    #    or if we can remove more after this, we accept.
    #    Actually, if we remove 1 and add 1, size is same. We only want to reduce size.
    #    So this move is only useful if it enables further reductions.
    #    Alternatively, try to cover U with fewer blocks than we removed? Impossible if U non-empty.
    #    So, "Remove and Repair" with 1 block is a neutral move.
    #    "Remove and Repair" with 0 blocks is a gain.
    
    # Better move: 
    # Try to replace a block B with a block B' such that B' covers all unique subsets of B 
    # AND B' is "better" (e.g., covers more already covered subsets? No, that doesn't help count).
    # The only way to reduce count is to find a block B that is redundant (unique subsets = 0) 
    # OR to find a set of blocks {B1, B2} that can be replaced by {B3} where B3 covers 
    # all unique subsets of B1 and B2.
    
    # Let's implement "Pair Replacement":
    # Pick two blocks B1, B2.
    # Remove them.
    # Let U be the union of unique subsets of B1 and B2.
    # If U is empty, accept (removed 2).
    # If U is not empty, try to find a single block B3 that covers U.
    # If found, add B3. Net change: -1 block. Accept.
    
    # This is a powerful move.
    
    while time.time() < deadline:
        if len(blocks) < 2:
            break
            
        # Try single block removal
        idx = rng.randrange(len(blocks))
        B = blocks[idx]
        
        # Temporarily remove B
        for tsub in itertools.combinations(B, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
                
        # Check if any t-subset is uncovered
        uncovered_U = []
        for tsub in itertools.combinations(B, t):
            if tsub not in counts:
                uncovered_U.append(tsub)
                
        if not uncovered_U:
            # B was redundant
            blocks.pop(idx)
            continue # Loop again to find more redundants
        
        # Restore B for now, we'll try pair replacement
        for tsub in itertools.combinations(B, t):
            counts[tsub] = counts.get(tsub, 0) + 1
            
        # Try Pair Replacement
        # Pick another block B2
        idx2 = rng.randrange(len(blocks))
        if idx2 == idx:
            continue
        B2 = blocks[idx2]
        
        # Remove B and B2
        for tsub in itertools.combinations(B, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
        for tsub in itertools.combinations(B2, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
                
        # Find U: t-subsets that are now uncovered (were covered by B or B2, now count 0)
        # Note: A t-subset might have been covered by B and B2. If so, count was 2, now 1. Not uncovered.
        # We only care about t-subsets with count 0.
        U = []
        for tsub in itertools.combinations(B, t):
            if tsub not in counts:
                U.append(tsub)
        for tsub in itertools.combinations(B2, t):
            if tsub not in counts:
                U.append(tsub)
                
        if not U:
            # Both were redundant relative to each other? No, just that their unique sets were covered by others.
            # Remove both
            # Remove larger index first to keep indices valid
            if idx > idx2:
                idx, idx2 = idx2, idx
            blocks.pop(idx2)
            blocks.pop(idx)
            continue
            
        # We need to cover U with a single block B3
        # U is a set of t-subsets.
        # Let S_U be the union of all elements in U.
        S_U = set()
        for u in U:
            S_U.update(u)
            
        if len(S_U) > k:
            # Cannot cover with one block
            # Restore
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            for tsub in itertools.combinations(B2, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            continue
            
        # We need to find a block B3 such that U is subset of combinations(B3, t).
        # This is equivalent to: For every u in U, u is subset of B3.
        # This implies S_U is subset of B3.
        # So B3 must contain S_U.
        # We can fill the remaining k - len(S_U) spots with random points.
        
        found_better = False
        # Try a few random completions of S_U to form B3
        available_points = [x for x in range(v) if x not in S_U]
        needed = k - len(S_U)
        
        # To increase success rate, we can try to pick points that are "good"
        # But for now, random is fine.
        for _ in range(20):
            if needed == 0:
                B3 = sorted(list(S_U))
            else:
                extra = rng.sample(available_points, needed)
                B3 = sorted(list(S_U) + extra)
                
            # Check if B3 covers all of U
            # Since B3 contains S_U, and every u in U is a subset of S_U, B3 covers all u in U.
            # So this is always valid for coverage of U.
            
            # Now, check if adding B3 allows us to keep the net gain of -1 block.
            # We removed 2, added 1. Net -1.
            # We must ensure the covering is still valid (it is, by construction of U and B3).
            # We should also check if this introduces redundancy that can be exploited later.
            
            # Add B3 to counts
            b3_tsubs = list(itertools.combinations(B3, t))
            for tsub in b3_tsubs:
                counts[tsub] = counts.get(tsub, 0) + 1
                
            # Update blocks list
            # Remove B and B2, add B3
            new_blocks = []
            remove_set = {idx, idx2}
            for i, b in enumerate(blocks):
                if i in remove_set:
                    continue
                new_blocks.append(b)
            new_blocks.append(B3)
            
            blocks = new_blocks
            found_better = True
            break # Accept this move
            
        if not found_better:
            # Restore B and B2
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            for tsub in itertools.combinations(B2, t):
                counts[tsub] = counts.get(tsub, 0) + 1

    # Final cleanup
    blocks, counts = _remove_redundant_blocks(v, k, t, blocks, counts)
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_tsubs = [tuple(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_len = float('inf')
    
    restart_num = 0
    while time.time() < deadline - 10.0:
        restart_num += 1
        rng = random.Random(1000 + restart_num)
        
        # 1. Greedy Construction
        blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline - 15.0)
        
        if not blocks:
            continue
            
        # 2. Local Search
        blocks = _local_search(v, k, t, blocks, rng, deadline - 5.0)
        
        # 3. Final Cleanup
        blocks, _ = _remove_redundant_blocks(v, k, t, blocks)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        best_blocks, _ = _remove_redundant_blocks(v, k, t, best_blocks)
        
    return best_blocks
