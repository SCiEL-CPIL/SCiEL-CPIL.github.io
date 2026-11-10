"""Covering design C(v, k, t) – Bitmask Greedy with Local Search Minimization."""
import itertools
import time
import random
import numpy as np

_DEADLINE_S = 145.0  # Safety margin from 150s evaluator limit
_START_TIME = time.time()

def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))

def _get_all_tsubs(v, t):
    """Generate all t-subsets as bitmasks."""
    subs = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        subs.append(mask)
    return subs

def _construct_greedy(v, k, t, rng, deadline_check=True):
    """
    Fast Greedy Set Cover using bitmasks.
    Returns a list of blocks (lists of points).
    """
    all_tsubs = _get_all_tsubs(v, t)
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset mask to its index
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Uncovered mask: bit i is 1 if t-sub i is uncovered
    uncovered = np.ones(n_tsubs, dtype=bool)
    
    blocks = []
    block_set = set()
    
    # Precompute for each possible k-block, which t-subsets it covers?
    # This is too large: C(19,7) = 50388. For each, C(7,4)=35 subsets.
    # Total operations ~ 50388 * 35 ~ 1.7M. This is feasible to precompute!
    
    # Precompute coverage masks for all possible blocks
    # block_mask_to_subs[block_mask] = list of t-sub indices covered
    # Actually, we can precompute a matrix or just a dict.
    # Let's precompute the list of t-sub indices for each k-block.
    
    block_to_subs = {}
    for block in itertools.combinations(range(v), k):
        mask = 0
        for p in block:
            mask |= (1 << p)
        # Find which t-subsets are in this block
        subs_covered = []
        for sub in itertools.combinations(block, t):
            sub_mask = 0
            for p in sub:
                sub_mask |= (1 << p)
            if sub_mask in sub_to_idx:
                subs_covered.append(sub_to_idx[sub_mask])
        block_to_subs[mask] = subs_covered
        
    # Now Greedy:
    # At each step, find the block that covers the most uncovered t-subsets.
    # We can iterate through all possible blocks and check their coverage against 'uncovered'.
    # To speed this up, we can maintain a count of uncovered t-subsets for each block?
    # No, 'uncovered' changes.
    # However, v=19 is small. C(19,7)=50388.
    # In each step, we check 50388 blocks. Each check involves iterating over its ~35 t-subsets.
    # 50388 * 35 = 1.7M operations per step.
    # If we need ~150 blocks, that's 150 * 1.7M = 255M operations.
    # In Python, this might be too slow (10-20s per restart?).
    # We need to be smarter.
    
    # Alternative: Randomized Greedy with limited candidates, but using bitmasks for speed.
    # Let's stick to the precomputed dict but use a smarter selection.
    
    # Actually, let's just use the precomputed `block_to_subs` and do a greedy pass.
    # To make it faster, we can sample candidates.
    
    # Let's try a different approach: 
    # 1. Generate a random valid covering using a fast heuristic.
    # 2. Use Local Search to minimize.
    
    # Fast Heuristic:
    # Pick random blocks, keep those that cover new t-subsets.
    # To ensure validity, we must cover all.
    
    # Let's do a "Greedy with Randomization":
    # At each step, pick a random subset of candidates (e.g., 1000) and pick the best among them.
    
    all_blocks_keys = list(block_to_subs.keys())
    n_blocks_possible = len(all_blocks_keys)
    
    while np.any(uncovered) and _time_left() > 1.0:
        # Sample candidates
        n_samples = min(2000, n_blocks_possible)
        candidates = rng.sample(all_blocks_keys, n_samples)
        
        best_block_key = None
        best_gain = -1
        
        for bkey in candidates:
            subs = block_to_subs[bkey]
            # Count how many of these subs are currently uncovered
            # uncovered is a numpy bool array
            gain = np.sum(uncovered[subs])
            if gain > best_gain:
                best_gain = gain
                best_block_key = bkey
        
        if best_block_key is None or best_gain <= 0:
            # If no candidate covers new stuff, pick a random one to avoid infinite loop
            # or break if truly stuck
            best_block_key = rng.choice(all_blocks_keys)
            subs = block_to_subs[best_block_key]
            if np.sum(uncovered[subs]) == 0:
                # Try a few more random ones
                found = False
                for _ in range(10):
                    bkey = rng.choice(all_blocks_keys)
                    if np.sum(uncovered[block_to_subs[bkey]]) > 0:
                        best_block_key = bkey
                        found = True
                        break
                if not found:
                    break # Stuck
            
        # Add the block
        blocks.append(best_block_key)
        block_set.add(best_block_key)
        
        # Update uncovered
        subs = block_to_subs[best_block_key]
        uncovered[subs] = False
        
        # If time is running out, stop
        if _time_left() < 0.5:
            break
            
    return blocks

def _local_search_minimize(blocks, v, k, t, rng, max_iter=1000):
    """
    Local search to minimize the number of blocks.
    Moves:
    1. Remove a block if it is redundant (all its t-subsets are covered by other blocks).
    2. Replace two blocks with one if possible (harder, skip for now).
    """
    if not blocks:
        return blocks
    
    # Precompute t-subsets for each block in the current covering
    # block_idx -> list of t-sub masks
    block_subs = []
    for b in blocks:
        subs = []
        for sub in itertools.combinations(b, t):
            mask = 0
            for p in sub:
                mask |= (1 << p)
            subs.append(mask)
        block_subs.append(subs)
    
    # Coverage count: for each t-sub mask, how many blocks cover it?
    # Since v=19, t=4, total t-subs = 3876.
    # We can use a dict or array.
    # Let's use a dict: sub_mask -> count
    sub_count = {}
    for subs in block_subs:
        for s in subs:
            sub_count[s] = sub_count.get(s, 0) + 1
    
    # All possible t-subsets
    all_tsubs = _get_all_tsubs(v, t)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter and _time_left() > 1.0:
        improved = False
        iterations += 1
        
        # Try to remove redundant blocks
        # A block is redundant if all its t-subsets have count > 1
        # We can check this efficiently.
        
        # Shuffle indices to randomize removal order
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if _time_left() < 1.0:
                break
            
            # Check if block idx is redundant
            is_redundant = True
            for s in block_subs[idx]:
                if sub_count[s] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block idx
                # Update sub_count
                for s in block_subs[idx]:
                    sub_count[s] -= 1
                
                # Remove from lists
                blocks.pop(idx)
                block_subs.pop(idx)
                improved = True
                # After removal, indices are shifted, so we break and restart
                # Or we can just break the inner loop and let the outer loop handle it
                break
        
    return blocks

def construct(v, k, t):
    random.seed(12345)
    best_blocks = None
    best_len = float('inf')
    
    # We will run multiple restarts of the greedy phase, then local search on the best ones.
    # Given the time limit, we can afford a few restarts.
    
    max_restarts = 5
    
    for r in range(max_restarts):
        if _time_left() < 10.0:
            break
            
        rng = random.Random(12345 + r * 7919)
        
        # Phase 1: Greedy Construction
        blocks = _construct_greedy(v, k, t, rng)
        
        # Check validity
        # We need to ensure all t-subsets are covered.
        # The greedy loop stops when uncovered is empty or time runs out.
        # If time runs out, it might be invalid.
        # Let's verify.
        
        # For verification, we can use the sub_count logic from local search.
        # But it's faster to just check if the greedy loop finished.
        # The greedy loop returns blocks. We don't know if it's valid.
        # Let's do a quick check.
        
        covered = set()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                covered.add(tuple(sorted(sub)))
        
        all_tsubs_set = set(tuple(sorted(sub)) for sub in itertools.combinations(range(v), t))
        
        if covered != all_tsubs_set:
            # Invalid, try to repair or skip
            # Simple repair: add random blocks until valid
            if _time_left() > 5.0:
                while covered != all_tsubs_set:
                    missing = all_tsubs_set - covered
                    if not missing:
                        break
                    target = next(iter(missing))
                    # Extend target to a block
                    block = list(target)
                    block_set_local = set(block)
                    remaining = [x for x in range(v) if x not in block_set_local]
                    rng.shuffle(remaining)
                    for x in remaining:
                        if len(block) >= k:
                            break
                        block.append(x)
                        block_set_local.add(x)
                    block = tuple(sorted(block))
                    if block not in set(tuple(b) for b in blocks):
                        blocks.append(list(block))
                        for sub in itertools.combinations(block, t):
                            covered.add(tuple(sorted(sub)))
        
        # Phase 2: Local Search Minimization
        if _time_left() > 5.0:
            blocks = _local_search_minimize(blocks, v, k, t, rng)
            
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
        if best_len <= 151:
            break
            
    # Fallback
    if best_blocks is None:
        # Simple greedy fallback
        rng = random.Random(99999)
        uncovered = set(tuple(sorted(sub)) for sub in itertools.combinations(range(v), t))
        blocks = []
        while uncovered and _time_left() > 0.1:
            target = min(uncovered)
            block = list(target)
            block_set_local = set(block)
            remaining = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block) >= k:
                    break
                block.append(x)
                block_set_local.add(x)
            block = tuple(sorted(block))
            if block not in set(tuple(b) for b in blocks):
                blocks.append(list(block))
                for sub in itertools.combinations(block, t):
                    uncovered.discard(tuple(sorted(sub)))
        best_blocks = blocks
        
    return best_blocks
