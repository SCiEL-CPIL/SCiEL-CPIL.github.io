"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets as frozensets
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {fs: i for i, fs in enumerate(all_tsubs)}
    
    # Precompute for each point, which t-subsets it participates in
    # point_tsubs[x] = list of indices of t-subsets containing x
    point_tsubs = [[] for _ in range(v)]
    for i, fs in enumerate(all_tsubs):
        for x in fs:
            point_tsubs[x].append(i)
    
    # Greedy construction
    def greedy_cover():
        uncovered = set(range(n_tsubs))  # indices of uncovered t-subsets
        blocks = []
        
        while uncovered:
            # Pick random uncovered t-subset as seed
            seed_idx = rng.choice(list(uncovered))
            seed = all_tsubs[seed_idx]
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x covers a new t-subset
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = tsub_to_idx.get(fs)
                        if idx is not None and idx in uncovered:
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
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered.discard(idx)
        
        return blocks

    # Build initial covering
    blocks = greedy_cover()
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # counts[i] = number of blocks covering t-subset i
    counts = [0] * n_tsubs
    for b in blocks:
        for c in itertools.combinations(b, t):
            idx = tsub_to_idx[frozenset(c)]
            counts[idx] += 1
    
    # single_count[i] = 1 if counts[i] == 1, else 0
    single = [1 if c == 1 else 0 for c in counts]
    n_single = sum(single)
    n_uncovered = 0  # should be 0 for valid covering
    
    def compute_block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        return [tsub_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]
    
    # Precompute t-subset indices for each block
    block_tsubs = [compute_block_tsub_indices(b) for b in blocks]
    
    # Local search
    iteration = 0
    stuck_count = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        progress = False
        
        # Strategy 1: Try to remove redundant blocks
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            # Check if block i is redundant (all its t-subsets have count > 1)
            redundant = True
            for idx in block_tsubs[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_tsubs[i]:
                    old_c = counts[idx]
                    counts[idx] = old_c - 1
                    if old_c == 1:
                        n_single -= 1
                        single[idx] = 0
                    elif old_c == 2:
                        n_single += 1
                        single[idx] = 1
                    if counts[idx] == 0:
                        n_uncovered += 1
                
                blocks.pop(i)
                block_tsubs.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                progress = True
                break
        
        if progress:
            stuck_count = 0
            continue
        
        # Strategy 2: Swap a block to enable further removals
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_tsubs = block_tsubs[i]
            
            # Find critical t-subsets (count == 1) in old block
            critical = [idx for idx in old_tsubs if counts[idx] == 1]
            
            if len(critical) > 0:
                # Collect points needed to cover all critical t-subsets
                points_needed = set()
                for idx in critical:
                    points_needed.update(all_tsubs[idx])
                
                if len(points_needed) <= k:
                    # Try to form a new block containing points_needed
                    new_block_set = set(points_needed)
                    cands = [x for x in range(v) if x not in new_block_set]
                    rng.shuffle(cands)
                    
                    # Greedily fill to maximize coverage of single-covered or uncovered subsets
                    for x in cands:
                        if len(new_block_set) >= k:
                            break
                        gain = 0
                        for rest in itertools.combinations(new_block_set, t-1):
                            fs = frozenset(list(rest) + [x])
                            idx = tsub_to_idx.get(fs)
                            if idx is not None:
                                if counts[idx] == 0:
                                    gain += 100
                                elif counts[idx] == 1:
                                    gain += 1
                        if gain > 0:
                            new_block_set.add(x)
                        elif len(new_block_set) < k:
                            new_block_set.add(x)
                    
                    if len(new_block_set) == k:
                        new_block = tuple(sorted(new_block_set))
                        if new_block != old_block:
                            new_tsubs = [tsub_to_idx[frozenset(c)] for c in itertools.combinations(new_block, t)]
                            
                            # Check if swap is valid: after removing old and adding new, no t-subset has count 0
                            # A t-subset becomes 0 only if it was in old_block (count 1) and not in new_block
                            valid = True
                            old_set = set(old_tsubs)
                            new_set = set(new_tsubs)
                            for idx in old_tsubs:
                                if counts[idx] == 1 and idx not in new_set:
                                    valid = False
                                    break
                            
                            if valid:
                                # Apply swap
                                for idx in old_tsubs:
                                    old_c = counts[idx]
                                    counts[idx] = old_c - 1
                                    if old_c == 1:
                                        n_single -= 1
                                        single[idx] = 0
                                    elif old_c == 2:
                                        n_single += 1
                                        single[idx] = 1
                                    if counts[idx] == 0:
                                        n_uncovered += 1
                                
                                for idx in new_tsubs:
                                    old_c = counts[idx]
                                    counts[idx] = old_c + 1
                                    if old_c == 0:
                                        n_uncovered -= 1
                                    elif old_c == 1:
                                        n_single += 1
                                        single[idx] = 1
                                    elif old_c == 2:
                                        n_single -= 1
                                        single[idx] = 0
                                
                                blocks[i] = new_block
                                block_tsubs[i] = new_tsubs
                                stuck_count = 0
                                continue
        
        # Strategy 3: If stuck, try random perturbation
        stuck_count += 1
        if stuck_count >= 50:
            # Replace a random block with a random block that covers its critical subsets
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_tsubs = block_tsubs[i]
                
                critical = [idx for idx in old_tsubs if counts[idx] == 1]
                points_needed = set()
                for idx in critical:
                    points_needed.update(all_tsubs[idx])
                
                if len(points_needed) <= k:
                    new_block_set = set(points_needed)
                    cands = [x for x in range(v) if x not in new_block_set]
                    rng.shuffle(cands)
                    for x in cands:
                        if len(new_block_set) >= k:
                            break
                        new_block_set.add(x)
                    
                    if len(new_block_set) == k:
                        new_block = tuple(sorted(new_block_set))
                        new_tsubs = [tsub_to_idx[frozenset(c)] for c in itertools.combinations(new_block, t)]
                        
                        # Check validity
                        valid = True
                        new_set = set(new_tsubs)
                        for idx in old_tsubs:
                            if counts[idx] == 1 and idx not in new_set:
                                valid = False
                                break
                        
                        if valid:
                            for idx in old_tsubs:
                                old_c = counts[idx]
                                counts[idx] = old_c - 1
                                if old_c == 1:
                                    n_single -= 1
                                    single[idx] = 0
                                elif old_c == 2:
                                    n_single += 1
                                    single[idx] = 1
                                if counts[idx] == 0:
                                    n_uncovered += 1
                            
                            for idx in new_tsubs:
                                old_c = counts[idx]
                                counts[idx] = old_c + 1
                                if old_c == 0:
                                    n_uncovered -= 1
                                elif old_c == 1:
                                    n_single += 1
                                    single[idx] = 1
                                elif old_c == 2:
                                    n_single -= 1
                                    single[idx] = 0
                            
                            blocks[i] = new_block
                            block_tsubs[i] = new_tsubs
                            stuck_count = 0
            else:
                # If no blocks, rebuild
                blocks = greedy_cover()
                best_blocks = list(blocks)
                best_count = len(blocks)
                counts = [0] * n_tsubs
                for b in blocks:
                    for c in itertools.combinations(b, t):
                        idx = tsub_to_idx[frozenset(c)]
                        counts[idx] += 1
                single = [1 if c == 1 else 0 for c in counts]
                n_single = sum(single)
                n_uncovered = 0
                block_tsubs = [compute_block_tsub_indices(b) for b in blocks]
                stuck_count = 0
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
