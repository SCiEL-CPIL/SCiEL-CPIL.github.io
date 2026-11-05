import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    num_tsubsets = len(all_tsubsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}

    # Precompute all k-blocks and their t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_per_block = int(np.math.comb(k, t))
    
    # For each block, precompute the list of t-subset indices it covers
    block_t_indices = []
    for blk in all_blocks:
        indices = [t_sub_to_idx[sub] for sub in itertools.comstitutions(blk, t)]
        block_t_indices.append(indices)
    
    # Precompute incidence matrix A (num_blocks x num_tsubsets) as sparse-ish
    # But since num_blocks * t_per_block is manageable (10626 * 35 = 371910), we can use dense or list of lists
    # Actually, for speed in greedy, we want to quickly compute gains.
    
    # Build a list of lists: for each t-subset index, which blocks cover it?
    # Or for each block, which t-subsets it covers?
    # Let's build block_to_tsubs: list of lists of int (t-subset indices)
    block_to_tsubs = block_t_indices
    
    # And tsub_to_blocks: list of lists of int (block indices)
    tsub_to_blocks = [[] for _ in range(num_tsubsets)]
    for bi, tsubs in enumerate(block_to_tsubs):
        for ts in tsubs:
            tsub_to_blocks[ts].append(bi)
    
    best_blocks = None
    best_len = float('inf')

    # Run multiple strategies
    strategies = ['standard', 'weighted', 'randomized', 'hybrid']
    trial = 0
    while time.time() < deadline - 5.0:
        strategy = strategies[trial % len(strategies)]
        
        # Greedy construction
        blocks = _greedy_cover(v, k, t, all_blocks, block_to_tsubs, all_tsubsets_set, deadline, trial, mode=strategy)
        
        if blocks is None:
            trial += 1
            continue

        # Local improvement
        improved_blocks = _remove_redundant(blocks, v, k, t, deadline)
        if len(improved_blocks) < len(blocks):
            blocks = improved_blocks

        # Try swap-based improvement
        if time.time() < deadline - 2.0:
            swapped_blocks = _swap_improve(blocks, v, k, t, all_tsubsets_set, deadline, trial)
            if len(swapped_blocks) < len(blocks):
                blocks = swapped_blocks
                blocks = _remove_redundant(blocks, v, k, t, deadline)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

        # Early stop if we hit a very good number
        if best_len <= 76:
            break

        trial += 1

    if best_blocks is None:
        # Fallback
        blocks = _greedy_cover(v, k, t, all_blocks, block_to_tsubs, all_tsubsets_set, deadline, -1, mode='standard')
        if blocks:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []

    return best_blocks

def _greedy_cover(v, k, t, all_blocks, block_to_tsubs, all_tsubsets_set, deadline, seed_val, mode='standard'):
    """
    Greedy covering with different modes.
    Uses precomputed block_to_tsubs for speed.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    # uncovered: set of t-subset indices (integers)
    uncovered = set(range(len(all_tsubsets_set)))
    blocks = []
    
    # For weighted mode, precompute weights
    # Weight = 1 / (number of blocks covering this t-subset)
    # This favors covering "rare" subsets early
    if mode == 'weighted' or mode == 'hybrid':
        tsub_freq = np.zeros(len(all_tsubsets_set), dtype=np.int32)
        for ts_idx in range(len(all_tsubsets_set)):
            tsub_freq[ts_idx] = len([b for b in all_blocks if ts_idx in block_to_tsubs[b]])
        # Actually, this is expensive to compute naively. Let's use the tsub_to_blocks logic if available, 
        # or just approximate. For now, let's stick to gain-based for speed, 
        # or use a simple heuristic.
        # Let's refine: we can compute freq once outside if we pass it, but here we'll just use gain.
        # To make 'weighted' distinct, we can penalize blocks that cover many "easy" (frequent) subsets.
        # Let's compute freq properly once per call if needed, or just use a different tie-breaking.
        # Given the time constraint, let's keep it simple: 'weighted' uses gain, 'standard' uses gain, 
        # but 'randomized' picks from top candidates.
        
    while uncovered:
        if time.time() > deadline:
            return None

        # Find block with max gain
        best_gain = -1
        best_block_idx = -1
        best_candidates = []
        
        for bi, tsubs in enumerate(block_to_tsubs):
            gain = 0
            for ts in tsubs:
                if ts in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
                best_candidates = [bi]
            elif gain == best_gain:
                best_candidates.append(bi)
        
        if best_gain == 0:
            # Should not happen if all_blocks covers all tsubsets
            return None

        if mode == 'randomized' and len(best_candidates) > 1:
            chosen_bi = rng.choice(best_candidates)
        else:
            chosen_bi = best_block_idx

        block = all_blocks[chosen_bi]
        blocks.append(list(block))
        
        for ts in block_to_tsubs[chosen_bi]:
            uncovered.discard(ts)

    return blocks

def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks.
    """
    if not blocks:
        return blocks

    cover_count = defaultdict(int)
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    improved = list(blocks)

    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break

        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break

        if redundant:
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)

    return improved

def _swap_improve(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Try 1-for-1 block replacement.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    improved = list(blocks)
    n_blocks = len(improved)

    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    critical = set(s for s, c in cover_count.items() if c == 1)

    for _pass in range(2):
        if time.time() > deadline:
            break

        any_change = False
        for i in range(n_blocks):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))
            old_tsub_set = set(old_tsubs)

            old_critical = [s for s in old_tsubs if s in critical]

            if not old_critical:
                continue

            candidates = set()
            for _ in range(20):
                new_block = list(old_block)
                remove_count = rng.randint(1, min(2, k))
                to_remove = rng.sample(new_block, remove_count)
                for x in to_remove:
                    new_block.remove(x)
                while len(new_block) < k:
                    x = rng.randint(0, v-1)
                    if x not in new_block:
                        new_block.append(x)
                candidates.add(tuple(sorted(new_block)))

            best_new_block = None
            best_gain = -1

            for cand in candidates:
                if cand == old_block:
                    continue

                cand_tsubs = set(itertools.combinations(cand, t))
                
                if not all(s in cand_tsubs for s in old_critical):
                    continue

                new_subs = cand_tsubs - old_tsub_set
                gain = len(new_subs)
                
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = cand

            if best_new_block is not None and best_gain > 0:
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                for s in itertools.combinations(best_new_block, t):
                    cover_count[s] += 1

                improved[i] = best_new_block
                any_change = True
                critical = set(s for s, c in cover_count.items() if c == 1)

        if not any_change:
            break

    return improved
