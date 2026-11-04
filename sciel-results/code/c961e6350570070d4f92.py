"""Covering design C(19,7,4) via greedy + local search."""
import itertools
import random
import time
import numpy as np
from math import comb


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    n_tsubsets = comb(v, t)
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute for each point: bitmask of t-subsets containing it
    # Using numpy bitarrays for speed
    point_masks = [np.zeros(n_tsubsets, dtype=bool) for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_masks[p][i] = True

    # For each point pair, precompute t-subsets containing both
    # point_pair_masks[(p,q)] = bitmask
    point_pair_masks = {}
    for p in range(v):
        for q in range(p + 1, v):
            mask = point_masks[p] & point_masks[q]
            point_pair_masks[(p, q)] = mask

    # For each triple, precompute t-subsets containing all three
    point_triple_masks = {}
    for p in range(v):
        for q in range(p + 1, v):
            for r in range(q + 1, v):
                mask = point_masks[p] & point_masks[q] & point_masks[r]
                point_triple_masks[(p, q, r)] = mask

    # For a block B, the t-subsets it covers:
    # union over all t-subsets of B
    # Precompute for each possible block (as frozenset) the coverage mask
    # That's C(19,7) = 50388 blocks, each with C(7,4)=35 t-subsets
    # Total: 50388 * 35 = 1,763,580 entries - feasible

    # Actually, let's precompute coverage for all 7-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-subsets it covers
    # block_cover[block_idx] = set of t-subset indices
    # For memory, use a list of sets
    block_covers = []
    for block in all_blocks:
        covers = set()
        for ts in itertools.combinations(block, t):
            covers.add(tsubset_to_idx[ts])
        block_covers.append(covers)

    # Block index lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    def greedy_cover(seed=None, use_random_seed=True):
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(n_tsubsets))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return blocks, False
            
            # Pick a random uncovered t-subset as seed
            seed_ts = random.choice(list(uncovered))
            seed_tuple = all_tsubsets[seed_ts]
            block = list(seed_tuple)
            
            while len(block) < k:
                if time.time() > deadline:
                    return blocks, False
                block_set = set(block)
                
                best_x = None
                best_gain = -1
                
                # For each candidate point x, compute gain
                # gain(x) = number of uncovered t-subsets S such that x in S and S\{x} ⊆ block
                # S\{x} is a (t-1)-subset of block
                # For t=4: S\{x} is a triple, so we check if each triple of block + x forms an uncovered 4-set
                
                block_triples = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for triple in block_triples:
                        S = tuple(sorted(triple + (x,)))
                        idx = tsubset_to_idx.get(S)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered.discard(idx)
        
        return blocks, len(uncovered) == 0

    def local_search(blocks, max_iterations=200):
        """Try to improve by removing redundant blocks and swapping."""
        if time.time() > deadline - 10:
            return blocks
        
        blocks = list(blocks)
        n = len(blocks)
        
        # Compute coverage count for each t-subset
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[tsubset_to_idx[ts]] += 1
        
        # Try to remove blocks that don't uniquely cover any t-subset
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_blocks = []
            for i, block in enumerate(blocks):
                if time.time() > deadline - 10:
                    break
                # Check if this block is needed
                # A block is needed if it's the only one covering some t-subset
                needed = False
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[ts]
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if not needed:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx[ts]
                        cover_count[idx] -= 1
                    improved = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
        
        # Try swaps: replace one block with another that covers more uncovered
        # Actually, after removal, all t-subsets should still be covered
        # Try to add blocks to reduce count further - this is hard
        # Instead, try a different greedy restart with the current solution as warm start
        
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random seeds
    for seed in range(100):
        if time.time() > deadline - 15:
            break
        blocks, complete = greedy_cover(seed=seed)
        if complete:
            # Apply local search
            blocks = local_search(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count <= 151:
                    break
    
    # If no complete solution, try deterministic greedy
    if best_blocks is None:
        uncovered = set(range(n_tsubsets))
        blocks = []
        while uncovered and time.time() < deadline - 5:
            seed_ts = min(uncovered)
            seed_tuple = all_tsubsets[seed_ts]
            block = list(seed_tuple)
            while len(block) < k:
                block_set = set(block)
                best_x = None
                best_gain = -1
                block_triples = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for triple in block_triples:
                        S = tuple(sorted(triple + (x,)))
                        idx = tsubset_to_idx.get(S)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered.discard(idx)
        
        if len(uncovered) == 0:
            best_blocks = local_search(blocks)
            best_count = len(best_blocks)
    
    if best_blocks is None:
        # Emergency fallback
        blocks = []
        for combo in itertools.combinations(range(v), k):
            blocks.append(list(combo))
            if time.time() > deadline - 1:
                break
        return blocks
    
    return best_blocks
