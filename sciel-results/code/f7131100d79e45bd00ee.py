"""Covering design C(19,8,5) - NumPy-accelerated randomized greedy with multi-restart + local search."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    total_tsubsets = sum(1 for _ in itertools.combinations(range(v), t))
    
    # Precompute all t-subsets as bitmasks
    all_tsubsets = [0] * total_tsubsets
    idx = 0
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_tsubsets[idx] = mask
        idx += 1
    
    # Map bitmask to index for fast lookup
    mask_to_idx = {m: i for i, m in enumerate(all_tsubsets)}
    
    # Precompute for each possible block (sorted k-tuple), the list of t-subset indices it covers
    # For v=19, k=8: C(19,8)=75582 blocks, each with C(8,5)=56 t-subsets
    block_tsub_cache = {}
    for blk in itertools.combinations(range(v), k):
        tsubs = []
        for comb in itertools.combinations(blk, t):
            mask = 0
            for y in comb:
                mask |= (1 << y)
            sub_idx = mask_to_idx.get(mask, -1)
            if sub_idx != -1:
                tsubs.append(sub_idx)
        block_tsub_cache[blk] = np.array(tsubs, dtype=np.int32)
    
    # Precompute for each point x and each (t-1)-subset rest (as bitmask), the t-subset index of rest+x
    # For gain computation during greedy: given current_block, for each candidate x,
    # gain = number of uncovered t-subsets that are subsets of current_block ∪ {x}
    # A t-subset S of current_block ∪ {x} either contains x (S = R ∪ {x} where R ⊂ current_block, |R|=t-1)
    # or doesn't (S ⊂ current_block, already counted). So new gain from x = count of uncovered R∪{x} for R in C(current_block, t-1)
    
    # Precompute: for each point x (0..v-1) and each (t-1)-subset of the other v-1 points,
    # the t-subset index. But that's C(18,4)*19 = 3060*19 = 58140 entries. Let's do it differently.
    # For each x in 0..v-1, and each combination of t-1 points from the remaining v-1 points:
    point_rest_tsub_idx = {}
    for x in range(v):
        others = [y for y in range(v) if y != x]
        for rest in itertools.combinations(others, t - 1):
            mask = (1 << x)
            for y in rest:
                mask |= (1 << y)
            sub_idx = mask_to_idx.get(mask, -1)
            if sub_idx != -1:
                point_rest_tsub_idx[(x, rest)] = sub_idx
    
    # NumPy array for uncovered status
    uncovered_arr = np.ones(total_tsubsets, dtype=bool)
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        uncovered_arr[:] = True
        uncovered_count = total_tsubsets
        blocks = []
        
        while uncovered_count > 0:
            if time.time() >= deadline:
                break
            
            # Find an uncovered t-subset to start a new block
            start_idx = -1
            while start_idx == -1:
                r = random.randint(0, total_tsubsets - 1)
                if uncovered_arr[r]:
                    start_idx = r
            
            start_tsub = all_tsubsets[start_idx]
            
            # Greedily build a block: start with points from start_tsub, then add best-gain points
            block = list(range(v))
            random.shuffle(block)
            
            # Initialize with the t points from start_tsub
            start_points = [i for i in range(v) if (start_tsub >> i) & 1]
            remaining = [x for x in block if x not in start_points]
            current_block = list(start_points)
            
            while len(current_block) < k:
                if time.time() >= deadline:
                    break
                
                # For each candidate point, compute how many new t-subsets it covers
                best_gain = -1
                best_point = None
                
                # Compute t-1 subsets of current_block
                t_minus_1_combos = list(itertools.combinations(current_block, t - 1))
                
                for x in remaining:
                    # Calculate indices of t-subsets formed by rest + x
                    indices = []
                    for rest in t_minus_1_combos:
                        sub_idx = point_rest_tsub_idx.get((x, rest), -1)
                        if sub_idx != -1:
                            indices.append(sub_idx)
                    
                    if indices:
                        gain = int(uncovered_arr[indices].sum())
                    else:
                        gain = 0
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None or best_gain == 0:
                    best_point = random.choice(remaining)
                
                current_block.append(best_point)
                remaining.remove(best_point)
            
            # Now current_block has k points. Use precomputed cache for fast coverage update.
            blk_tuple = tuple(sorted(current_block))
            tsub_indices = block_tsub_cache[blk_tuple]
            newly_covered = int(uncovered_arr[tsub_indices].sum())
            uncovered_arr[tsub_indices] = False
            uncovered_count -= newly_covered
            blocks.append(sorted(current_block))
        
        if uncovered_count == 0:
            # Local search: remove redundant blocks
            blocks = _remove_redundant_blocks(blocks, v, k, t, deadline, block_tsub_cache, mask_to_idx, total_tsubsets)
            # Block swap local search
            if time.time() < deadline - 10.0:
                blocks = _swap_local_search(blocks, v, k, t, deadline, block_tsub_cache, mask_to_idx, total_tsubsets)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
    
    if best_blocks is None:
        # Fallback: run deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline, block_tsub_cache, mask_to_idx, total_tsubsets):
    """Iteratively remove blocks whose removal still leaves all t-subsets covered."""
    if not blocks:
        return blocks
    
    # Count coverage using precomputed cache
    coverage = np.zeros(total_tsubsets, dtype=np.int32)
    block_tsubsets = []
    for block in blocks:
        blk_tuple = tuple(sorted(block))
        tsubs = block_tsub_cache[blk_tuple]
        block_tsubsets.append(tsubs)
        coverage[tsubs] += 1
    
    active = [True] * len(blocks)
    
    # Iteratively try to remove blocks
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        order = list(range(len(blocks)))
        random.shuffle(order)
        
        for bi in order:
            if not active[bi]:
                continue
            tsubs = block_tsubsets[bi]
            if len(tsubs) == 0:
                active[bi] = False
                changed = True
                continue
            
            min_cov = int(coverage[tsubs].min())
            if min_cov > 1:
                active[bi] = False
                coverage[tsubs] -= 1
                changed = True
    
    result = [blocks[i] for i in range(len(blocks)) if active[i]]
    return result


def _swap_local_search(blocks, v, k, t, deadline, block_tsub_cache, mask_to_idx, total_tsubsets):
    """Try to remove one block and replace it with a better one, or just remove if possible."""
    if len(blocks) < 2:
        return blocks
    
    coverage = np.zeros(total_tsubsets, dtype=np.int32)
    block_tsubsets = []
    for block in blocks:
        blk_tuple = tuple(sorted(block))
        tsubs = block_tsub_cache[blk_tuple]
        block_tsubsets.append(tsubs)
        coverage[tsubs] += 1
    
    n = len(blocks)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        order = list(range(n))
        random.shuffle(order)
        
        for bi in order:
            if len(blocks) <= 1:
                break
            tsubs = block_tsubsets[bi]
            if len(tsubs) == 0:
                # Remove empty block
                coverage[tsubs] -= 1
                blocks.pop(bi)
                block_tsubsets.pop(bi)
                n -= 1
                improved = True
                break
            
            # Check if we can remove this block (all its t-subsets covered by others)
            if int(coverage[tsubs].min()) > 1:
                coverage[tsubs] -= 1
                blocks.pop(bi)
                block_tsubsets.pop(bi)
                n -= 1
                improved = True
                break
    
    return blocks
# EVOLVE-BLOCK-END
