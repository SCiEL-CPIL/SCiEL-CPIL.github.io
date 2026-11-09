import itertools
import time
from collections import Counter

# Precompute all 4-subsets of 19 points and their indices
_ALL_4SUBSETS = list(itertools.combinations(range(19), 4))
_NUM_4SUBSETS = len(_ALL_4SUBSETS)  # 3876
_4SUBSET_INDEX = {c: i for i, c in enumerate(_ALL_4SUBSETS)}

# For each point, bitmask of 4-subsets containing that point
_POINT_MASKS = []
for p in range(19):
    mask = 0
    for i, c in enumerate(_ALL_4SUBSETS):
        if p in c:
            mask |= (1 << i)
    _POINT_MASKS.append(mask)

# All 7-subsets of 19 points with their block masks
_ALL_7SUBSETS = list(itertools.combinations(range(19), 7))
_BLOCK_MASKS = []
for b in _ALL_7SUBSETS:
    mask = 0
    for c in itertools.combinations(b, 4):
        mask |= (1 << _4SUBSET_INDEX[c])
    _BLOCK_MASKS.append(mask)

# Precompute popcount for 4-subsets within a 7-subset (for local search)
# For each 7-subset, count of 4-subsets it covers = C(7,4) = 35

def _popcount(x):
    return bin(x).count('1')


def _greedy_cover(v, k, t, deadline):
    """Fast greedy covering using bitmask operations."""
    # For v=19, k=7, t=4, use precomputed structures
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        return _generic_greedy(v, k, t, deadline)
    
    uncovered = (1 << _NUM_4SUBSETS) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Find the first uncovered 4-subset
        # Get index of lowest set bit
        low_bit = uncovered & (-uncovered)
        idx = low_bit.bit_length() - 1
        first_4sub = _ALL_4SUBSETS[idx]
        
        # Start block with the 4 points of this 4-subset
        block = list(first_4sub)
        block_set = set(block)
        
        # Build current block mask
        block_mask = 0
        for p in block:
            block_mask |= _POINT_MASKS[p]
        
        # Greedily add points to maximize new coverage
        while len(block) < k:
            best_point = -1
            best_gain = 0
            for p in range(v):
                if p in block_set:
                    continue
                # Gain = number of uncovered 4-subsets that contain p AND are completed by current block
                # A 4-subset is covered by (block + p) if it contains p and 3 points from block
                # Using bitmask: 4-subsets containing p = _POINT_MASKS[p]
                # 4-subsets with at least 3 points in block = OR of C(block,3) 4-subsets
                # Simpler: gain = popcount(uncovered & _POINT_MASKS[p]) but that overcounts
                # Actually we need 4-subsets that contain p and whose other 3 points are in block
                # Let's compute: for each 3-subset of block, the 4-subset (3-subset + p)
                # That's C(len(block), 3) 4-subsets
                # But using bitmask: _POINT_MASKS[p] & (mask of 4-subsets with >=3 points in block)
                # The mask of 4-subsets with >=3 points in block = OR over all 4-subsets containing 3+ block points
                # Simpler approach: compute directly
                gain = 0
                for combo3 in itertools.combinations(block, 3):
                    # Check if (combo3 + p) is an uncovered 4-subset
                    c4 = tuple(sorted(combo3 + (p,)))
                    if c4 in _4SUBSET_INDEX:
                        bit = 1 << _4SUBSET_INDEX[c4]
                        if uncovered & bit:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain == 0:
                # No useful point, just add any available
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        block_mask = 0
        for p in block:
            block_mask |= _POINT_MASKS[p]
        uncovered &= ~block_mask
    
    return blocks


def _generic_greedy(v, k, t, deadline):
    """Generic greedy for non-19/7/4 parameters."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {c: i for i, c in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Point masks
    point_masks = []
    for p in range(v):
        mask = 0
        for i, c in enumerate(all_tsubs):
            if p in c:
                mask |= (1 << i)
        point_masks.append(mask)
    
    uncovered = (1 << num_tsubs) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        low_bit = uncovered & (-uncovered)
        idx = low_bit.bit_length() - 1
        first_tsub = all_tsubs[idx]
        
        block = list(first_tsub)
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = 0
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    c_t = tuple(sorted(combo + (p,)))
                    if c_t in tsub_index:
                        bit = 1 << tsub_index[c_t]
                        if uncovered & bit:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point == -1:
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        block_mask = 0
        for p in block:
            block_mask |= point_masks[p]
        uncovered &= ~block_mask
    
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: replace blocks with better ones."""
    if v != 19 or k != 7 or t != 4:
        return blocks
    
    num_4subs = _NUM_4SUBSETS
    all_7subs = _ALL_7SUBSETS
    block_masks = _BLOCK_MASKS
    
    # Current coverage
    covered = 0
    for b in blocks:
        idx = all_7subs.index(b)
        covered |= block_masks[idx]
    
    # Try removing blocks if still covered
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if block i is redundant
            without_i = 0
            for j in range(len(blocks)):
                if j != i:
                    idx = all_7subs.index(blocks[j])
                    without_i |= block_masks[idx]
            if (without_i | block_masks[all_7subs.index(blocks[i])]) == without_i:
                # Block i is redundant
                blocks.pop(i)
                covered = without_i
                changed = True
                break
    
    # Try replacing blocks
    for _ in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        best_improvement = 0
        best_swap = None
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            current_mask = block_masks[all_7subs.index(blocks[i])]
            # Try replacing with a random 7-subset
            import random
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_block = tuple(random.sample(range(v), k))
                new_block = tuple(sorted(new_block))
                new_mask = block_masks[all_7subs.index(new_block)]
                # Coverage without old block i
                rest_covered = 0
                for j in range(len(blocks)):
                    if j != i:
                        rest_covered |= block_masks[all_7subs.index(blocks[j])]
                old_total = rest_covered | current_mask
                new_total = rest_covered | new_mask
                if _popcount(new_total) > _popcount(old_total):
                    improvement = _popcount(new_total) - _popcount(old_total)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i, new_block)
        if best_swap:
            i, new_block = best_swap
            blocks[i] = new_block
            covered = 0
            for b in blocks:
                covered |= block_masks[all_7subs.index(b)]
        else:
            break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Run greedy
    blocks = _greedy_cover(v, k, t, deadline)
    
    # If time remains, try local search
    if time.time() < deadline - 5.0 and blocks:
        blocks = _local_search(blocks, v, k, t, deadline)
    
    # Convert to lists
    return [list(b) for b in blocks]
