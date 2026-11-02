"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys


def _greedy_cover_fast(v, k, t, deadline, rng):
    """
    Fast greedy covering using precomputed data structures.
    Returns list of blocks (tuples).
    """
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    
    # uncovered: boolean array, True if t-subset is not yet covered
    uncovered = bytearray([1] * n_tsub)
    n_uncovered = n_tsub
    blocks = []
    
    # Precompute for each pair of points (a,b), the list of t-subset indices containing both
    # This is expensive for large v but v=19 is small
    # Actually, let's precompute: for each point x, which t-subsets contain x
    # And for each block candidate, we can compute gain incrementally
    
    # For efficiency: precompute for each (subset of size up to k-1), the t-subsets it covers
    # But that's C(19,8) = 75582 blocks * C(8,5)=56 subsets = 4.2M entries - too many
    
    # Alternative: for each point x, precompute list of t-subset indices containing x
    point_tsubs = [[] for _ in range(v)]
    for idx, sub in enumerate(all_tsubsets):
        for x in sub:
            point_tsubs[x].append(idx)
    
    # Greedy: repeatedly pick a block that covers the most uncovered t-subsets
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered t-subset to seed the block
        # Find an uncovered index
        start_idx = rng.randint(0, n_tsub - 1)
        while not uncovered[start_idx]:
            start_idx = rng.randint(0, n_tsub - 1)
        
        start_sub = all_tsubsets[start_idx]
        block = list(start_sub)
        
        # Greedily extend block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_points = []
            for x in range(v):
                if x in block:
                    continue
                # Gain = number of uncovered t-subsets that would be covered by adding x
                # A t-subset S is newly covered if S ⊆ block ∪ {x} and S is uncovered
                # S must contain x and (t-1) elements from block
                # So: for each (t-1)-subset R of block, check if R ∪ {x} is an uncovered t-subset
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = sub_to_idx.get(sub, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain and gain > 0:
                    best_points.append(x)
            
            if not best_points or best_gain <= 0:
                break
            
            block.append(rng.choice(best_points))
        
        if len(block) < t:
            continue
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0 and uncovered[idx]:
                uncovered[idx] = 0
                n_uncovered -= 1
    
    return blocks


def _local_search_polish(blocks, v, k, t, deadline, rng):
    """
    Local search: try swapping one point in a block with another to increase total coverage.
    A swap is beneficial if the new block covers more t-subsets than the old block
    (counting only t-subsets that were previously covered only by this block).
    """
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    sub_to_idx = {sub: i for i, sub in enumerate(all_tsubsets)}
    
    # coverage_count[i] = number of blocks covering t-subset i
    coverage_count = [0] * n_tsub
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx.get(sub, -1)
            if idx >= 0:
                coverage_count[idx] += 1
    
    n_blocks = len(blocks)
    improved = True
    max_iterations = 200  # Cap iterations to stay within time
    
    while improved and time.time() < deadline:
        improved = False
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random block to try modifying
            bidx = rng.randint(0, n_blocks - 1)
            block = list(blocks[bidx])
            
            # Try each position and each replacement point
            best_swap = None
            best_gain = 0
            
            for pos in range(k):
                old_point = block[pos]
                for new_point in range(v):
                    if new_point == old_point:
                        continue
                    if new_point in block:
                        continue
                    
                    # Compute gain of swap
                    # T-subsets lost: those containing old_point but not new_point, covered only by this block
                    # T-subsets gained: those containing new_point but not old_point, not covered by any other block
                    new_block = block[:pos] + [new_point] + block[pos+1:]
                    
                    gain = 0
                    # T-subsets that would be newly covered (contain new_point, not old_point)
                    for sub in itertools.combinations(new_block, t):
                        if new_point not in sub:
                            continue
                        idx = sub_to_idx.get(sub, -1)
                        if idx >= 0 and coverage_count[idx] == 0:
                            gain += 1
                    
                    # T-subsets that would be lost (contain old_point, not new_point)
                    for sub in itertools.combinations(block, t):
                        if old_point not in sub:
                            continue
                        idx = sub_to_idx.get(sub, -1)
                        if idx >= 0 and coverage_count[idx] == 1:
                            gain -= 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (pos, old_point, new_point)
            
            if best_swap and best_gain > 0:
                pos, old_point, new_point = best_swap
                block = list(blocks[bidx])
                
                # Update coverage counts for old block
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx.get(sub, -1)
                    if idx >= 0:
                        coverage_count[idx] -= 1
                
                # Perform swap
                block[pos] = new_point
                block.sort()
                new_block = tuple(block)
                blocks[bidx] = new_block
                
                # Update coverage counts for new block
                for sub in itertools.combinations(new_block, t):
                    idx = sub_to_idx.get(sub, -1)
                    if idx >= 0:
                        coverage_count[idx] += 1
                
                improved = True
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, deadline):
    """
    Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
    """
    if not blocks:
        return blocks
    
    sub_to_idx = {}
    idx = 0
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = idx
        idx += 1
    
    n_tsubsets = len(sub_to_idx)
    coverage = [set() for _ in range(n_tsubsets)]
    
    for bidx, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                coverage[sub_to_idx[sub]].add(bidx)
    
    non_redundant = []
    for bidx, block in enumerate(blocks):
        if time.time() > deadline:
            break
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub in sub_to_idx:
                idx = sub_to_idx[sub]
                if len(coverage[idx]) == 1:
                    redundant = False
                    break
        if not redundant:
            non_redundant.append(block)
    
    return non_redundant


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple random restarts
    for seed in [42, 1, 2, 3, 7, 13]:
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        blocks = _greedy_cover_fast(v, k, t, deadline, rng)
        
        # Verify coverage
        if blocks:
            sub_to_idx = {sub: i for i, sub in enumerate(itertools.combinations(range(v), t))}
            covered = bytearray([0] * len(sub_to_idx))
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx.get(sub, -1)
                    if idx >= 0:
                        covered[idx] = 1
            n_uncovered = sum(1 - c for c in covered)
            
            if n_uncovered == 0:
                # Try local search polish
                if time.time() < deadline - 10:
                    polished = _local_search_polish(blocks, v, k, t, deadline, random.Random(seed + 100))
                    if len(polished) < len(blocks):
                        blocks = polished
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks
    
    # Remove redundant blocks from best
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _remove_redundant_blocks(best_blocks, v, t, deadline)
    
    if best_blocks is None:
        return []
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
