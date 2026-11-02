import itertools
import time
import random
from collections import defaultdict

def _precompute_blocks(v, k, t):
    """Precompute all k-subsets and their t-subsets."""
    all_blocks = []
    block_triples_map = {}  # block tuple -> list of triples
    for block in itertools.combinations(range(v), k):
        tris = list(itertools.combinations(block, t))
        all_blocks.append(block)
        block_triples_map[block] = tris
    return all_blocks, block_triples_map

def _build_pivot_index(all_blocks, block_triples_map, t):
    """Map each t-subset (pivot) to list of block indices containing it."""
    pivot_index = defaultdict(list)
    for idx, block in enumerate(all_blocks):
        for tri in block_triples_map[block]:
            pivot_index[tri].append(idx)
    return pivot_index

def _greedy_cover_fast(v, k, t, rng, time_limit, all_blocks, block_triples_map, pivot_index):
    """Fast greedy using precomputed data."""
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            break
        
        # Pick random uncovered triple as pivot
        pivot_triple = rng.choice(tuple(uncovered))
        
        # Get all blocks containing this pivot
        candidate_indices = pivot_index[pivot_triple]
        
        best_gain = -1
        best_idx = -1
        
        for idx in candidate_indices:
            block = all_blocks[idx]
            tris = block_triples_map[block]
            gain = 0
            for tri in tris:
                if tri in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx >= 0:
            block = all_blocks[best_idx]
            blocks.append(list(block))
            for tri in block_triples_map[block]:
                uncovered.discard(tri)
    
    return blocks

def _reduce_covering(blocks, v, k, t):
    """Remove redundant blocks."""
    if not blocks:
        return blocks
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        for i, block in enumerate(blocks):
            is_essential = False
            for tri in block_triples[i]:
                if triple_cover_count[tri] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                changed = True
                for tri in block_triples[i]:
                    triple_cover_count[tri] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search(blocks, v, k, t, rng, time_limit, all_blocks, block_triples_map, pivot_index):
    """Local search: try replacing blocks with better ones, then reduce."""
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            original_block = blocks[i]
            original_triples = block_triples[i]
            original_block_tuple = tuple(original_block)
            
            # Find triples covered only by this block (rare triples)
            rare_triples = [tri for tri in original_triples if triple_cover_count[tri] == 1]
            
            # Strategy: find a block that covers all rare_triples and maximizes new coverage
            if rare_triples:
                # Find blocks containing all rare triples
                candidate_indices = set(range(len(all_blocks)))
                for tri in rare_triples:
                    candidate_indices &= set(pivot_index.get(tri, []))
                
                best_gain = -1
                best_idx = -1
                for idx in candidate_indices:
                    block = all_blocks[idx]
                    if block == original_block_tuple:
                        continue
                    tris = block_triples_map[block]
                    # Check if swap is valid: all lost triples must be covered by others
                    old_set = set(original_triples)
                    new_set = set(tris)
                    lost = old_set - new_set
                    valid = all(triple_cover_count[tri] > 1 for tri in lost)
                    if not valid:
                        continue
                    gain = sum(1 for tri in tris if tri in set(itertools.combinations(range(v), t)) - set().union(*[set(block_triples[j]) for j in range(len(blocks)) if j != i]))
                    # Simpler: count new uncovered triples
                    gain = 0
                    for tri in tris:
                        if tri not in old_set and triple_cover_count[tri] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
            else:
                # No rare triples, try to find a block that covers more currently-uncovered triples
                # or just a random swap to enable reduction
                candidate_indices = set(range(len(all_blocks)))
                best_gain = -1
                best_idx = -1
                for idx in candidate_indices:
                    block = all_blocks[idx]
                    if block == original_block_tuple:
                        continue
                    tris = block_triples_map[block]
                    old_set = set(original_triples)
                    new_set = set(tris)
                    lost = old_set - new_set
                    valid = all(triple_cover_count[tri] > 1 for tri in lost)
                    if not valid:
                        continue
                    gain = sum(1 for tri in tris if tri not in old_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
            
            if best_idx >= 0:
                new_block = all_blocks[best_idx]
                new_triples = block_triples_map[new_block]
                
                # Perform swap
                for tri in original_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = list(new_block)
                block_triples[i] = list(new_triples)
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    triple_cover_count = defaultdict(int)
                    block_triples = []
                    for block in blocks:
                        tris = list(itertools.combinations(block, t))
                        block_triples.append(tris)
                        for tri in tris:
                            triple_cover_count[tri] += 1
                    break
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_triples_map = _precompute_blocks(v, k, t)
    pivot_index = _build_pivot_index(all_blocks, block_triples_map, t)
    
    best_blocks = None
    best_size = float('inf')
    
    # Multiple greedy runs with different seeds
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(seed)
        blocks = _greedy_cover_fast(v, k, t, rng, deadline, all_blocks, block_triples_map, pivot_index)
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Local search on the best so far
            if time.time() < deadline - 30:
                rng_ls = random.Random(seed + 10000)
                improved = _local_search(best_blocks, v, k, t, rng_ls, deadline - 10, all_blocks, block_triples_map, pivot_index)
                improved = _reduce_covering(improved, v, k, t)
                if len(improved) < best_size:
                    best_size = len(improved)
                    best_blocks = improved
        
        if best_size <= 69:
            break
    
    return best_blocks
