"""Local search for covering design C(24,7,3) with efficient coverage tracking."""
import itertools
import random
import time

# Precompute all C(24,3)=2024 triples as integer indices
# Triple (a,b,c) with a<b<c maps to a unique index 0..2023
def _build_triple_index(v, t):
    """Build mapping from (a,b,c) sorted tuple to index, and list of triples."""
    triples = []
    triple_to_idx = {}
    idx = 0
    for c in itertools.combinations(range(v), t):
        triple_to_idx[c] = idx
        triples.append(c)
        idx += 1
    return triples, triple_to_idx

def _block_triple_indices(block, triple_to_idx):
    """Return list of triple indices covered by a block."""
    return [triple_to_idx[c] for c in itertools.combinations(block, t)]

def _greedy_cover(v, k, t, rng, triples, triple_to_idx):
    """Greedy construction: repeatedly pick block covering most uncovered triples."""
    n_triples = len(triples)
    uncovered = bytearray([1] * n_triples)
    n_uncov = n_triples
    blocks = []
    
    # Precompute for each point, the list of triples containing it (for fast gain calc)
    # triples_by_point[x] = list of (triple_idx, other_two_points)
    triples_by_point = [[] for _ in range(v)]
    for idx, (a, b, c) in enumerate(triples):
        triples_by_point[a].append((idx, b, c))
        triples_by_point[b].append((idx, a, c))
        triples_by_point[c].append((idx, a, b))
    
    while n_uncov > 0:
        # Pick a random uncovered triple as seed
        # Find an uncovered triple
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break
        if seed_idx < 0:
            break
        seed = set(triples[seed_idx])
        
        # Greedily add points to maximize coverage of uncovered triples
        block_set = set(seed)
        while len(block_set) < k:
            best_x = None
            best_gain = 0
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # For each pair in block_set, count uncovered triples containing x and that pair
                # We need: for each (i,j) in block_set with i<j, check if (i,j,x) is uncovered
                # More efficiently: for each triple containing x, check if other two are in block_set
                for (tidx, p1, p2) in triples_by_point[x]:
                    if uncovered[tidx]:
                        # Check if both p1,p2 are in block_set
                        if p1 in block_set and p2 in block_set:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain == 0:
                break
            block_set.add(best_x)
        
        # If we couldn't fill to k, still use what we have (but need exactly k)
        if len(block_set) < k:
            # Fill randomly
            remaining = [x for x in range(v) if x not in block_set]
            rng.shuffle(remaining)
            for x in remaining:
                if len(block_set) >= k:
                    break
                block_set.add(x)
        
        block = tuple(sorted(block_set))
        blocks.append(block)
        
        # Update uncovered
        for tidx in _block_triple_indices(block, triple_to_idx):
            if uncovered[tidx]:
                uncovered[tidx] = 0
                n_uncov -= 1
    
    return blocks

def _try_remove_block(blocks, triple_counts, block_triples, i):
    """Try to remove block i. Returns True if removed."""
    bt = block_triples[i]
    # Check if any triple in this block has count == 1
    for tidx in bt:
        if triple_counts[tidx] == 1:
            return False
    # Safe to remove
    for tidx in bt:
        triple_counts[tidx] -= 1
    del blocks[i]
    del block_triples[i]
    return True

def _local_search(blocks, triple_counts, block_triples, v, k, t, rng, triples, triple_to_idx, time_limit, start_time):
    """Local search: remove redundant blocks and perform swaps."""
    n_blocks = len(blocks)
    
    while time.time() - start_time < time_limit:
        # Phase 1: Remove redundant blocks (repeat until none)
        removed_any = True
        while removed_any:
            removed_any = False
            for i in range(len(blocks)):
                if _try_remove_block(blocks, triple_counts, block_triples, i):
                    n_blocks = len(blocks)
                    removed_any = True
                    break
        
        # Phase 2: Try swaps
        # For each block, try to replace it with a better one
        for i in range(len(blocks)):
            if time.time() - start_time > time_limit:
                break
            
            old_block = blocks[i]
            old_bt = block_triples[i]
            
            # Find critical triples (count==1) in this block
            critical = []
            for tidx in old_bt:
                if triple_counts[tidx] == 1:
                    critical.append(tidx)
            
            if not critical:
                # No critical triples, can remove (should have been caught in phase 1)
                continue
            
            # Points that must be in new block (union of all critical triples)
            must_points = set()
            for tidx in critical:
                must_points.update(triples[tidx])
            
            if len(must_points) > k:
                continue
            
            # Try to build a new block containing must_points
            # Fill remaining k - len(must_points) slots greedily to maximize coverage
            new_set = set(must_points)
            slots = k - len(new_set)
            
            if slots > 0:
                # Candidates: points not in new_set
                cands = [x for x in range(v) if x not in new_set]
                rng.shuffle(cands)
                
                # Greedy fill: pick point that covers most uncovered or low-count triples
                for x in cands:
                    if len(new_set) >= k:
                        break
                    # Compute gain: number of triples containing x and (t-1) points in new_set
                    # that have low count
                    gain = 0
                    # For each (t-1)-subset of new_set, check if adding x creates a low-count triple
                    new_set_list = list(new_set)
                    for rest in itertools.combinations(new_set_list, t - 1):
                        # Check triple (rest + x)
                        tri = tuple(sorted(list(rest) + [x]))
                        tidx = triple_to_idx.get(tri, -1)
                        if tidx >= 0:
                            c = triple_counts[tidx]
                            if c == 0:
                                gain += 10
                            elif c == 1:
                                gain += 1
                    if gain > 0:
                        new_set.add(x)
                    elif len(new_set) < k and len(cands) > 0:
                        new_set.add(x)
            
            if len(new_set) != k:
                continue
            
            new_block = tuple(sorted(new_set))
            if new_block == old_block:
                continue
            
            new_bt = _block_triple_indices(new_block, triple_to_idx)
            
            # Validate: after removing old_block and adding new_block, no triple becomes uncovered
            # Check: for each triple in old_block with count==1, it must be in new_block
            new_bt_set = set(new_bt)
            valid = True
            for tidx in old_bt:
                if triple_counts[tidx] == 1 and tidx not in new_bt_set:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Perform swap
            for tidx in old_bt:
                triple_counts[tidx] -= 1
            for tidx in new_bt:
                triple_counts[tidx] += 1
            blocks[i] = new_block
            block_triples[i] = new_bt
        
        # If we can't make progress, try a random perturbation
        # Replace a random block with a random valid block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_bt = block_triples[i]
            
            # Try random replacement
            for attempt in range(10):
                # Random block
                new_block = tuple(sorted(rng.sample(range(v), k)))
                if new_block == old_block:
                    continue
                new_bt = _block_triple_indices(new_block, triple_to_idx)
                
                # Validate
                new_bt_set = set(new_bt)
                valid = True
                for tidx in old_bt:
                    if triple_counts[tidx] == 1 and tidx not in new_bt_set:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Perform swap
                for tidx in old_bt:
                    triple_counts[tidx] -= 1
                for tidx in new_bt:
                    triple_counts[tidx] += 1
                blocks[i] = new_block
                block_triples[i] = new_bt
                break
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin
    
    # Build triple index
    triples, triple_to_idx = _build_triple_index(v, t)
    n_triples = len(triples)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different seeds
    seeds = [42, 123, 456, 789, 1011, 2024, 3141, 2718, 1618, 1414]
    for seed in seeds:
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(seed)
        
        # Greedy init
        blocks = _greedy_cover(v, k, t, rng, triples, triple_to_idx)
        
        # Build coverage data
        triple_counts = [0] * n_triples
        block_triples = []
        for b in blocks:
            bt = _block_triple_indices(b, triple_to_idx)
            block_triples.append(bt)
            for tidx in bt:
                triple_counts[tidx] += 1
        
        # Verify all triples covered
        if min(triple_counts) < 0:
            continue
        
        # Local search
        remaining_time = time_limit - (time.time() - start_time)
        blocks = _local_search(blocks, triple_counts, block_triples, v, k, t, rng, triples, triple_to_idx, remaining_time * 0.8, start_time)
        
        # Final polish: try removing blocks one more time
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            if _try_remove_block(blocks, triple_counts, block_triples, i):
                pass
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
    
    # Final verification and polish
    if best_blocks is None:
        # Fallback: greedy only
        rng = random.Random(42)
        best_blocks = _greedy_cover(v, k, t, rng, triples, triple_to_idx)
    
    # Final exhaustive removal pass
    triple_counts = [0] * n_triples
    for b in best_blocks:
        for tidx in _block_triple_indices(b, triple_to_idx):
            triple_counts[tidx] += 1
    
    changed = True
    while changed:
        changed = False
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            bt = _block_triple_indices(best_blocks[i], triple_to_idx)
            removable = True
            for tidx in bt:
                if triple_counts[tidx] == 1:
                    removable = False
                    break
            if removable:
                for tidx in bt:
                    triple_counts[tidx] -= 1
                del best_blocks[i]
                changed = True
    
    return [list(b) for b in best_blocks]
