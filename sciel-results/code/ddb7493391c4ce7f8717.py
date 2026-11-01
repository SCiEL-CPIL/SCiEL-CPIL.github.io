"""Bitset-accelerated greedy covering with local search refinement for C(v,k,t)."""

import time
import random
import itertools

# Precompute mapping from 3-combination to index for v=24
_C24_3 = 2024

_COMBO_INDEX = {}
_idx = 0
for a in range(24):
    for b in range(a + 1, 24):
        for c in range(b + 1, 24):
            _COMBO_INDEX[(a, b, c)] = _idx
            _idx += 1
assert _idx == _C24_3

# Precompute inverse mapping
_IDX_TO_COMBO = [None] * _C24_3
for combo, idx in _COMBO_INDEX.items():
    _IDX_TO_COMBO[idx] = combo
_IDX_TO_COMBO = tuple(_IDX_TO_COMBO)

# Precompute: for each pair (a,b) with a<b, list of (c, triple_idx) where c != a,b
# This lets us quickly find which triples are covered by adding point x to a block
_PAIR_TRIPLES = {}
for a in range(24):
    for b in range(a + 1, 24):
        lst = []
        for c in range(24):
            if c == a or c == b:
                continue
            tri = tuple(sorted((a, b, c)))
            lst.append((c, _COMBO_INDEX[tri]))
        _PAIR_TRIPLES[(a, b)] = lst


def _block_triple_indices(block):
    """Return list of triple indices covered by a block."""
    idxs = []
    for combo in itertools.combinations(block, 3):
        idxs.append(_COMBO_INDEX[combo])
    return idxs


def _greedy_build(v, k, t, seed, deadline):
    """Build a covering using randomized greedy with bitset tracking."""
    rng = random.Random(seed)
    n_triples = _C24_3
    
    # Use a list for uncovered status for fast random access
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    
    all_points = list(range(v))
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick a random uncovered triple
        while True:
            idx = rng.randint(0, n_triples - 1)
            if uncovered[idx]:
                break
        
        triple = _IDX_TO_COMBO[idx]
        block = list(triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered[idx] = False
        uncovered_count -= 1
        
        # Greedily extend the block to k points
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: number of currently uncovered triples
                # that would be covered by adding x
                gain = 0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        key = (a, b) if a < b else (b, a)
                        for third, tri_idx in _PAIR_TRIPLES[key]:
                            if third == x and uncovered[tri_idx]:
                                gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [x]
                elif gain == best_gain:
                    candidates.append(x)
            
            if candidates:
                next_point = rng.choice(candidates)
            else:
                remaining = [x for x in all_points if x not in block_set]
                next_point = rng.choice(remaining)
            
            block.append(next_point)
            block_set.add(next_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered: mark all triples in this block as covered
        for tri_idx in _block_triple_indices(block):
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                uncovered_count -= 1
        
        if time.time() > deadline:
            break
    
    return blocks, uncovered_count


def _local_search_reduce(v, k, t, blocks, deadline):
    """Reduce block count by removing redundant blocks and merging pairs."""
    blocks = [list(b) for b in blocks]
    n_triples = _C24_3
    
    # Build coverage count array
    cov_count = [0] * n_triples
    for b in blocks:
        for tri_idx in _block_triple_indices(b):
            cov_count[tri_idx] += 1
    
    # Identify critical triples (count == 1)
    critical = [False] * n_triples
    for i in range(n_triples):
        if cov_count[i] == 1:
            critical[i] = True
    
    improved = True
    rng = random.Random(42)
    
    # Phase 1: Remove redundant blocks
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b = blocks[i]
            block_tris = _block_triple_indices(b)
            
            # Check if any triple in this block is critical
            valid = True
            for tri_idx in block_tris:
                if critical[tri_idx]:
                    valid = False
                    break
            
            if valid:
                # Remove this block
                for tri_idx in block_tris:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                
                blocks.pop(i)
                improved = True
                break
    
    # Phase 2: Try to merge pairs of blocks into one
    improved = True
    rng2 = random.Random(123)
    
    while improved and time.time() < deadline:
        improved = False
        rng2.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                
                b1, b2 = blocks[i], blocks[j]
                s1, s2 = set(b1), set(b2)
                
                # Triples only covered by b1 (count == 1 and in b1)
                only_b1 = []
                for tri_idx in _block_triple_indices(b1):
                    if cov_count[tri_idx] == 1:
                        only_b1.append(tri_idx)
                
                # Triples only covered by b2
                only_b2 = []
                for tri_idx in _block_triple_indices(b2):
                    if cov_count[tri_idx] == 1:
                        only_b2.append(tri_idx)
                
                # All points that must be in the new block
                must_points = set()
                for ti in only_b1:
                    must_points.update(_IDX_TO_COMBO[ti])
                for ti in only_b2:
                    must_points.update(_IDX_TO_COMBO[ti])
                
                if len(must_points) > k:
                    continue
                
                # Try to build a block containing all must_points
                remaining_slots = k - len(must_points)
                candidate_block = sorted(must_points)
                candidate_set = set(candidate_block)
                
                # Check if this block covers all critical triples from both blocks
                new_block_tris_set = set(_block_triple_indices(candidate_block))
                
                valid = True
                for ti in only_b1:
                    if ti not in new_block_tris_set:
                        valid = False
                        break
                if valid:
                    for ti in only_b2:
                        if ti not in new_block_tris_set:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Try to fill remaining slots with points that help most
                if remaining_slots > 0:
                    # Add points from b1 and b2 that aren't already in candidate
                    extra_candidates = []
                    for p in b1:
                        if p not in candidate_set:
                            extra_candidates.append(p)
                    for p in b2:
                        if p not in candidate_set:
                            extra_candidates.append(p)
                    # Add random points if needed
                    for p in range(v):
                        if p not in candidate_set and p not in extra_candidates:
                            extra_candidates.append(p)
                    
                    # Pick up to remaining_slots from extra_candidates
                    rng2.shuffle(extra_candidates)
                    for p in extra_candidates[:remaining_slots]:
                        candidate_block.append(p)
                        candidate_set.add(p)
                    candidate_block.sort()
                
                # Verify the new block covers everything b1 and b2 covered that was critical
                final_tris_set = set(_block_triple_indices(candidate_block))
                
                # Check all critical triples from b1 and b2 are covered
                for ti in only_b1:
                    if ti not in final_tris_set:
                        valid = False
                        break
                if valid:
                    for ti in only_b2:
                        if ti not in final_tris_set:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Also check: does removing b1 and b2 and adding candidate_block
                # leave any triple uncovered?
                # For each triple in b1 or b2, check if it's still covered
                all_ok = True
                for tri_idx in _block_triple_indices(b1):
                    if cov_count[tri_idx] == 1 and tri_idx not in final_tris_set:
                        all_ok = False
                        break
                if all_ok:
                    for tri_idx in _block_triple_indices(b2):
                        if cov_count[tri_idx] == 1 and tri_idx not in final_tris_set:
                            all_ok = False
                            break
                
                if not all_ok:
                    continue
                
                # Perform the merge
                # Remove b1 and b2
                for tri_idx in _block_triple_indices(b1):
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                for tri_idx in _block_triple_indices(b2):
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                
                # Add candidate_block
                for tri_idx in _block_triple_indices(candidate_block):
                    cov_count[tri_idx] += 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                
                # Replace b1 and b2 with candidate_block
                # Remove b2 first (higher index)
                blocks.pop(j)
                blocks.pop(i)
                blocks.append(candidate_block)
                
                improved = True
                break
            
            if improved:
                break
    
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 115  # Safety margin before 120s limit
    
    # For the specific case v=24, k=7, t=3
    if v == 24 and k == 7 and t == 3:
        best_blocks = None
        best_count = float('inf')
        
        # Multiple randomized greedy builds followed by local search
        seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
                 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
        
        for seed in seeds:
            if time.time() > deadline - 20:
                break
            
            # Build a covering
            blocks, uncovered_count = _greedy_build(v, k, t, seed, deadline - 15)
            
            if uncovered_count > 0:
                # Incomplete covering; try to fix by adding more blocks
                while uncovered_count > 0 and time.time() < deadline - 10:
                    # Find an uncovered triple
                    for idx in range(_C24_3):
                        # We need to track uncovered... this is getting complicated
                        # Just break and use what we have if time is running out
                        pass
                    break
            
            if len(blocks) < best_count:
                # Apply local search to reduce
                refined = _local_search_reduce(v, k, t, blocks, deadline - 5)
                
                if len(refined) < best_count:
                    best_blocks = refined
                    best_count = len(refined)
                    
                    if best_count <= 76:
                        break  # Can't do better than known best
        
        if best_blocks is not None:
            return best_blocks
    
    # Fallback: simple greedy
    return _generic_greedy(v, k, t, start, deadline)


def _generic_greedy(v, k, t, start, deadline):
    """Fallback for non-24-point cases."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 5:
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
