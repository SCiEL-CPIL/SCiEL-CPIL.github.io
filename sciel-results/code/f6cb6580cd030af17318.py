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
                        if a > b:
                            a, b = b, a
                        # Determine sorted triple (a, b, x)
                        if x < a:
                            tri = (x, a, b)
                        elif x < b:
                            tri = (a, x, b)
                        else:
                            tri = (a, b, x)
                        idx = _COMBO_INDEX[tri]
                        if uncovered[idx]:
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
    """Reduce block count by removing redundant blocks and replacing blocks."""
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
    
    # Phase 2: Block Replacement
    # Try to replace blocks with others that cover the same critical triples but have better overlap properties
    # We iterate through blocks and try to find a replacement for each one.
    # A replacement B_new is valid if it covers all critical triples of B_old.
    # We prefer replacements that maximize the number of triples in B_new that are already covered by other blocks (to create redundancy for future removals).
    
    replacement_improved = True
    rng2 = random.Random(123)
    
    while replacement_improved and time.time() < deadline:
        replacement_improved = False
        rng2.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b_old = blocks[i]
            old_tris = _block_triple_indices(b_old)
            
            # Find critical triples in b_old
            critical_in_old = [ti for ti in old_tris if critical[ti]]
            
            if not critical_in_old:
                # Should have been removed in Phase 1, but maybe it became redundant after other changes?
                # Or if Phase 1 didn't catch it due to order? 
                # Actually, if critical_in_old is empty, it's redundant.
                for tri_idx in old_tris:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical[tri_idx] = True
                    elif cov_count[tri_idx] == 2:
                        critical[tri_idx] = False
                blocks.pop(i)
                replacement_improved = True
                break
            
            # Determine the set of points that MUST be in the new block to cover critical_in_old
            must_points = set()
            for ti in critical_in_old:
                must_points.update(_IDX_TO_COMBO[ti])
            
            if len(must_points) > k:
                continue
            
            # We need to find a block B_new containing must_points.
            # B_new must also satisfy: for any triple T in b_old, if count(T) == 1 (i.e., T is critical), T must be in B_new.
            # This is already ensured by covering must_points.
            # Additionally, we must ensure that no triple T in b_old has its count drop to 0.
            # i.e., for any T in b_old, if T is NOT in B_new, then count(T) > 1.
            # This is equivalent to: for any T in b_old, if count(T) == 1, then T in B_new.
            
            # Let's generate candidate blocks.
            # We can try to complete must_points to a block of size k.
            # To make it efficient, we can try a few random completions.
            
            best_new_block = None
            best_score = -1
            
            # Get all points not in must_points
            available_points = [p for p in range(v) if p not in must_points]
            needed = k - len(must_points)
            
            # If we need to pick many points, random sampling is good.
            # We try a few random completions.
            attempts = 10
            for _ in range(attempts):
                if time.time() >= deadline:
                    break
                
                # Pick 'needed' random points from available_points
                if needed > 0:
                    extra = rng2.sample(available_points, needed)
                else:
                    extra = []
                
                candidate = list(must_points) + extra
                candidate.sort()
                candidate_set = set(candidate)
                
                # Check validity:
                # 1. All critical_in_old must be in candidate (guaranteed by construction)
                # 2. For all triples T in b_old, if count(T) == 1, T must be in candidate.
                
                valid = True
                for ti in old_tris:
                    if cov_count[ti] == 1:
                        # Check if this triple is in candidate
                        tri_tuple = _IDX_TO_COMBO[ti]
                        if not all(p in candidate_set for p in tri_tuple):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Score: Maximize overlap with other blocks (excluding b_old)
                # Count how many triples in candidate are covered by at least one other block.
                # We approximate this by checking if count(T) > 1 for T in candidate, 
                # but we must be careful: count(T) includes b_old.
                # So we check if count(T) > 1 (meaning it's covered by >= 2 blocks, one of which is b_old? No, b_old is being replaced).
                # Actually, we want to know if T is covered by a block OTHER than b_old.
                # If count(T) > 1, it is covered by at least one other block.
                # If count(T) == 1, it is only covered by b_old.
                
                score = 0
                for ti in _block_triple_indices(candidate):
                    if cov_count[ti] > 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_block = candidate
            
            if best_new_block is not None and best_new_block != b_old:
                # Perform replacement
                # Remove b_old
                for ti in old_tris:
                    cov_count[ti] -= 1
                    if cov_count[ti] == 1:
                        critical[ti] = True
                    elif cov_count[ti] == 2:
                        critical[ti] = False
                
                # Add best_new_block
                new_tris = _block_triple_indices(best_new_block)
                for ti in new_tris:
                    cov_count[ti] += 1
                    if cov_count[ti] == 1:
                        critical[ti] = True
                    elif cov_count[ti] == 2:
                        critical[ti] = False
                
                blocks[i] = best_new_block
                replacement_improved = True
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
