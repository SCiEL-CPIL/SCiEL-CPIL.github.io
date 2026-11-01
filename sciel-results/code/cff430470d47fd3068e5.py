import itertools
import time
import random
from collections import defaultdict

def _precompute_triples(v, t):
    """Pre-compute all t-subsets as sorted tuples and map to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triples.append(combo)
        triple_to_idx[combo] = idx
    return triples, triple_to_idx

def _precompute_point_triples(v, t, triple_to_idx):
    """For each point x, precompute list of triple indices containing x."""
    point_triples = [[] for _ in range(v)]
    for tri_idx, tri in enumerate(triple_to_idx):
        for x in tri:
            point_triples[x].append(tri_idx)
    return point_triples

def _precompute_pair_triples(v, t, triple_to_idx):
    """For each pair (a,b), precompute list of (third_point, triple_idx)."""
    pair_triples = {}
    for a in range(v):
        for b in range(a + 1, v):
            lst = []
            for c in range(v):
                if c == a or c == b:
                    continue
                tri = tuple(sorted((a, b, c)))
                if tri in triple_to_idx:
                    lst.append((c, triple_to_idx[tri]))
            pair_triples[(a, b)] = lst
    return pair_triples

def _greedy_bitset(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples, point_triples):
    """Fast greedy using bitset representation of uncovered triples."""
    rng = random.Random(seed)
    n_triples = len(triples)
    
    # Use Python int as bitset: bit i is 1 if triple i is uncovered
    uncovered = (1 << n_triples) - 1
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple
        # Find a random set bit
        while True:
            idx = rng.randint(0, n_triples - 1)
            if (uncovered >> idx) & 1:
                break
        
        triple = triples[idx]
        block = list(triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered &= ~(1 << idx)
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: count uncovered triples that x would cover
                # A triple (a,b,c) is covered by adding x if:
                # - x is one of a,b,c
                # - the other two points are in block
                # - the triple is currently uncovered
                gain = 0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        key = (a, b) if a < b else (b, a)
                        for third, tri_idx in pair_triples[key]:
                            if third == x and ((uncovered >> tri_idx) & 1):
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
        for tri in itertools.combinations(block, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            uncovered &= ~(1 << tri_idx)
        
        if uncovered == 0:
            break
    
    return blocks

def _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline):
    """Try to remove blocks if the covering remains valid, using coverage counts."""
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    # Build coverage count array
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
            
    # Identify critical triples (count == 1)
    critical_triples = set()
    for i in range(n_triples):
        if cov_count[i] == 1:
            critical_triples.add(i)

    improved = True
    rng = random.Random(42)
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            b = blocks[i]
            block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            
            # Check if any triple in this block is critical (count 1)
            valid = True
            for tri_idx in block_triples:
                if tri_idx in critical_triples:
                    valid = False
                    break
            
            if valid:
                for tri_idx in block_triples:
                    cov_count[tri_idx] -= 1
                    if cov_count[tri_idx] == 1:
                        critical_triples.add(tri_idx)
                    elif cov_count[tri_idx] == 2:
                        critical_triples.discard(tri_idx)
                
                blocks.pop(i)
                improved = True
                break
    
    return blocks

def _local_search_replace(v, k, t, blocks, triples, triple_to_idx, pair_triples, deadline):
    """Try to replace pairs of blocks with a single block to reduce count."""
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    
    critical_triples = set(i for i in range(n_triples) if cov_count[i] == 1)
    
    improved = True
    rng = random.Random(123)
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                
                b1, b2 = blocks[i], blocks[j]
                s1, s2 = set(b1), set(b2)
                
                # Triples only covered by b1 or b2
                only_b1 = []
                for tri in itertools.combinations(b1, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] == 1:
                        only_b1.append(ti)
                
                only_b2 = []
                for tri in itertools.combinations(b2, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] == 1:
                        only_b2.append(ti)
                
                # All points that must be in the new block
                must_points = set()
                for ti in only_b1:
                    must_points.update(triples[ti])
                for ti in only_b2:
                    must_points.update(triples[ti])
                
                if len(must_points) > k:
                    continue
                
                # Try to build a block containing all must_points
                remaining_slots = k - len(must_points)
                candidate_block = sorted(must_points)
                candidate_set = set(candidate_block)
                
                # Check if this block covers all critical triples from both blocks
                new_block_triples = set()
                for tri in itertools.combinations(candidate_block, t):
                    new_block_triples.add(tuple(sorted(tri)))
                
                valid = True
                for ti in only_b1:
                    if triples[ti] not in new_block_triples:
                        valid = False
                        break
                if valid:
                    for ti in only_b2:
                        if triples[ti] not in new_block_triples:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Remove b1 and b2, add new block
                for tri in itertools.combinations(b1, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                    if cov_count[ti] == 1:
                        critical_triples.add(ti)
                    elif cov_count[ti] == 2:
                        critical_triples.discard(ti)
                
                for tri in itertools.combinations(b2, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                    if cov_count[ti] == 1:
                        critical_triples.add(ti)
                    elif cov_count[ti] == 2:
                        critical_triples.discard(ti)
                
                # Verify no triple has count 0
                if any(cov_count[ti] == 0 for ti in range(n_triples)):
                    # Rollback
                    for tri in itertools.combinations(b1, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        cov_count[ti] += 1
                        if cov_count[ti] == 2:
                            critical_triples.add(ti)
                        elif cov_count[ti] == 1:
                            critical_triples.discard(ti)
                    for tri in itertools.combinations(b2, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        cov_count[ti] += 1
                        if cov_count[ti] == 2:
                            critical_triples.add(ti)
                        elif cov_count[ti] == 1:
                            critical_triples.discard(ti)
                    continue
                
                # Add new block's triples
                for tri in itertools.combinations(candidate_block, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] += 1
                    if cov_count[ti] == 2:
                        critical_triples.discard(ti)
                
                # Replace
                blocks.pop(j)
                blocks.pop(i)
                blocks.append(candidate_block)
                improved = True
                break
        
        # After replacement, try removals again
        if improved:
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() >= deadline:
                    break
                b = blocks[i]
                block_triples = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
                valid = all(cov_count[ti] > 1 for ti in block_triples)
                if valid:
                    for ti in block_triples:
                        cov_count[ti] -= 1
                        if cov_count[ti] == 1:
                            critical_triples.add(ti)
                        elif cov_count[ti] == 2:
                            critical_triples.discard(ti)
                    blocks.pop(i)
                    improved = True
    
    return blocks

def _swap_local_search(v, k, t, blocks, triples, triple_to_idx, deadline):
    """Try to swap points between blocks to improve coverage efficiency."""
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            cov_count[tri_idx] += 1
    
    improved = True
    rng = random.Random(789)
    
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(blocks)
        
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                
                b1, b2 = blocks[i], blocks[j]
                s1, s2 = set(b1), set(b2)
                
                # Try swapping one point from b1 with one from b2
                for p1 in b1:
                    for p2 in b2:
                        if p1 == p2:
                            continue
                        
                        # Compute what triples would change
                        # Triples lost from b1: those containing p1
                        # Triples gained in b1: those containing p2 but not p1
                        # Triples lost from b2: those containing p2
                        # Triples gained in b2: those containing p1 but not p2
                        
                        new_b1 = [x for x in b1 if x != p1] + [p2]
                        new_b2 = [x for x in b2 if x != p2] + [p1]
                        
                        # Check if this swap improves things
                        # Count how many triples become uncovered
                        # and how many become over-covered
                        gain = 0
                        loss = 0
                        
                        for tri in itertools.combinations(b1, t):
                            ti = triple_to_idx[tuple(sorted(tri))]
                            if p1 in tri:
                                # This triple is lost from b1
                                if cov_count[ti] == 1:
                                    loss += 1
                            # Check if p2 is in tri (it shouldn't be if p2 not in b1)
                        
                        for tri in itertools.combinations(b2, t):
                            ti = triple_to_idx[tuple(sorted(tri))]
                            if p2 in tri:
                                # This triple is lost from b2
                                if cov_count[ti] == 1:
                                    loss += 1
                        
                        # Triples gained
                        for tri in itertools.combinations(new_b1, t):
                            ti = triple_to_idx[tuple(sorted(tri))]
                            if p2 in tri and p1 not in tri:
                                # New triple in b1
                                if cov_count[ti] == 0:
                                    gain += 1
                        
                        for tri in itertools.combinations(new_b2, t):
                            ti = triple_to_idx[tuple(sorted(tri))]
                            if p1 in tri and p2 not in tri:
                                # New triple in b2
                                if cov_count[ti] == 0:
                                    gain += 1
                        
                        # Only accept if we don't create uncovered triples
                        # and ideally reduce redundancy
                        if loss == 0:
                            # Apply the swap
                            for tri in itertools.combinations(b1, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] -= 1
                            for tri in itertools.combinations(b2, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] -= 1
                            
                            for tri in itertools.combinations(new_b1, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] += 1
                            for tri in itertools.combinations(new_b2, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] += 1
                            
                            blocks[i] = sorted(new_b1)
                            blocks[j] = sorted(new_b2)
                            improved = True
                            break
                        elif gain > loss * 2:
                            # Accept if significant gain
                            for tri in itertools.combinations(b1, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] -= 1
                            for tri in itertools.combinations(b2, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] -= 1
                            
                            for tri in itertools.combinations(new_b1, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] += 1
                            for tri in itertools.combinations(new_b2, t):
                                ti = triple_to_idx[tuple(sorted(tri))]
                                cov_count[ti] += 1
                            
                            blocks[i] = sorted(new_b1)
                            blocks[j] = sorted(new_b2)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Pre-compute
    triples, triple_to_idx = _precompute_triples(v, t)
    all_points = list(range(v))
    pair_triples = _precompute_pair_triples(v, t, triple_to_idx)
    point_triples = _precompute_point_triples(v, t, triple_to_idx)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        if time.time() + 3.0 > deadline:
            break
        
        blocks = _greedy_bitset(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples, point_triples)
        
        # Apply local search if we have time
        if time.time() + 10.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Try replacement local search if we have more time
        if time.time() + 15.0 < deadline:
            blocks = _local_search_replace(v, k, t, blocks, triples, triple_to_idx, pair_triples, deadline)
        
        # Try swap local search if we have more time
        if time.time() + 10.0 < deadline:
            blocks = _swap_local_search(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Final removal pass
        if time.time() + 5.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        if best_len == 69:
            break
        
        seed += 1
    
    return best_blocks
