import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _precompute_triples(v, t):
    """Pre-compute all t-subsets as sorted tuples and map to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triples.append(combo)
        triple_to_idx[combo] = idx
    return triples, triple_to_idx

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

def _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples):
    """Fast greedy using precomputed pair-triple lookup."""
    rng = random.Random(seed)
    n_triples = len(triples)
    
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    
    while uncovered_count > 0:
        # Pick a random uncovered triple
        while True:
            idx = rng.randint(0, n_triples - 1)
            if uncovered[idx]:
                break
        
        triple = triples[idx]
        block = list(triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered[idx] = False
        uncovered_count -= 1
        
        while len(block) < k:
            best_gain = -1
            candidates = []
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain using pair_triples lookup
                gain = 0
                n_block = len(block)
                for i in range(n_block):
                    for j in range(i + 1, n_block):
                        a, b = block[i], block[j]
                        key = (a, b) if a < b else (b, a)
                        for third, tri_idx in pair_triples[key]:
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
        for tri in itertools.combinations(block, t):
            tri_idx = triple_to_idx[tuple(sorted(tri))]
            if uncovered[tri_idx]:
                uncovered[tri_idx] = False
                uncovered_count -= 1
        
        if uncovered_count == 0:
            break
    
    return blocks

def _maxsat_local_search(v, k, t, blocks, triples, triple_to_idx, deadline, max_iters=5000):
    """WalkSAT-style local search to minimize block count while maintaining coverage.
    
    Formulation: Each block is a boolean variable (selected or not).
    Each triple constraint is a clause that must be satisfied (at least one selected block covers it).
    Objective: minimize number of selected blocks (soft objective).
    
    We use a smooth surrogate: score = sum(1/triple_coverage) + lambda * num_blocks
    where triple_coverage is the number of selected blocks covering each triple.
    """
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)
    n_triples = len(triples)
    
    if n_blocks <= 1:
        return blocks
    
    # Precompute which triples each block covers
    block_triples = []
    for b in blocks:
        tris = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
    
    # Build triple-to-blocks mapping
    triple_to_blocks = defaultdict(set)
    for bi, tris in enumerate(block_triples):
        for ti in tris:
            triple_to_blocks[ti].add(bi)
    
    # Initial state: all blocks selected
    selected = [True] * n_blocks
    cov_count = [0] * n_triples
    for bi in range(n_blocks):
        if selected[bi]:
            for ti in block_triples[bi]:
                cov_count[ti] += 1
    
    best_blocks = [list(blocks[i]) for i in range(n_blocks) if selected[i]]
    best_score = sum(1.0 / cov_count[ti] if cov_count[ti] > 0 else 1000.0 for ti in range(n_triples)) + 0.01 * len(best_blocks)
    
    rng = random.Random(777)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Compute current score
        current_score = sum(1.0 / cov_count[ti] if cov_count[ti] > 0 else 1000.0 for ti in range(n_triples)) + 0.01 * sum(selected)
        
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(blocks[i]) for i in range(n_blocks) if selected[i]]
        
        # If all triples covered and we've minimized, try to remove more blocks
        all_covered = all(cov_count[ti] > 0 for ti in range(n_triples))
        
        if not all_covered:
            # Pick a random uncovered triple and flip a block that covers it
            uncovered_triples = [ti for ti in range(n_triples) if cov_count[ti] == 0]
            if not uncovered_triples:
                break
            ti = rng.choice(uncovered_triples)
            covering_blocks = [bi for bi in triple_to_blocks[ti] if not selected[bi]]
            if covering_blocks:
                bi = rng.choice(covering_blocks)
                selected[bi] = True
                for tj in block_triples[bi]:
                    cov_count[tj] += 1
            continue
        
        # Try to flip a random block
        # Weight toward removing blocks (to reduce count)
        if rng.random() < 0.6:
            # Try to remove a block
            removable = [bi for bi in range(n_blocks) if selected[bi] and all(cov_count[tj] > 1 for tj in block_triples[bi])]
            if removable:
                bi = rng.choice(removable)
                selected[bi] = False
                for tj in block_triples[bi]:
                    cov_count[tj] -= 1
            else:
                # Can't remove any, try adding
                addable = [bi for bi in range(n_blocks) if not selected[bi]]
                if addable:
                    bi = rng.choice(addable)
                    selected[bi] = True
                    for tj in block_triples[bi]:
                        cov_count[tj] += 1
        else:
            # Try to add a block (to improve coverage redundancy)
            addable = [bi for bi in range(n_blocks) if not selected[bi]]
            if addable:
                bi = rng.choice(addable)
                selected[bi] = True
                for tj in block_triples[bi]:
                    cov_count[tj] += 1
            else:
                # All selected, try removing
                removable = [bi for bi in range(n_blocks) if selected[bi] and all(cov_count[tj] > 1 for tj in block_triples[bi])]
                if removable:
                    bi = rng.choice(removable)
                    selected[bi] = False
                    for tj in block_triples[bi]:
                        cov_count[tj] -= 1
    
    return best_blocks

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
                
                # Check that all non-critical triples we'd lose are covered elsewhere
                # Actually, we just need to verify the final covering is valid
                # For now, accept if critical triples are covered
                
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

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Pre-compute
    triples, triple_to_idx = _precompute_triples(v, t)
    all_points = list(range(v))
    pair_triples = _precompute_pair_triples(v, t, triple_to_idx)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        if time.time() + 5.0 > deadline:
            break
        
        blocks = _greedy_fast(v, k, t, seed, triples, triple_to_idx, all_points, pair_triples)
        
        # Apply local search if we have time
        if time.time() + 15.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Try replacement local search if we have more time
        if time.time() + 20.0 < deadline:
            blocks = _local_search_replace(v, k, t, blocks, triples, triple_to_idx, pair_triples, deadline)
        
        # Try Max-SAT-style local search
        if time.time() + 10.0 < deadline:
            blocks_msat = _maxsat_local_search(v, k, t, blocks, triples, triple_to_idx, deadline, max_iters=3000)
            if len(blocks_msat) < len(blocks):
                blocks = blocks_msat
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        if best_len == 69:
            break
        
        seed += 1
    
    return best_blocks
