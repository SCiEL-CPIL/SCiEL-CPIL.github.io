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
            # Try to remove a block - prioritize blocks with high redundancy
            removable = []
            for bi in range(n_blocks):
                if selected[bi]:
                    min_cov = min(cov_count[tj] for tj in block_triples[bi])
                    if min_cov > 1:
                        removable.append((min_cov, bi))
            if removable:
                # Prefer blocks with highest minimum coverage (most redundant)
                removable.sort(reverse=True)
                # Sample from top candidates with some randomness
                top_n = min(10, len(removable))
                bi = removable[rng.randint(0, top_n - 1)][1]
                selected[bi] = False
                for tj in block_triples[bi]:
                    cov_count[tj] -= 1
            else:
                # Can't remove any, try adding a block that creates redundancy
                addable = []
                for bi in range(n_blocks):
                    if not selected[bi]:
                        # Count how many triples this block would make more redundant
                        redundancy_gain = 0
                        for tj in block_triples[bi]:
                            if cov_count[tj] == 1:
                                redundancy_gain += 1
                        addable.append((redundancy_gain, bi))
                if addable:
                    addable.sort(reverse=True)
                    top_n = min(10, len(addable))
                    bi = addable[rng.randint(0, top_n - 1)][1]
                    selected[bi] = True
                    for tj in block_triples[bi]:
                        cov_count[tj] += 1
        else:
            # Try to add a block (to improve coverage redundancy)
            addable = []
            for bi in range(n_blocks):
                if not selected[bi]:
                    redundancy_gain = 0
                    for tj in block_triples[bi]:
                        if cov_count[tj] == 1:
                            redundancy_gain += 1
                    addable.append((redundancy_gain, bi))
            if addable:
                addable.sort(reverse=True)
                top_n = min(10, len(addable))
                bi = addable[rng.randint(0, top_n - 1)][1]
                selected[bi] = True
                for tj in block_triples[bi]:
                    cov_count[tj] += 1
            else:
                # All selected, try removing most redundant block
                removable = []
                for bi in range(n_blocks):
                    if selected[bi]:
                        min_cov = min(cov_count[tj] for tj in block_triples[bi])
                        if min_cov > 1:
                            removable.append((min_cov, bi))
                if removable:
                    removable.sort(reverse=True)
                    top_n = min(10, len(removable))
                    bi = removable[rng.randint(0, top_n - 1)][1]
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

def _simulated_annealing(v, k, t, blocks, triples, triple_to_idx, deadline, max_iters=20000):
    """Simulated annealing on a pool of candidate blocks.
    
    State: binary vector indicating which blocks are selected.
    Energy: num_uncovered * large_penalty + num_selected_blocks.
    Moves: flip one block (add or remove), or swap two blocks.
    Accept worse moves with probability exp(-delta_E / T).
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
    
    # Build triple-to-blocks mapping for fast lookup
    triple_to_blocks = defaultdict(list)
    for bi, tris in enumerate(block_triples):
        for ti in tris:
            triple_to_blocks[ti].append(bi)
    
    # Initial state: all blocks selected
    selected = [True] * n_blocks
    cov_count = np.zeros(n_triples, dtype=np.int32)
    for bi in range(n_blocks):
        for ti in block_triples[bi]:
            cov_count[ti] += 1
    
    # Current energy
    num_uncovered = int(np.sum(cov_count == 0))
    num_selected = n_blocks
    current_energy = num_uncovered * 10000 + num_selected
    
    best_energy = current_energy
    best_selected = selected.copy()
    
    # Temperature schedule
    T_start = 80.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max(max_iters, 1))
    T = T_start
    
    rng = random.Random(999)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Choose move type: flip (70%) or swap (30%)
        move_type = rng.random()
        
        if move_type < 0.7:
            # Flip one block
            bi = rng.randint(0, n_blocks - 1)
            
            # Compute energy change
            if selected[bi]:
                # Removing block bi
                delta_uncovered = 0
                for ti in block_triples[bi]:
                    if cov_count[ti] == 1:
                        delta_uncovered += 1
                delta_selected = -1
            else:
                # Adding block bi
                delta_uncovered = 0
                for ti in block_triples[bi]:
                    if cov_count[ti] == 0:
                        delta_uncovered -= 1
                delta_selected = 1
            
            delta_energy = delta_uncovered * 10000 + delta_selected
            
            # Accept or reject
            if delta_energy <= 0:
                accept = True
            else:
                if T > T_end:
                    ratio = -delta_energy / T
                    if ratio > 50:
                        accept = False
                    elif ratio > -50:
                        accept = rng.random() < np.exp(ratio)
                    else:
                        accept = rng.random() < 0.5
                else:
                    accept = False
            
            if accept:
                selected[bi] = not selected[bi]
                if selected[bi]:
                    for ti in block_triples[bi]:
                        cov_count[ti] += 1
                else:
                    for ti in block_triples[bi]:
                        cov_count[ti] -= 1
                
                current_energy += delta_energy
                num_uncovered = int(np.sum(cov_count == 0))
                num_selected = int(np.sum(selected))
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()
        
        else:
            # Swap: remove one selected block, add one unselected block
            selected_indices = [i for i in range(n_blocks) if selected[i]]
            unselected_indices = [i for i in range(n_blocks) if not selected[i]]
            
            if selected_indices and unselected_indices:
                bi = rng.choice(selected_indices)
                bj = rng.choice(unselected_indices)
                
                # Compute energy change for removing bi and adding bj
                delta_uncovered = 0
                delta_selected = 0
                
                # Temporarily remove bi
                for ti in block_triples[bi]:
                    if cov_count[ti] == 1:
                        delta_uncovered += 1
                    cov_count[ti] -= 1
                
                # Add bj
                for ti in block_triples[bj]:
                    if cov_count[ti] == 0:
                        delta_uncovered -= 1
                    cov_count[ti] += 1
                
                delta_selected = 0  # net zero (remove one, add one)
                delta_energy = delta_uncovered * 10000 + delta_selected
                
                # Rollback
                for ti in block_triples[bj]:
                    cov_count[ti] -= 1
                for ti in block_triples[bi]:
                    cov_count[ti] += 1
                
                # Accept or reject
                if delta_energy <= 0:
                    accept = True
                else:
                    if T > T_end:
                        ratio = -delta_energy / T
                        if ratio > 50:
                            accept = False
                        elif ratio > -50:
                            accept = rng.random() < np.exp(ratio)
                        else:
                            accept = rng.random() < 0.5
                    else:
                        accept = False
                
                if accept:
                    selected[bi] = False
                    selected[bj] = True
                    for ti in block_triples[bi]:
                        cov_count[ti] -= 1
                    for ti in block_triples[bj]:
                        cov_count[ti] += 1
                    
                    current_energy += delta_energy
                    num_uncovered = int(np.sum(cov_count == 0))
                    num_selected = int(np.sum(selected))
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_selected = selected.copy()
        
        # Cool down
        T *= cooling
    
    # Extract best blocks
    best_blocks = [list(blocks[i]) for i in range(n_blocks) if best_selected[i]]
    
    # Verify and fix: if any triple is uncovered, add blocks greedily
    if best_blocks:
        final_cov = np.zeros(n_triples, dtype=np.int32)
        for b in best_blocks:
            for tri in itertools.combinations(b, t):
                ti = triple_to_idx[tuple(sorted(tri))]
                final_cov[ti] += 1
        
        uncovered_mask = final_cov == 0
        if np.any(uncovered_mask):
            # Add missing coverage from the full pool
            for bi in range(n_blocks):
                if not best_selected[bi]:
                    for ti in block_triples[bi]:
                        if final_cov[ti] == 0:
                            best_selected[bi] = True
                            for tj in block_triples[bi]:
                                final_cov[tj] += 1
                            break
    
    best_blocks = [list(blocks[i]) for i in range(n_blocks) if best_selected[i]]
    return best_blocks

def _aggressive_polish(v, k, t, blocks, triples, triple_to_idx, deadline):
    """Aggressive final polish: try removing blocks in random order, then try pairwise replacements."""
    blocks = [list(b) for b in blocks]
    n_triples = len(triples)
    rng = random.Random(555)
    
    # Build coverage counts
    cov_count = [0] * n_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            ti = triple_to_idx[tuple(sorted(tri))]
            cov_count[ti] += 1
    
    # Phase 1: Aggressive removal - try many random permutations
    for perm_iter in range(20):
        if time.time() >= deadline:
            break
        
        # Try removing blocks in random order
        order = list(range(len(blocks)))
        rng.shuffle(order)
        
        for i in order:
            if time.time() >= deadline:
                break
            if i >= len(blocks):
                continue
            
            b = blocks[i]
            block_tris = [triple_to_idx[tuple(sorted(tri))] for tri in itertools.combinations(b, t)]
            
            # Check if all triples in this block are covered by other blocks
            can_remove = True
            for ti in block_tris:
                if cov_count[ti] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for ti in block_tris:
                    cov_count[ti] -= 1
                blocks.pop(i)
    
    # Phase 2: Try to replace pairs of blocks with a single better block
    for replace_iter in range(10):
        if time.time() >= deadline:
            break
        
        improved = False
        for i in range(len(blocks)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                
                b1, b2 = blocks[i], blocks[j]
                s1, s2 = set(b1), set(b2)
                
                # Find triples uniquely covered by b1 or b2
                unique_to_b1 = []
                for tri in itertools.combinations(b1, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] == 1:
                        unique_to_b1.append(ti)
                
                unique_to_b2 = []
                for tri in itertools.combinations(b2, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] == 1:
                        unique_to_b2.append(ti)
                
                # Must include all points from unique triples
                must_pts = set()
                for ti in unique_to_b1 + unique_to_b2:
                    must_pts.update(triples[ti])
                
                if len(must_pts) > k:
                    continue
                
                # Try to complete the block
                base = sorted(must_pts)
                base_set = set(base)
                
                # Fill remaining slots with points that maximize coverage of uncovered triples
                remaining_slots = k - len(base)
                if remaining_slots <= 0:
                    candidate = base
                else:
                    # Pick points that cover the most currently-single-covered triples
                    candidate = list(base)
                    cand_set = set(base)
                    
                    for _ in range(remaining_slots):
                        best_pt = None
                        best_gain = -1
                        for pt in range(v):
                            if pt in cand_set:
                                continue
                            gain = 0
                            for a in candidate:
                                for b_pt in candidate:
                                    if a == b_pt:
                                        continue
                                    key = (a, b_pt) if a < b_pt else (b_pt, a)
                                    # Check if (a, b_pt, pt) forms a triple with count 1
                                    tri = tuple(sorted((a, b_pt, pt)))
                                    ti = triple_to_idx.get(tri)
                                    if ti is not None and cov_count[ti] == 1:
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_pt = pt
                        if best_pt is not None:
                            candidate.append(best_pt)
                            cand_set.add(best_pt)
                        else:
                            break
                    
                    candidate = sorted(candidate)
                
                # Verify the new block covers all required triples
                new_tris = set()
                for tri in itertools.combinations(candidate, t):
                    new_tris.add(tuple(sorted(tri)))
                
                valid = True
                for ti in unique_to_b1 + unique_to_b2:
                    if triples[ti] not in new_tris:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check that removing b1 and b2 doesn't uncover any triple
                would_uncover = False
                for tri in itertools.combinations(b1, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    if cov_count[ti] == 1 and triples[ti] not in new_tris:
                        would_uncover = True
                        break
                if not would_uncover:
                    for tri in itertools.combinations(b2, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        if cov_count[ti] == 1 and triples[ti] not in new_tris:
                            would_uncover = True
                            break
                
                if would_uncover:
                    continue
                
                # Perform the replacement
                for tri in itertools.combinations(b1, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                for tri in itertools.combinations(b2, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] -= 1
                for tri in itertools.combinations(candidate, t):
                    ti = triple_to_idx[tuple(sorted(tri))]
                    cov_count[ti] += 1
                
                # Check validity
                if any(cov_count[ti] == 0 for ti in range(n_triples)):
                    # Rollback
                    for tri in itertools.combinations(candidate, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        cov_count[ti] -= 1
                    for tri in itertools.combinations(b2, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        cov_count[ti] += 1
                    for tri in itertools.combinations(b1, t):
                        ti = triple_to_idx[tuple(sorted(tri))]
                        cov_count[ti] += 1
                    continue
                
                # Remove b1 and b2, add candidate
                blocks.pop(j)
                blocks.pop(i)
                blocks.append(candidate)
                improved = True
                break
        
        if not improved:
            break
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
        if time.time() + 10.0 < deadline:
            blocks = _local_search_fast(v, k, t, blocks, triples, triple_to_idx, deadline)
        
        # Try replacement local search if we have more time
        if time.time() + 15.0 < deadline:
            blocks = _local_search_replace(v, k, t, blocks, triples, triple_to_idx, pair_triples, deadline)
        
        # Try Max-SAT-style local search with more iterations
        if time.time() + 12.0 < deadline:
            blocks_msat = _maxsat_local_search(v, k, t, blocks, triples, triple_to_idx, deadline, max_iters=3000)
            if len(blocks_msat) < len(blocks):
                blocks = blocks_msat
        
        # Try simulated annealing with swap moves - this can escape local minima
        if time.time() + 18.0 < deadline:
            blocks_sa = _simulated_annealing(v, k, t, blocks, triples, triple_to_idx, deadline, max_iters=20000)
            if len(blocks_sa) < len(blocks):
                blocks = blocks_sa
        
        # New: aggressive polish pass
        if time.time() + 10.0 < deadline:
            blocks_polish = _aggressive_polish(v, k, t, blocks, triples, triple_to_idx, deadline)
            if len(blocks_polish) < len(blocks):
                blocks = blocks_polish
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        if best_len == 69:
            break
        
        seed += 1
    
    # Final polish pass: use remaining time to refine the best solution
    if best_blocks and time.time() + 8.0 < deadline:
        polish_deadline = time.time() + min(10.0, deadline - time.time() - 1.0)
        if polish_deadline > time.time() + 2.0:
            # Run final local search on best solution
            polished = _local_search_fast(v, k, t, best_blocks, triples, triple_to_idx, polish_deadline)
            if len(polished) < best_len:
                best_blocks = polished
                best_len = len(polished)
            
            # Try max-sat on polished
            if time.time() + 5.0 < polish_deadline:
                polished2 = _maxsat_local_search(v, k, t, best_blocks, triples, triple_to_idx, polish_deadline, max_iters=2000)
                if len(polished2) < best_len:
                    best_blocks = polished2
                    best_len = len(polished2)
            
            # Try aggressive polish on final best
            if time.time() + 5.0 < polish_deadline:
                polished3 = _aggressive_polish(v, k, t, best_blocks, triples, triple_to_idx, polish_deadline)
                if len(polished3) < best_len:
                    best_blocks = polished3
                    best_len = len(polished3)
    
    return best_blocks
