import itertools
import time
import random
import sys
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            if a == b: continue
            for c in range(v):
                if a == c or b == c: continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
    
    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]
    
    def greedy_cover(uncovered_mask=None, random_state=0, aggressive=False):
        rng = random.Random(random_state)
        if uncovered_mask is None:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            num_uncovered = num_triples
        else:
            uncovered = uncovered_mask.copy()
            num_uncovered = int(np.sum(uncovered))
            
        blocks = []
        points = list(range(v))
        
        # Precompute all pairs for each block size increment
        # We'll use a priority queue to select the next triple to cover
        # based on "criticality" - how few blocks can still cover it
        
        # For speed, we'll use a simpler approach: always pick the uncovered
        # triple with the most constrained completion (fewest candidate points)
        # but for now, keep the argmax approach and optimize the inner loop
        
        while num_uncovered > 0:
            # Find an uncovered triple - use the first one for speed
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break
                    
                block_pairs = list(itertools.combinations(block, 2))
                num_pairs = len(block_pairs)
                
                # Vectorized gain computation
                # For each candidate, count how many uncovered triples it would cover
                gains = np.zeros(len(candidates), dtype=np.int32)
                for ci, x in enumerate(candidates):
                    g = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            g += 1
                    gains[ci] = g
                
                # Pick the candidate with max gain, break ties randomly
                max_gain = int(np.max(gains))
                if max_gain <= 0:
                    # All candidates have 0 gain, pick randomly
                    best_point = rng.choice(candidates)
                else:
                    best_indices = [i for i, g in enumerate(gains) if g == max_gain]
                    best_idx = rng.choice(best_indices)
                    best_point = candidates[best_idx]
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered mask
                for (p1, p2) in block_pairs:
                    idx = get_triple_idx(best_point, p1, p2)
                    if uncovered[idx]:
                        uncovered[idx] = 0
                        num_uncovered -= 1
                        
            blocks.append(tuple(sorted(block)))
            
        return blocks, uncovered, num_uncovered
    
    def trim_blocks(blocks):
        if not blocks:
            return blocks
            
        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        
        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)
            
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active[i]]
    
    def local_search_trim(blocks, max_iters=50):
        """Try to remove blocks to reduce count while maintaining coverage."""
        if not blocks:
            return blocks
            
        current_blocks = [tuple(b) for b in blocks]
        
        def get_counts(blist):
            c = np.zeros(num_triples, dtype=np.int32)
            for b in blist:
                for triple in itertools.combinations(b, t):
                    c[triple_to_idx[triple]] += 1
            return c

        counts = get_counts(current_blocks)
        
        for _ in range(max_iters):
            improved = False
            n = len(current_blocks)
            for i in range(n):
                # Try removing block i by decrementing counts
                old_counts = counts.copy()
                for triple in itertools.combinations(current_blocks[i], t):
                    idx = triple_to_idx[triple]
                    counts[idx] -= 1
                
                if np.all(counts > 0):
                    # Successful removal
                    current_blocks.pop(i)
                    improved = True
                    break
                else:
                    # Restore counts
                    counts = old_counts
            
            if not improved:
                break
                
        return current_blocks
    
    def is_valid(blocks):
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                covered[idx] = 1
        return np.all(covered == 1)
    
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    
    while time.time() < deadline - 10:
        # Phase 1: Fresh greedy with multiple random seeds
        for seed_offset in range(3):
            if time.time() >= deadline - 10:
                break
            blocks, uncovered, num_uncovered = greedy_cover(random_state=seed_counter + seed_offset)
            trimmed = trim_blocks(blocks)
            
            if len(trimmed) < best_len:
                best_len = len(trimmed)
                best_blocks = list(trimmed)
        
        if best_len <= 69:
            break
            
        # Phase 2: Perturbation and repair
        if time.time() < deadline - 5:
            rng_pert = random.Random(seed_counter + 1000)
            
            # Strategy: Remove blocks with lowest essentiality
            if len(trimmed) > 5:
                counts = np.zeros(num_triples, dtype=np.int32)
                for block in trimmed:
                    for triple in itertools.combinations(block, t):
                        counts[triple_to_idx[triple]] += 1
                # Score blocks by how many uniquely-covered triples they have
                block_scores = []
                for i, block in enumerate(trimmed):
                    score = 0
                    for triple in itertools.combinations(block, t):
                        if counts[triple_to_idx[triple]] == 1:
                            score += 1
                    block_scores.append(score)
                # Remove blocks with lowest scores (most redundant)
                sorted_idx = sorted(range(len(trimmed)), key=lambda x: block_scores[x])
                num_remove = min(2, len(trimmed) - 1)
                remove_idx = set(sorted_idx[:num_remove])
                remaining = [b for i, b in enumerate(trimmed) if i not in remove_idx]
                
                covered_mask = np.zeros(num_triples, dtype=np.uint8)
                for block in remaining:
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        covered_mask[idx] = 1
                
                uncovered_mask = 1 - covered_mask
                num_uncovered = int(np.sum(uncovered_mask))
                
                if num_uncovered > 0:
                    repaired_blocks, _, _ = greedy_cover(uncovered_mask=uncovered_mask, random_state=seed_counter + 200)
                    combined = remaining + repaired_blocks
                else:
                    combined = remaining
                    
                trimmed_combined = trim_blocks(combined)
                trimmed_combined = local_search_trim(trimmed_combined, max_iters=30)
                
                if len(trimmed_combined) < best_len:
                    best_len = len(trimmed_combined)
                    best_blocks = list(trimmed_combined)
                    
        seed_counter += 1
    
    # Final polish: local search on best solution
    if best_blocks and time.time() < deadline - 2:
        best_blocks = local_search_trim(best_blocks, max_iters=200)
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
