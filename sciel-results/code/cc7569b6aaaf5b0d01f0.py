import itertools
import random
import time
import sys
import math

_PRECOMP_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[key]
        
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = []
    block_triples = []
    blocks_covering_triple = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block)
        
        block_triple_indices = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            block_triple_indices.append(idx)
            blocks_covering_triple[idx].append(block_id)
            
        block_triples.append(block_triple_indices)
        
    _PRECOMP_CACHE[key] = (all_blocks, block_triples, blocks_covering_triple, num_triples)
    return _PRECOMP_CACHE[key]

def _frequency_based_greedy(v, k, t, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, deadline):
    """
    Frequency-based unbiased greedy: weight each candidate block by the product
    of inverse frequencies of its constituent points. This balances point load.
    """
    uncovered = [1] * num_triples  # 1 if uncovered, 0 if covered
    uncovered_count = num_triples
    selected_blocks = []
    used_blocks = set()
    point_freq = [0] * v
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick a random uncovered triple
        while True:
            chosen_triple = rng.randrange(num_triples)
            if uncovered[chosen_triple] == 1:
                break
        
        candidates = blocks_covering_triple[chosen_triple]
        
        best_block_idx = -1
        best_score = -1.0
        
        for b_idx in candidates:
            if b_idx in used_blocks:
                continue
            
            # Count how many currently uncovered triples this block covers
            cover_count = 0
            for t_idx in block_triples[b_idx]:
                if uncovered[t_idx] == 1:
                    cover_count += 1
            
            if cover_count == 0:
                continue
            
            # Frequency-based weight: product of inverse (1 + freq) for each point
            # This penalizes blocks that use already-frequently-used points
            freq_weight = 1.0
            for p in all_blocks[b_idx]:
                freq_weight *= 1.0 / (1.0 + point_freq[p])
            
            # Combined score: marginal gain * frequency balance
            score = cover_count * freq_weight
            
            if score > best_score:
                best_score = score
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            # Fallback: just pick any unused block covering chosen_triple
            for b_idx in candidates:
                if b_idx not in used_blocks:
                    best_block_idx = b_idx
                    break
            if best_block_idx == -1:
                break
        
        selected_blocks.append(best_block_idx)
        used_blocks.add(best_block_idx)
        
        # Update point frequencies
        for p in all_blocks[best_block_idx]:
            point_freq[p] += 1
        
        # Update coverage
        for t_idx in block_triples[best_block_idx]:
            if uncovered[t_idx] == 1:
                uncovered[t_idx] = 0
                uncovered_count -= 1
    
    return selected_blocks

def _local_search_refine(v, k, t, selected_blocks, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, deadline):
    """
    Local search: try to remove redundant blocks, then try swap-based improvements.
    """
    selected_blocks = selected_blocks[:]
    
    # Phase 1: Remove redundant blocks (iteratively)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute coverage count for each triple
        coverage = [0] * num_triples
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                coverage[t_idx] += 1
        
        # Find removable blocks
        to_remove = []
        for i, b_idx in enumerate(selected_blocks):
            essential = False
            for t_idx in block_triples[b_idx]:
                if coverage[t_idx] == 1:
                    essential = True
                    break
            if not essential:
                to_remove.append(i)
        
        if to_remove:
            for i in reversed(to_remove):
                selected_blocks.pop(i)
            improved = True
    
    # Phase 2: Try to replace pairs of blocks with a single block
    if time.time() < deadline and len(selected_blocks) > 2:
        n = len(selected_blocks)
        # Sample a subset of pairs to try (don't do all O(n^2) if n is large)
        max_pairs = 500
        pairs_tried = 0
        
        # Shuffle to get diverse pairs
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in range(min(n, 30)):
            if pairs_tried >= max_pairs or time.time() >= deadline:
                break
            for j in range(i + 1, min(n, i + 30)):
                if pairs_tried >= max_pairs or time.time() >= deadline:
                    break
                pairs_tried += 1
                
                # Try removing blocks i and j, see if we can add one block to cover the gap
                # This is expensive, so only do it occasionally
                if pairs_tried % 50 != 0:
                    continue
                
                # Compute what would be uncovered if we remove blocks i and j
                removed = [selected_blocks[i], selected_blocks[j]]
                remaining = [b for idx, b in enumerate(selected_blocks) if idx != i and idx != j]
                
                # Coverage of remaining blocks
                coverage = [0] * num_triples
                for b_idx in remaining:
                    for t_idx in block_triples[b_idx]:
                        coverage[t_idx] += 1
                
                # Find uncovered triples
                uncovered_triples = []
                for t_idx in range(num_triples):
                    if coverage[t_idx] == 0:
                        uncovered_triples.append(t_idx)
                
                if not uncovered_triples:
                    # Great! We can remove both blocks
                    selected_blocks = remaining
                    break
                elif len(uncovered_triples) <= 10:
                    # Try to find a single block that covers all uncovered triples
                    # This is a set cover problem on the uncovered triples
                    best_single = -1
                    best_covered = -1
                    for b_idx in range(len(all_blocks)):
                        if b_idx in removed:
                            continue
                        cnt = 0
                        for t_idx in uncovered_triples:
                            # Check if block b_idx covers triple t_idx
                            # We need to check if all elements of triple t_idx are in block b_idx
                            pass  # This is too slow, skip for now
                    
                    # Skip this phase for now as it's too slow
    
    return selected_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_blocks, block_triples, blocks_covering_triple, num_triples = _get_precomputed(v, k, t)
    
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 0
    
    while time.time() < deadline:
        rng = random.Random()
        
        # Use frequency-based greedy
        selected_blocks = _frequency_based_greedy(v, k, t, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, deadline)
        
        # Local search refinement
        if time.time() < deadline:
            selected_blocks = _local_search_refine(v, k, t, selected_blocks, all_blocks, block_triples, blocks_covering_triple, num_triples, rng, deadline)
        
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks[:]
            
            if best_size <= 69:
                break
        
        num_restarts += 1
        
        # Allow more restarts if we're fast
        if num_restarts > 50 and time.time() > start_time + 60:
            break
    
    if best_solution is None:
        return []
        
    return [list(all_blocks[b_idx]) for b_idx in best_solution]
