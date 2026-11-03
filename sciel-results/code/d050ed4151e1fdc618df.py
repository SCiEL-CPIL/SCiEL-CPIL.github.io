"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys
from collections import Counter

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples
       to maximize the reduction in the entropy of the uncovered triple distribution.
    2. Local Search: Optimize the initial solution by removing redundant blocks
       and replacing blocks with more efficient ones.
    """
    all_triples = set(itertools.combinations(range(v), t))
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples)
    
    # Phase 2: Local Search Optimization with incremental coverage counts
    blocks = _local_search(v, k, t, all_triples, blocks)
    
    # Phase 3: Final deduplication
    blocks = _deduplicate(blocks)
    
    return blocks

def _info_greedy(v, k, t, all_triples):
    """
    Greedy covering that maximizes entropy reduction.
    We approximate entropy reduction by weighting uncovered triples based on their
    'scarcity' or 'isolation'. Triples that share few points with other uncovered
    triples are 'isolated' and harder to cover efficiently in bulk, so we prioritize them.
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Pre-compute all triples as tuples for fast lookup
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    while uncovered:
        # Count point frequencies in uncovered triples
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Score each uncovered triple by rarity of its points
        # High score = contains rare points = high priority
        best_triple = None
        best_score = -1
        for triple in uncovered:
            score = 0.0
            for p in triple:
                score += 1.0 / (point_freq[p] + 1)
            if score > best_score:
                best_score = score
                best_triple = triple
        
        # Build a block around best_triple
        block = list(best_triple)
        
        # Extend the block by picking points that maximize the weighted gain
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            # Precompute pairs in current block for faster gain calculation
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0.0
                # Calculate weighted gain of adding x
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            # Weight this triple by rarity of its points
                            w = 0.0
                            for p in triple:
                                w += 1.0 / (point_freq[p] + 1)
                            gain += w
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _local_search(v, k, t, all_triples, blocks, max_iters=2000):
    """
    Local search to reduce the number of blocks.
    Uses incremental coverage counting for efficiency.
    """
    blocks = [list(b) for b in blocks]
    
    # Build coverage count: for each triple, how many blocks cover it
    triple_cover_count = Counter()
    for b in blocks:
        for triple in itertools.combinations(b, t):
            triple_cover_count[triple] += 1
    
    # Pass 1: Remove redundant blocks (those that don't uniquely cover any triple)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            # Check if removing block i still leaves all triples covered
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if triple_cover_count[triple] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for triple in itertools.combinations(blocks[i], t):
                    triple_cover_count[triple] -= 1
                blocks.pop(i)
                changed = True
                break
    
    # Pass 2: Try to replace blocks with more efficient ones
    random.seed(42)
    for _ in range(max_iters):
        if not blocks:
            break
        idx = random.randint(0, len(blocks) - 1)
        
        # Find triples covered ONLY by blocks[idx]
        must_cover = set()
        for triple in itertools.combinations(blocks[idx], t):
            if triple_cover_count[triple] == 1:
                must_cover.add(triple)
                
        if not must_cover:
            # Block is redundant, remove it
            for triple in itertools.combinations(blocks[idx], t):
                triple_cover_count[triple] -= 1
            blocks.pop(idx)
            continue
            
        # Try to construct a new block that covers must_cover
        if len(must_cover) > 10:
            continue
            
        candidate_points = set()
        for triple in must_cover:
            candidate_points.update(triple)
            
        if len(candidate_points) > k:
            continue
            
        new_block = list(candidate_points)
        remaining_must = set(must_cover)
        
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            new_block_set = set(new_block)
            for x in range(v):
                if x in new_block_set:
                    continue
                gain = 0
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        triple = tuple(sorted([new_block[i], new_block[j], x]))
                        if triple in remaining_must:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None or best_gain == 0:
                break
            new_block.append(best_point)
            
        new_block.sort()
        if len(new_block) == k and len(set(new_block)) == k:
            new_covered = set(itertools.combinations(new_block, t))
            if must_cover.issubset(new_covered):
                # Replace block
                old_block = blocks[idx]
                for triple in itertools.combinations(old_block, t):
                    triple_cover_count[triple] -= 1
                for triple in itertools.combinations(new_block, t):
                    triple_cover_count[triple] += 1
                blocks[idx] = new_block
                
    return blocks

def _deduplicate(blocks):
    """Remove duplicate blocks."""
    seen = set()
    unique_blocks = []
    for b in blocks:
        b_tuple = tuple(b)
        if b_tuple not in seen:
            seen.add(b_tuple)
            unique_blocks.append(b)
    return unique_blocks
