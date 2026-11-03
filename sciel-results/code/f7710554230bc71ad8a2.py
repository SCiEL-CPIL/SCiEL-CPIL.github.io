"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys

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
    
    # Phase 2: Local Search Optimization
    blocks = _local_search(v, k, t, all_triples, blocks)
    
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
    
    # Pre-compute point-triple incidence for faster weight calculation?
    # For v=24, t=3, total triples = 2024. This is small enough to recompute weights each step.
    
    while uncovered:
        # Calculate weights for all uncovered triples
        # Weight = 1 / (1 + number of other uncovered triples sharing at least one point)
        # This favors isolated triples.
        weights = {}
        # To speed this up, we can just pick the "best" block directly using a heuristic
        # that combines immediate gain and scarcity.
        
        # Heuristic: Pick the uncovered triple with the highest "scarcity score"
        # Scarcity score of a triple t is inversely proportional to the number of
        # uncovered triples that intersect t.
        
        # For performance, we don't calculate full weights for all triples every step.
        # Instead, we select a seed triple that is "critical".
        
        # A triple is critical if it has low connectivity to other uncovered triples.
        # Let's find the triple with the minimum intersection count with other uncovered triples.
        
        # Optimization: Instead of O(N^2) intersection check, we can use a simpler proxy:
        # Pick a random uncovered triple, or the one with the "rarest" points.
        
        # Let's use a dynamic weight approach:
        # 1. Count how many times each point appears in uncovered triples.
        # 2. Points with high frequency are "common". Points with low frequency are "rare".
        # 3. Triples containing "rare" points are harder to cover and should be prioritized.
        
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Score each uncovered triple by the sum of inverses of its points' frequencies.
        # High score = contains rare points = high priority.
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
        # Weighted gain = sum of weights of newly covered triples
        # We use the same point_freq-based weight for the gain calculation to be consistent.
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # Calculate weighted gain of adding x
                # New triples formed: {block[i], block[j], x}
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            # Weight this triple by rarity of its points
                            # Use current point_freq (slightly stale but good approximation)
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

def _local_search(v, k, t, all_triples, blocks, max_iters=5000):
    """
    Local search to reduce the number of blocks using coverage counts and swap operations.
    """
    blocks = list(blocks)
    
    def get_coverage_counts(block_list):
        counts = {}
        for b in block_list:
            for triple in itertools.combinations(b, t):
                counts[triple] = counts.get(triple, 0) + 1
        return counts

    # Pass 1: Remove redundant blocks efficiently
    changed = True
    while changed:
        changed = False
        counts = get_coverage_counts(blocks)
        for i in range(len(blocks)):
            # Check if any triple in blocks[i] has count 1
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                if counts[triple] == 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break

    # Pass 2: Swap-based local search
    # We try to swap a block with a new block that covers the same "unique" triples
    # but potentially allows for better overall coverage or simply is a valid replacement.
    # To be efficient, we only consider swaps if the current block has "unique" triples
    # that are few in number, making it easy to find a replacement.
    
    random.seed(42)
    for _ in range(max_iters):
        if not blocks:
            break
        idx = random.randint(0, len(blocks) - 1)
        current_block = blocks[idx]
        
        # Calculate coverage counts without the current block
        # This is O(N * C(k, t)) which is acceptable for N~80, k=7, t=3 (35 triples/block)
        counts = {}
        for j, b in enumerate(blocks):
            if j == idx:
                continue
            for triple in itertools.combinations(b, t):
                counts[triple] = counts.get(triple, 0) + 1
        
        # Find triples covered ONLY by current_block
        must_cover = set()
        for triple in itertools.combinations(current_block, t):
            if counts.get(triple, 0) == 0:
                must_cover.add(triple)
                
        if not must_cover:
            # Block is redundant, remove it
            blocks.pop(idx)
            continue
            
        # If must_cover is too large, skip
        if len(must_cover) > 15:
            continue
            
        # Try to find a new block that covers must_cover
        # Start with points from must_cover
        candidate_points = set()
        for triple in must_cover:
            candidate_points.update(triple)
            
        if len(candidate_points) > k:
            continue
            
        new_block = list(candidate_points)
        
        # Greedily extend new_block to cover must_cover and maximize additional coverage
        # We prioritize covering must_cover first.
        remaining_must = set(must_cover)
        
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            best_must_gain = -1
            
            for x in range(v):
                if x in new_block:
                    continue
                
                must_gain = 0
                total_gain = 0
                
                # Check coverage of must_cover
                for i in range(len(new_block)):
                    for j in range(i+1, len(new_block)):
                        triple = tuple(sorted([new_block[i], new_block[j], x]))
                        if triple in remaining_must:
                            must_gain += 1
                            total_gain += 1
                        # Also count general coverage (uncovered by others)
                        elif counts.get(triple, 0) == 0:
                            total_gain += 0.5 # Weight for additional coverage
                
                if must_gain > best_must_gain or (must_gain == best_must_gain and total_gain > best_gain):
                    best_must_gain = must_gain
                    best_gain = total_gain
                    best_point = x
            
            if best_point is None or best_must_gain == 0:
                # If we can't cover any more must_cover, break
                # But we must ensure all must_cover are covered before accepting
                break
            new_block.append(best_point)
            
        new_block.sort()
        if len(new_block) == k and len(set(new_block)) == k:
            new_covered = set(itertools.combinations(new_block, t))
            if must_cover.issubset(new_covered):
                # Check if this swap is beneficial or neutral
                # Ideally, we want to maintain the number of blocks or reduce it.
                # Since we are replacing 1 with 1, the count stays same.
                # However, the new block might cover more "other" triples,
                # potentially making other blocks redundant in the next pass.
                blocks[idx] = new_block
                
    return blocks
