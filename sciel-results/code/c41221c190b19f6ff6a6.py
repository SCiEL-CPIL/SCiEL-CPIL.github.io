"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
from collections import Counter


def _compute_block_triples(block, t):
    """Return set of t-tuples in the block."""
    return set(itertools.combinations(block, t))


def _greedy_cover(v, k, t, rng=None):
    """Randomized greedy covering with weighted scoring."""
    if rng is None:
        rng = random.Random(42)
    
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    # Precompute: for each point, which triples contain it
    point_triples = {}
    for triple in all_triples:
        for p in triple:
            if p not in point_triples:
                point_triples[p] = []
            point_triples[p].append(triple)
    
    while uncovered:
        # Pick a random uncovered triple to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        block_set = set(block)
        
        # Compute current coverage of this partial block
        # For each candidate point, compute how many new uncovered triples it covers
        while len(block) < k:
            best_point = None
            best_gain = -1
            best_score = -1
            
            # Precompute which triples in uncovered are "close" to current block
            # A triple is covered by adding x if it contains x and (t-1) elements from block
            # We need triples that have exactly (t-1) elements in block and x not in block
            
            # For efficiency, for each uncovered triple, check if it has exactly t-1 elements in block
            # and the remaining element is a candidate
            candidate_counts = Counter()
            for triple in uncovered:
                # Count how many elements of triple are in block
                in_block = sum(1 for x in triple if x in block_set)
                if in_block == t - 1:
                    # The element not in block is the one that would complete this triple
                    for x in triple:
                        if x not in block_set:
                            candidate_counts[x] += 1
                            break
            
            # Also consider triples that have fewer elements in block but could be covered
            # Actually, only triples with exactly t-1 in block can be completed by one new point
            
            if not candidate_counts:
                # No point can complete any triple, just pick randomly
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_point = rng.choice(candidates)
                else:
                    break
            else:
                # Score: primary is coverage count, secondary is random tie-break
                max_count = max(candidate_counts.values())
                top_candidates = [p for p, c in candidate_counts.items() if c == max_count]
                best_point = rng.choice(top_candidates)
            
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            # Fill remaining with random points
            remaining = [x for x in range(v) if x not in block_set]
            rng.shuffle(remaining)
            for x in remaining[:k - len(block)]:
                block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered triples
        for triple in _compute_block_triples(block, t):
            uncovered.discard(triple)
    
    return blocks


def _local_search(blocks, v, k, t, max_iters=500, rng=None):
    """Try to improve the covering by swapping points in blocks."""
    if rng is None:
        rng = random.Random(123)
    
    # Build coverage count for each triple
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {triple: i for i, triple in enumerate(all_triples)}
    
    # Coverage count: how many blocks cover each triple
    cover_count = np.zeros(len(all_triples), dtype=np.int32)
    for block in blocks:
        for triple in _compute_block_triples(block, t):
            cover_count[triple_index[triple]] += 1
    
    # Find triples with cover_count == 0 (shouldn't exist if valid)
    # Try to reduce blocks or improve structure
    
    best_blocks = [list(b) for b in blocks]
    best_len = len(blocks)
    
    for _ in range(max_iters):
        # Pick a random block
        bi = rng.randrange(len(blocks))
        block = list(blocks[bi])
        
        # Try swapping one point in this block with a point not in it
        for pos in range(k):
            old_point = block[pos]
            # Try replacing with a point not in block
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                continue
            new_point = rng.choice(candidates)
            
            # Compute effect: which triples lose coverage, which gain
            # Triples that contain old_point and (t-1) other points from block
            lost = []
            gained = []
            
            for triple in _compute_block_triples(block, t):
                if old_point in triple:
                    lost.append(triple)
            
            new_block = block.copy()
            new_block[pos] = new_point
            new_block.sort()
            
            for triple in _compute_block_triples(new_block, t):
                if new_point in triple:
                    gained.append(triple)
            
            # Check if this swap is safe: no triple goes from count 1 to 0
            safe = True
            for triple in lost:
                idx = triple_index[triple]
                if cover_count[idx] <= 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Accept if it doesn't increase uncovered triples (shouldn't if safe)
            # Apply swap
            for triple in lost:
                cover_count[triple_index[triple]] -= 1
            for triple in gained:
                cover_count[triple_index[triple]] += 1
            
            blocks[bi] = new_block
            break  # Only one swap per iteration
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    # Run multiple randomized greedy attempts and keep the best
    best_blocks = None
    best_len = float('inf')
    
    # Use multiple seeds for diversity
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        rng = random.Random(seed)
        blocks = _greedy_cover(v, k, t, rng=rng)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
        best_len = len(blocks)
    
    # Try local search to potentially improve
    rng = random.Random(999)
    improved = _local_search(best_blocks, v, k, t, max_iters=200, rng=rng)
    
    # Verify the improved version is still valid
    all_triples = set(itertools.combinations(range(v), t))
    covered = set()
    for block in improved:
        for triple in itertools.combinations(block, t):
            covered.add(triple)
    
    if all_triples.issubset(covered):
        best_blocks = improved
        best_len = len(best_blocks)
    
    return best_blocks
# EVOLVE-BLOCK-END
