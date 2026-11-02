import itertools
import time
import random
from collections import defaultdict

def _greedy_cover(v, k, t, rng):
    """
    Constructs a covering using a weighted greedy heuristic.
    Prioritizes covering triples that are harder to cover.
    """
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    
    # Pre-compute for each point, which triples it can be part of
    # This helps in weighting the greedy choice
    
    while uncovered:
        # Weighted selection: pick uncovered triple with higher "difficulty"
        # Difficulty = number of potential blocks that could cover it (lower = harder)
        # For simplicity, use a heuristic: triples with more "isolated" points are harder
        
        # For now, use random selection but with a bias
        # Pick a random uncovered triple
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        
        # Greedily add points to the block with weighted gain
        while len(block) < k:
            best_gain = -1
            best_points = []
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for a, b in itertools.combinations(block, 2):
                    triple = tuple(sorted((a, b, x)))
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                x = rng.choice(best_points)
            
            block.append(x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the list of blocks that cover it
    triple_cover_count = defaultdict(int)
    
    # Build the map
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
    
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for block in blocks:
            # Check if this block is essential
            is_essential = False
            for triple in itertools.combinations(block, t):
                if triple_cover_count[tuple(triple)] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
            else:
                # Remove this block
                changed = True
                # Update cover counts
                for triple in itertools.combinations(block, t):
                    triple_cover_count[tuple(triple)] -= 1
                # Do not add to new_blocks
        
        blocks = new_blocks
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run to have a baseline
    # Using a fixed seed for reproducibility of the first attempt
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # Optional: Vary the greedy strategy slightly
        # For now, just randomizing the choice among best gains is enough
        
        blocks = _greedy_cover(v, k, t, rng)
        
        # Check time before reduction, as it can be slower
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a known good value, we could stop
            # Lower bound for C(24,7,3) is 69. If we get 69, we can stop.
            if best_size <= 69:
                break
                
    return best_blocks
