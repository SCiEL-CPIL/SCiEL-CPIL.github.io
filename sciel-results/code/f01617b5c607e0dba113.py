import itertools
import time
import random
from collections import defaultdict

def _greedy_cover(v, k, t, rng, deadline=None, randomize=False):
    """
    Constructs a covering using a greedy heuristic with optional randomization.
    If deadline is provided, stops early if time is exceeded.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
            
        # Pick a random uncovered triple to start the block (if randomizing)
        if randomize and rng.random() < 0.5:
            # Random choice from uncovered set
            # Converting to list is expensive, but for v=24, t=3, C(24,3)=2024, it's manageable
            # However, we can just pick a random element from the set if we don't need specific distribution
            # rng.choice requires a sequence. Let's use a faster method or just accept the cost for small sets.
            # Actually, for better performance, we can iterate if the set is small, but here it starts large.
            # Let's just pick min or max for deterministic, or random if we want diversity.
            # For now, let's stick to min for speed in non-random, and random sample for random.
            # To avoid list(uncovered) overhead, we can use a trick or just accept it.
            # C(24,3) is 2024. list() of 2024 tuples is fast.
            start_triple = rng.choice(list(uncovered))
        else:
            start_triple = min(uncovered)
            
        block = list(start_triple)
        
        # Greedily add points to the block
        while len(block) < k:
            if deadline is not None and time.time() > deadline:
                break
                
            # Calculate gain for each candidate point
            best_gain = -1
            best_points = []
            block_set = set(block)
            
            # Pre-calculate pairs in current block for faster gain calculation
            # Gain of x is number of (t-1)-subsets S in block such that S U {x} in uncovered
            # For t=3, this is pairs in block.
            
            # Iterate over candidates
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                # Check all (t-1) combinations in block
                # For t=3, itertools.combinations(block, 2)
                for rest in itertools.combinations(block, t - 1):
                    # Create the t-subset {x} U rest
                    # We need to check if this is in uncovered
                    # To make a hashable key, we sort it
                    # Since rest is sorted (from combinations) and x is a single element,
                    # we can insert x into rest efficiently or just sort the tuple.
                    # For small t, tuple(sorted(...)) is fine.
                    subset = tuple(sorted(rest + (x,)))
                    if subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_points = [x]
                elif gain == best_gain:
                    best_points.append(x)
            
            # If no gain, pick random point not in block
            if not best_points:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    x = rng.choice(candidates)
                else:
                    break
            else:
                # If randomizing, pick randomly from best points
                if randomize:
                    x = rng.choice(best_points)
                else:
                    x = best_points[0]
            
            block.append(x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            
            # Remove covered triples
            for triple in itertools.combinations(block, t):
                uncovered.discard(triple)
                
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t, deadline=None):
    """
    Attempts to remove redundant blocks from the covering.
    Returns the reduced list of blocks.
    """
    if not blocks:
        return blocks
    
    # Map each triple to the list of blocks that cover it
    # Or just count how many blocks cover each triple
    triple_cover_count = defaultdict(int)
    
    # Build the map
    for block in blocks:
        for triple in itertools.combinations(block, t):
            triple_cover_count[tuple(triple)] += 1
    
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_blocks = None
    best_size = float('inf')
    
    # Initial deterministic run to have a baseline
    rng = random.Random(42)
    blocks = _greedy_cover(v, k, t, rng, deadline, randomize=False)
    if blocks:
        blocks = _reduce_covering(blocks, v, k, t, deadline)
        best_blocks = blocks
        best_size = len(blocks)
    
    # Try multiple randomized runs
    seed_counter = 0
    # We want to spend most of the time searching, but ensure we have time for final reduction
    # Let's say we stop new searches at 130s, leaving 10s for final reduction of best
    search_deadline = start_time + 130
    
    while time.time() < search_deadline:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # Use randomized greedy
        blocks = _greedy_cover(v, k, t, rng, deadline, randomize=True)
        
        if not blocks:
            continue
            
        # Check time before reduction
        if time.time() > deadline:
            continue
            
        blocks = _reduce_covering(blocks, v, k, t, deadline)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a known good value, we could stop
            # Lower bound for C(24,7,3) is 69.
            if best_size <= 69:
                break
                
    # Final reduction on the best found blocks, using remaining time
    if best_blocks is not None:
        # Use the remaining time for a potentially more thorough reduction
        # The current _reduce_covering is already fairly aggressive (repeated passes)
        # We can just call it again with the final deadline
        best_blocks = _reduce_covering(best_blocks, v, k, t, deadline)
        
    return best_blocks
