import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Leave 10 seconds for safety
    deadline = start_time + 140.0
    
    # Precompute all triples and map to index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    best_blocks = None
    best_count = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None # Timeout
            
            # Pick a random uncovered triple to seed the block
            # This breaks symmetry and allows exploring different coverings
            start_trip = random.choice(tuple(uncovered))
            block = list(start_trip)
            
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                block_pairs = list(itertools.combinations(block, t - 1))
                candidates = [x for x in range(v) if x not in block]
                
                best_gain = -1
                best_points = []
                
                for x in candidates:
                    gain = 0
                    for p in block_pairs:
                        # Construct triple
                        # Optimization: avoid sorting if possible?
                        # t=3, p is a pair. x is int.
                        # We need the canonical form (sorted tuple)
                        trip = tuple(sorted((x,) + p))
                        if trip in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_points:
                    block.append(random.choice(best_points))
                else:
                    # Fallback if no gain (should not happen if uncovered is not empty)
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            for trip in itertools.combinations(block, t):
                uncovered.discard(trip)
        
        # Post-processing: Peeling to remove redundant blocks
        # Initialize coverage counts
        cover_count = [0] * num_triples
        for b in blocks:
            for trip in itertools.combinations(b, t):
                idx = triple_to_idx[trip]
                cover_count[idx] += 1
        
        # Iteratively remove blocks that are completely redundant
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if all triples in b are covered by others
                is_redundant = True
                for trip in itertools.combinations(b, t):
                    idx = triple_to_idx[trip]
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove this block
                    for trip in itertools.combinations(b, t):
                        idx = triple_to_idx[trip]
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        return blocks

    # Main loop: Run multiple randomized greedy instances
    seed = 1
    while time.time() < deadline:
        blocks = run_greedy(seed)
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        seed += 1
        
        # If we have a very good solution, we might want to spend time polishing it?
        # For now, just keep restarting.
        
    if best_blocks is None:
        # Fallback: Deterministic greedy (seed behavior)
        # This ensures we return *something* if randomization fails or times out early
        uncovered = set(all_triples)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            start_trip = next(iter(uncovered))
            block = list(start_trip)
            while len(block) < k:
                block_pairs = list(itertools.combinations(block, t - 1))
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for p in block_pairs:
                        trip = tuple(sorted((x,) + p))
                        if trip in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None: break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for trip in itertools.combinations(block, t):
                uncovered.discard(trip)
        best_blocks = blocks

    return best_blocks
