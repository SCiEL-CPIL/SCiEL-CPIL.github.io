import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses a randomized greedy algorithm with candidate generation based on uncovered triples,
    followed by redundancy removal. Multiple runs are performed to find a better solution.
    """
    if v == 24 and k == 7 and t == 3:
        return _construct_c24_7_3()
    
    # Fallback for other parameters (simple greedy)
    return _simple_greedy(v, k, t)

def _simple_greedy(v, k, t):
    """Fallback simple greedy for non-specific parameters."""
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
    return blocks

def _construct_c24_7_3():
    v = 24
    k = 7
    t = 3
    
    # Precompute all 3-subsets and map them to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples) # 2024
    
    best_blocks = None
    best_len = float('inf')
    
    # Time limit check helper
    start_time = time.time()
    time_limit = 110.0 # Leave some buffer for overhead
    
    # Run multiple randomized greedy instances
    max_runs = 100
    
    for run in range(max_runs):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(run)
        
        # covered[i] is True if triple i is covered
        covered = np.zeros(n_triples, dtype=bool)
        blocks = []
        
        # Optionally fix the first block to break symmetry
        # Block {0,1,2,3,4,5,6}
        first_block = (0, 1, 2, 3, 4, 5, 6)
        blocks.append(first_block)
        for trip in itertools.combinations(first_block, t):
            covered[triple_to_idx[trip]] = True
        
        # Greedy loop
        while not covered.all():
            # Find an uncovered triple to guide candidate generation
            # Randomly picking an uncovered one helps with diversity
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            target_trip_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            target_trip = all_triples[target_trip_idx]
            
            # Generate candidate blocks containing target_trip
            # We need to choose k-t = 4 more points from the remaining v-t = 21 points
            remaining_points = [p for p in range(v) if p not in target_trip]
            
            # To speed up, we don't generate ALL candidates if there are too many, 
            # but 5985 is manageable.
            candidates = []
            # Optimization: We can sample a subset of candidates if time is tight, 
            # but 5985 * 35 ops is small. Let's generate all for accuracy in gain.
            # Actually, generating 5985 tuples every step might be slow in Python.
            # Let's try to be smarter: 
            # We can pre-compute the list of points for each triple? No.
            
            # Let's just iterate combinations
            # It might be better to pick a random subset of candidates to speed up 
            # and rely on multiple runs.
            # However, for C(24,7,3), the gain difference is often small.
            # Let's try evaluating a random subset of 100 candidates from the 5985.
            
            # Sample candidates
            # If we sample too few, we might miss the best block.
            # Let's evaluate all 5985 if it's fast enough.
            # 5985 * 35 = ~210k operations. 80 blocks * 210k = 17M ops. 
            # Python can do ~10-50M simple ops/sec. So ~1-2 seconds per run.
            # 100 runs = 100-200 seconds. Too slow.
            
            # Optimization: Reduce candidates.
            # Only consider candidates that cover at least 1 other uncovered triple?
            # Or just sample 500 candidates.
            
            n_samples = 1000
            if len(remaining_points) > 4:
                # Sample 4 points from 21
                # We need to be careful to generate distinct blocks
                sampled_indices = random.sample(range(len(remaining_points)), n_samples)
                candidate_points = [remaining_points[i] for i in sampled_indices]
                # Wait, random.sample gives indices, we need combinations of 4
                # This is tricky. Better to just generate all and take a random subset of the result?
                # Or generate random combinations directly.
                
                # Direct random generation of 4-subsets from 21
                candidates = []
                seen = set()
                for _ in range(n_samples):
                    pts = random.sample(remaining_points, 4)
                    block = tuple(sorted(target_trip + tuple(pts)))
                    if block not in seen:
                        seen.add(block)
                        candidates.append(block)
            else:
                candidates = [tuple(sorted(target_trip + tuple(remaining_points)))]
            
            if not candidates:
                # Fallback if sampling failed (unlikely)
                candidates = [tuple(sorted(target_trip + tuple(random.sample(remaining_points, 4))))]
            
            # Evaluate gain for each candidate
            best_gain = -1
            best_block = None
            # We want to maximize the number of NEWLY covered triples
            for block in candidates:
                # Count how many triples in this block are currently uncovered
                gain = 0
                for trip in itertools.combinations(block, t):
                    idx = triple_to_idx[trip]
                    if not covered[idx]:
                        gain += 1
                
                # Add a small random tie-breaker to encourage diversity
                gain += random.random() * 0.1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Update covered mask
            for trip in itertools.combinations(best_block, t):
                covered[triple_to_idx[trip]] = True
        
        # Post-processing: Remove redundant blocks
        # A block is redundant if all its triples are covered by other blocks
        # This is O(N^2 * 35). With N~80, 80^2 * 35 = 224k ops. Fast.
        
        # We need to check if removing block i leaves all its triples covered by others.
        # Efficient check:
        # 1. Count how many times each triple is covered by the current set of blocks.
        # 2. If a triple is covered by exactly 1 block, that block is NOT redundant.
        # 3. If a block contains no triple that is covered by exactly 1 block, it is redundant.
        
        # Let's implement this check
        triple_counts = np.zeros(n_triples, dtype=int)
        block_triples = []
        for block in blocks:
            trips = [triple_to_idx[trip] for trip in itertools.combinations(block, t)]
            block_triples.append(trips)
            for idx in trips:
                triple_counts[idx] += 1
        
        # Identify essential triples (covered by only 1 block)
        essential_mask = (triple_counts == 1)
        
        new_blocks = []
        for i, block in enumerate(blocks):
            # Check if this block has any essential triple
            is_essential = False
            for idx in block_triples[i]:
                if essential_mask[idx]:
                    is_essential = True
                    break
            if not is_essential:
                # Block is redundant, remove it
                continue
            else:
                new_blocks.append(block)
        
        # Update best
        if len(new_blocks) < best_len:
            best_len = len(new_blocks)
            best_blocks = new_blocks
            
            # Early exit if we hit the known lower bound or a very good number
            # Lower bound is 69. Best known is 76.
            if best_len <= 76:
                # We can try to optimize further or just break if we find 76
                # Let's run a few more to see if we can get 75
                if best_len <= 75:
                    break

    if best_blocks is None:
        # Fallback if something went wrong
        return _simple_greedy(v, k, t)
        
    return [list(b) for b in best_blocks]
