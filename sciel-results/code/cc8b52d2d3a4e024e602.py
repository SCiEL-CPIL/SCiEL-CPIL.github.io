import itertools
import time
import random

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy set cover.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    
    best_blocks = None
    
    # To speed up restarts, we might want to shuffle the order of points or subsets
    # But randomizing the selection within the greedy step is more effective.
    
    while time.time() < deadline:
        current_uncovered = set(uncovered)
        current_blocks = []
        
        while current_uncovered:
            # Pick a random uncovered subset to seed the new block
            # Using list(current_uncovered) is O(N), which is expensive.
            # Better: iterate until we find one? No, we need a random one.
            # For v=24, t=3, N = 2024. list() is fast enough.
            seed_subset = random.choice(list(current_uncovered))
            block = list(seed_subset)
            
            # Fill the block
            while len(block) < k:
                # Calculate gain for each candidate point x not in block
                # gain(x) = number of subsets S in current_uncovered 
                # such that S contains x and t-1 elements from block
                
                best_gain = -1
                best_candidates = []
                
                # We can optimize by precomputing combinations of block of size t-1
                block_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    for combo in block_combos:
                        # Check if sorted(combo + (x,)) is in current_uncovered
                        # To make it a tuple for set lookup, we need it sorted
                        # Since combo is from itertools.combinations, it is sorted.
                        # We just need to insert x in the correct sorted position.
                        # Or just create the tuple and sort it.
                        # t is small (3), so sorting is cheap.
                        subset = tuple(sorted(combo + (x,)))
                        if subset in current_uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # This should not happen if current_uncovered is not empty and k < v
                    # But if all gains are 0, we pick a random point to avoid infinite loop
                    # Actually, if gain is 0 for all, we still need to add a point to reach size k?
                    # No, if gain is 0, adding it doesn't help cover new subsets.
                    # But we must form a valid block of size k.
                    # The problem says "every t-subset lies in some block".
                    # If we add points with 0 gain, we might end up with a block that covers nothing new.
                    # This is inefficient. 
                    # However, the greedy step *starts* with an uncovered subset, so the first points 
                    # (from seed) already cover that subset.
                    # Wait, the seed subset is added to `block`. 
                    # Do we remove it from `current_uncovered` immediately?
                    # No, we remove all subsets of the final block at the end.
                    # But during the filling process, the seed subset is still in `current_uncovered`.
                    # So adding points that complete the seed subset will have gain >= 1.
                    # Specifically, the seed subset itself will be covered once the block is complete.
                    # Actually, we should remove subsets covered by the *current* partial block?
                    # No, a partial block doesn't cover any t-subset yet (unless t <= len(block)).
                    # If t <= len(block), then subsets of the block are covered.
                    # In our loop, len(block) starts at t (size of seed).
                    # So the seed subset IS covered by the current block.
                    # We should remove it from current_uncovered immediately to avoid counting it again?
                    # Or just let it be, and remove all at the end.
                    # If we don't remove it, gain calculations for other points might be affected?
                    # No, gain counts subsets in `current_uncovered`.
                    # If the seed subset is in `current_uncovered`, and we add a point x,
                    # does x help cover the seed subset? No, the seed subset is already in the block.
                    # So the seed subset will be removed at the end.
                    # It's safer to remove subsets covered by the current block immediately?
                    # Let's stick to removing at the end of the block construction.
                    # But we must ensure we don't pick a point that doesn't add any *new* coverage if possible.
                    
                    # If best_gain is 0, we pick a random point not in block.
                    available = [x for x in range(v) if x not in block]
                    x = random.choice(available)
                    block.append(x)
                else:
                    # Pick randomly from best candidates to diversify
                    x = random.choice(best_candidates)
                    block.append(x)
            
            block.sort()
            current_blocks.append(block)
            
            # Remove all t-subsets of this block from current_uncovered
            for subset in itertools.combinations(block, t):
                current_uncovered.discard(tuple(subset))
        
        # Check if this run is better than the best so far
        if best_blocks is None or len(current_blocks) < len(best_blocks):
            best_blocks = current_blocks
            # Optional: early termination if we reach a known lower bound or very good score
            # For C(24,7,3), best known is 76.
            # If we find 76, we can stop.
            if len(current_blocks) <= 76:
                break
        
        # Reset random seed for reproducibility in some contexts? 
        # No, we want diversity.
        
    if best_blocks is None:
        # Fallback: return the seed greedy result or just an empty list?
        # The task requires a valid covering. 
        # If we timed out before finishing a single run, we have no valid covering.
        # We must ensure at least one run completes.
        # With v=24, k=7, t=3, one run takes:
        # ~76 blocks. Each block construction: 4 points added (7-3).
        # Each step: 24 points * 15 combos = 360 checks.
        # 76 * 4 * 360 = ~100,000 operations. Very fast.
        # So we will definitely complete many runs.
        return []
        
    return best_blocks
