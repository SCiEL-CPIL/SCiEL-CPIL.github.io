import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses randomized greedy construction followed by local search (deletion).
    """
    start_time = time.time()
    # Leave 10s buffer for safety
    deadline = start_time + 140.0
    
    # Precomputation
    all_ts = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    
    def randomized_greedy():
        # uncovered_count tracks how many times a t-subset is covered.
        # We start with 0.
        # We want to minimize the number of blocks, so we want to cover as many 0s as possible.
        # Actually, we just need to ensure every t-subset is covered at least once.
        # So we track "uncovered" as count == 0.
        # When we add a block, we increment counts.
        # Wait, my previous logic was:
        # "uncovered_count" started at 0.
        # "gain" = number of triples with count == 0.
        # Then "uncovered_count[indices] -= 1" ??
        # No, that logic was flawed in the sketch above.
        
        # Correct logic:
        # Let `covered` be a boolean array or count array.
        # Initialize `covered` to 0.
        # While there exists a triple with `covered` == 0:
        #   Pick a seed triple with `covered` == 0.
        #   Find block that maximizes number of triples with `covered` == 0.
        #   Add block.
        #   Increment `covered` for all triples in block.
        
        # My sketch had `uncovered_count` starting at 0 and decrementing?
        # No, I wrote `uncovered_count[indices] -= 1` in the sketch but initialized to 0. That would go negative.
        # Let's stick to `cover_counts` starting at 0 and incrementing.
        
        cover_counts = np.zeros(n_ts, dtype=np.int32)
        blocks = []
        
        # To speed up "find min", we can just check if np.any(cover_counts == 0)
        # And find indices where it is 0.
        
        while True:
            # Check if all covered
            if not np.any(cover_counts == 0):
                break
            
            # Find indices of uncovered triples
            zero_indices = np.where(cover_counts == 0)[0]
            
            # Pick a random seed from uncovered
            seed_idx = zero_indices[random.randint(0, len(zero_indices) - 1)]
            seed_ts = all_ts[seed_idx]
            
            # Generate candidates containing seed_ts
            seed_points = set(seed_ts)
            remaining_points = [p for p in range(v) if p not in seed_points]
            
            best_gain = -1
            best_candidates = []
            
            # Iterate combinations of k-t points from remaining
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(seed_ts + combo))
                
                # Calculate gain: number of triples in block with cover_count == 0
                block_triples = list(itertools.combinations(block, t))
                # Get indices
                indices = [t_to_idx[ts] for ts in block_triples]
                
                # Count zeros
                # cover_counts[indices] is an array slice
                gain = np.sum(cover_counts[indices] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [block]
                elif gain == best_gain:
                    best_candidates.append(block)
            
            # Pick random from best
            chosen_block = random.choice(best_candidates)
            blocks.append(chosen_block)
            
            # Update cover_counts
            block_triples = list(itertools.combinations(chosen_block, t))
            indices = [t_to_idx[ts] for ts in block_triples]
            cover_counts[indices] += 1
            
        return blocks

    def remove_redundant(blocks):
        changed = True
        while changed:
            changed = False
            # Build coverage counts
            cover_counts = np.zeros(n_ts, dtype=np.int32)
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    cover_counts[t_to_idx[ts]] += 1
            
            new_blocks = []
            for block in blocks:
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    if cover_counts[t_to_idx[ts]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple iterations
    while time.time() < deadline:
        blocks = randomized_greedy()
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback
        blocks = randomized_greedy()
        blocks = remove_redundant(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
