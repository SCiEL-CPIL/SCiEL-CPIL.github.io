import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = set(all_t_subsets)
    
    # Mapping from subset to index for faster counting? 
    # Or just use sets of tuples. Sets of tuples are fine for v=19.
    
    def greedy_cover(seed_val):
        random.seed(seed_val)
        uncovered = set(t_subsets_set)
        blocks = []
        
        # To speed up gain calculation, we might precompute neighbors?
        # But v=19 is small.
        
        while uncovered:
            # Pick a pivot subset
            # Heuristic: Pick one that has fewest "potential" completions? 
            # Or just random to diversify.
            # Let's pick the first one for determinism in the main run, 
            # but random for restarts.
            pivot = random.choice(list(uncovered))
            
            block = list(pivot)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                
                # Candidate points
                candidates = [x for x in range(v) if x not in block]
                
                for p in candidates:
                    # Calculate gain: how many uncovered t-subsets are covered by adding p
                    # A t-subset S is covered by block+{p} if S is subset of block+{p}
                    # Since S has size t, and block has size i, 
                    # S must contain p and (t-1) elements from block.
                    gain = 0
                    # Generate all (t-1) subsets of block
                    # And check if (subset + p) is in uncovered
                    for rest in itertools.combinations(block, t - 1):
                        candidate_subset = tuple(sorted(rest + (p,)))
                        if candidate_subset in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        if gain == 0:
                            # If no gain, break early? 
                            # We still need to add a point to complete the block.
                            # If all gains are 0, we pick the first candidate.
                            pass
                
                # If best_gain is 0, we still add the first candidate to maintain block size
                if best_point == -1:
                    best_point = candidates[0]
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # Remove all t-subsets of block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    def reduce_blocks(blocks):
        # Calculate coverage counts
        # Use a dictionary or defaultdict
        cover_count = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                # Check if this block is essential
                essential = False
                for sub in itertools.combinations(block, t):
                    if cover_count.get(sub, 0) == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
            
            # Optimization: If no change, we can break. 
            # Note: Removing one block might make another non-essential in the next iteration.
            # The while loop handles this.
            
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    # Estimate time per run
    # Greedy is ~O(N_blocks * k * v * C(k, t-1))
    # N_blocks ~ 150. k=7, v=19. C(6,3)=20.
    # 150 * 7 * 19 * 20 ~ 400k ops. Very fast.
    # Reduction is ~O(N_blocks * C(k,t) * iterations)
    # 150 * 35 * 150 ~ 800k ops. Fast.
    
    # We can run many restarts.
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        blocks = greedy_cover(i)
        reduced = reduce_blocks(blocks)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced
            
            # Early exit if we hit a very good score?
            # Lower bound is 111. Best known 151.
            # If we get <= 151, it's good.
            if best_size <= 140:
                break

    if best_blocks is None:
        # Fallback to simple greedy if all else fails (shouldn't happen)
        best_blocks = greedy_cover(0)
        best_blocks = reduce_blocks(best_blocks)

    return best_blocks
