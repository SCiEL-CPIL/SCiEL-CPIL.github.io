import itertools
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a multi-start greedy heuristic.
    """
    all_t_subsets = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts to find a better solution
    # 120 seconds limit allows for many fast greedy runs
    num_restarts = 50
    
    for seed in range(num_restarts):
        random.seed(seed)
        blocks = []
        # coverage[t_subset] = number of blocks containing it
        coverage = {ts: 0 for ts in all_t_subsets}
        uncovered_count = len(all_t_subsets)
        
        while uncovered_count > 0:
            # Find all uncovered t-subsets
            # To save time, we can just pick a random uncovered one
            # But maintaining a list of uncovered ones is better for O(1) access if we update it
            # However, updating the list of uncovered ones is tricky when coverage goes 0->1.
            # Given small size, scanning is fine, or we can just pick random from all and check.
            # Better: maintain a set of uncovered subsets.
            
            # Let's use a set for uncovered for efficient removal
            # Re-initializing or updating a set of 11k items is fast.
            # But we need to know which ones are uncovered.
            # Let's just iterate. With 11k items, finding one uncovered is fast.
            # Actually, let's keep a list of uncovered ones and shuffle it occasionally?
            # No, simpler: just pick a random t-subset from the whole list until we find an uncovered one.
            # Probability of picking uncovered decreases as we go, but initially it's high.
            # In the end, it might be slow.
            
            # Efficient approach: maintain a set `uncovered_set`.
            # Initialize: set(all_t_subsets)
            # When a block is added, for each t-subset in the block:
            #   coverage[ts] += 1
            #   if coverage[ts] == 1: remove from uncovered_set
            
            # Let's implement this version.
            pass # Placeholder for logic below
            
        # The logic above is a bit mixed. Let's write it cleanly.
        
    return best_blocks

# Let's rewrite the construct function properly.

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 100 # Fast enough to run many
    
    for seed in range(num_restarts):
        random.seed(seed)
        blocks = []
        coverage = {ts: 0 for ts in all_t_subsets}
        uncovered_set = set(all_t_subsets)
        
        while uncovered_set:
            # Pick a random uncovered t-subset to start the new block
            start_subset = random.choice(tuple(uncovered_set))
            block = list(start_subset)
            
            # Extend the block to size k
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # A t-subset T is newly covered if T \ {x} is a (t-1)-subset of block
                    # and T is currently uncovered.
                    # T = (t-1)-subset from block + x
                    for rest in itertools.combinations(block, t - 1):
                        T = tuple(sorted(rest + (x,)))
                        if T in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # Pick randomly from best candidates to diversify
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    # Fallback: pick any point not in block
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break # Should not happen if k <= v
            
            block.sort()
            blocks.append(block)
            
            # Update coverage and uncovered_set
            for ts in itertools.combinations(block, t):
                coverage[ts] += 1
                if coverage[ts] == 1:
                    uncovered_set.discard(ts)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        # Early stopping if we hit a known good bound? 
        # Best known is 335. If we get <= 335, we can stop? 
        # The task asks to minimize. We should keep trying if we can.
        # But 100 restarts of this greedy is likely to find a good solution.
        
    return best_blocks
