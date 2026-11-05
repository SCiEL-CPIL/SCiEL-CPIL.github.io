import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index for fast lookup
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    # We will run as many as possible within the time limit
    max_restarts = 10000  # High number, limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        covered = [False] * num_t_subs
        blocks = []
        uncovered_set = set(range(num_t_subs))
        completed = True
        
        while uncovered_set:
            if time.time() > deadline:
                completed = False
                break
                
            # Pick a random uncovered subset to start the block
            # Converting to list is O(N) but N is small (~4000)
            idx = random.choice(list(uncovered_set))
            start_sub = all_t_subs[idx]
            
            # Initialize block with the start subset
            block = list(start_sub)
            
            # If t == k, we are done with this block
            while len(block) < k:
                # Find best point to add
                max_gain = -1
                best_points = []
                
                # Precompute (t-1) subsets of current block
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                # Evaluate each candidate point
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check all (t-1) subsets of block
                    for p in block_t_minus_1:
                        # Form candidate t-subset S = P U {x}
                        # P is a tuple of t-1 elements
                        s = tuple(sorted(p + (x,)))
                        
                        # Check if S is a valid t-subset and uncovered
                        if s in sub_to_idx:
                            s_idx = sub_to_idx[s]
                            if not covered[s_idx]:
                                gain += 1
                                
                    if gain > max_gain:
                        max_gain = gain
                        best_points = [x]
                    elif gain == max_gain:
                        best_points.append(x)
                
                # Pick a point randomly from best_points
                if best_points:
                    new_point = random.choice(best_points)
                    block.append(new_point)
                else:
                    # Fallback: pick any random point not in block
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        new_point = random.choice(remaining)
                        block.append(new_point)
                    else:
                        completed = False
                        break
                
                # Update covered set
                for s in itertools.combinations(block, t):
                    s_idx = sub_to_idx[s]
                    if not covered[s_idx]:
                        covered[s_idx] = True
                        uncovered_set.discard(s_idx)
                        
            blocks.append(block)
            
        if completed and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # Early termination if we hit the lower bound
            if best_count <= 131:
                break
                
    if best_blocks is None:
        # If we didn't find a valid covering (should not happen), 
        # return an empty list which will be invalid.
        return []
        
    return best_blocks
