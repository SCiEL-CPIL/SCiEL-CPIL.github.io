import itertools
import time
import random
from math import comb

def construct(v, k, t):
    # Parameters for this specific task
    # v=19, k=8, t=5
    
    if v != 19 or k != 8 or t != 5:
        # Fallback for other parameters
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

    # Specific optimized solution for C(19, 8, 5)
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Precompute indices for all 5-subsets
    all_5subs = list(itertools.combinations(range(v), t))
    idx5 = {sub: i for i, sub in enumerate(all_5subs)}
    num_5subs = len(all_5subs)
    
    # Precompute quad_idx_map
    # quad_idx_map[Q] = {x: idx5[Q U {x}] for x in range(v) if x not in Q}
    quad_idx_map = {}
    for Q in itertools.combinations(range(v), 4):
        quad_idx_map[Q] = {}
        for x in range(v):
            if x not in Q:
                S = tuple(sorted(Q + (x,)))
                quad_idx_map[Q][x] = idx5[S]
                
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    for seed in range(2000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # U is a bitmask of uncovered 5-subsets
        U = (1 << num_5subs) - 1
        blocks = []
        
        while U:
            # Find an uncovered 5-subset to start the block
            # Pick a random set bit in U
            start_sub_idx = -1
            # Optimization: if U is sparse, random picking is good.
            # If U is dense, random picking is also good.
            while start_sub_idx == -1:
                cand_idx = random.randint(0, num_5subs - 1)
                if (U >> cand_idx) & 1:
                    start_sub_idx = cand_idx
            
            start_sub = all_5subs[start_sub_idx]
            block = list(start_sub)
            
            # Grow the block
            while len(block) < k:
                # Calculate gain for each candidate x not in block
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute quads of current block
                block_quads = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    x_map = quad_idx_map
                    for quad in block_quads:
                        idx = x_map[quad][x]
                        if (U >> idx) & 1:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Tie-breaking: random
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    # Should not happen
                    break
                    
                block.append(best_x)
                
                # Update U: remove all 5-subsets contained in block
                for sub in itertools.combinations(block, t):
                    idx = idx5[sub]
                    U &= ~(1 << idx)
            
            block.sort()
            blocks.append(block)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
                
            # Optional: if we reached a very good score, we might stop early?
            # But we want to minimize, so keep going if time allows.
            
    if best_blocks is None:
        return []
        
    return best_blocks
