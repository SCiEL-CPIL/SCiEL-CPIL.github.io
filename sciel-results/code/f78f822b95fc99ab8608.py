import itertools
import random
import numpy as np

def construct(v, k, t):
    # Map t-subsets to indices for fast lookup
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts
    # Each restart is fast. Let's do 100 restarts.
    num_restarts = 100
    
    for _ in range(num_restarts):
        uncovered_count = np.ones(num_t_subs, dtype=np.int8)
        blocks = []
        
        # Shuffle indices to randomize the order in which uncovered subsets are addressed
        order = list(range(num_t_subs))
        random.shuffle(order)
        
        for idx in order:
            if uncovered_count[idx] > 0:
                # Start a new block with this t-subset
                seed_sub = t_subs[idx]
                block = list(seed_sub)
                
                # Greedily add points
                while len(block) < k:
                    best_gain = -1
                    candidates = []
                    
                    # Candidates are points not in block
                    for x in range(v):
                        if x in block:
                            continue
                        
                        # Calculate gain: how many t-subsets in (block + x) are uncovered?
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            t_sub = tuple(sorted(rest + (x,)))
                            if uncovered_count[t_sub_to_idx[t_sub]] > 0:
                                gain += 1
                                
                        if gain > best_gain:
                            best_gain = gain
                            candidates = [x]
                        elif gain == best_gain:
                            candidates.append(x)
                            
                    if not candidates:
                        # Should not happen
                        break
                    
                    # Randomly pick from best candidates
                    best_point = random.choice(candidates)
                    block.append(best_point)
                    
                # Sort block
                block.sort()
                blocks.append(block)
                
                # Update coverage
                for t_sub in itertools.combinations(block, t):
                    uncovered_count[t_sub_to_idx[t_sub]] -= 1
                    
        # Post-processing: Remove redundant blocks
        cover_count = np.zeros(num_t_subs, dtype=np.int8)
        for block in blocks:
            for t_sub in itertools.combinations(block, t):
                cover_count[t_sub_to_idx[t_sub]] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to avoid index shifting issues
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                removable = True
                for t_sub in itertools.combinations(block, t):
                    if cover_count[t_sub_to_idx[t_sub]] == 1:
                        removable = False
                        break
                if removable:
                    for t_sub in itertools.combinations(block, t):
                        cover_count[t_sub_to_idx[t_sub]] -= 1
                    blocks.pop(i)
                    changed = True
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
