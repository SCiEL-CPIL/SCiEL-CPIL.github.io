import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all t-subsets and their indices
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    covered = np.zeros(n_subs, dtype=bool)
    uncovered_set = set(range(n_subs))
    
    blocks = []
    
    while uncovered_set and time.time() < deadline:
        # Start a new block with the first uncovered subset
        first_uncovered = min(uncovered_set)
        current_block = list(all_subs[first_uncovered])
        
        # Greedy expansion
        while len(current_block) < k:
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in current_block:
                    continue
                
                gain = 0
                # Iterate over combinations of t-1 points from current_block
                if t == 3:
                    # Optimization for t=3: pairs
                    cb = current_block
                    l = len(cb)
                    for i in range(l):
                        p1 = cb[i]
                        for j in range(i+1, l):
                            p2 = cb[j]
                            # Create tuple and sort
                            # Since x is new, we can check if (min, mid, max) is in sub_to_idx
                            # And if it's uncovered
                            a, b, c = x, p1, p2
                            if a > b: a, b = b, a
                            if b > c: b, c = c, b
                            if a > b: a, b = b, a
                            triple = (a, b, c)
                            idx = sub_to_idx.get(triple)
                            if idx is not None and not covered[idx]:
                                gain += 1
                else:
                    for combo in itertools.combinations(current_block, t-1):
                        triple = tuple(sorted(combo + (x,)))
                        idx = sub_to_idx.get(triple)
                        if idx is not None and not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
                
            current_block.append(best_point)
            
            # Update covered status
            for sub in itertools.combinations(current_block, t):
                idx = sub_to_idx.get(tuple(sorted(sub)))
                if idx is not None and not covered[idx]:
                    covered[idx] = True
                    uncovered_set.remove(idx)
            
            if not uncovered_set:
                break
        
        current_block.sort()
        blocks.append(current_block)
        
        if time.time() > deadline:
            break

    # Post-processing: Remove redundant blocks
    def try_remove_blocks(blocks):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cover_count = np.zeros(n_subs, dtype=int)
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx.get(tuple(sorted(sub)))
                    if idx is not None:
                        cover_count[idx] += 1
            
            to_remove = []
            for i, block in enumerate(blocks):
                redundant = True
                for sub in itertools.combinations(block, t):
                    idx = sub_to_idx.get(tuple(sorted(sub)))
                    if idx is not None and cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            if to_remove:
                for i in reversed(to_remove):
                    blocks.pop(i)
                changed = True
                
        return blocks

    if time.time() < deadline:
        blocks = try_remove_blocks(blocks)
        
    return blocks
