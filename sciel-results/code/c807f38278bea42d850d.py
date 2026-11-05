import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_4_subs = list(itertools.combinations(range(v), t))
    idx4 = {sub: i for i, sub in enumerate(all_4_subs)}
    n_4 = len(all_4_subs)
    
    gain_table = {}
    for sub3 in itertools.combinations(range(v), 3):
        gain_table[sub3] = {}
        for x in range(v):
            if x not in sub3:
                sub4 = tuple(sorted(sub3 + (x,)))
                gain_table[sub3][x] = idx4[sub4]
                
    best_blocks = None
    best_count = float('inf')
    
    # Seed the random generator for reproducibility if needed, but default is fine
    # random.seed(42) 
    
    while time.time() < deadline:
        cover_count = [0] * n_4
        uncovered_set = set(range(n_4))
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered subset to start the block
            # This helps escape local minima compared to always picking the first
            target_idx = random.choice(list(uncovered_set))
            block = list(all_4_subs[target_idx])
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    for sub3 in itertools.combinations(block, 3):
                        idx = gain_table[sub3][x]
                        if cover_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                block.append(best_x)
                
                for sub3 in itertools.combinations(block[:-1], 3):
                    idx = gain_table[sub3][best_x]
                    if cover_count[idx] == 0:
                        cover_count[idx] = 1
                        uncovered_set.remove(idx)
                    else:
                        cover_count[idx] += 1
                        
            blocks.append(tuple(block))
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
            # If we reach a very good score, we might want to stop or continue
            # Best known is 151. If we hit 151, we are competitive.
            # If we hit < 151, we are breaking the record.
            if best_count <= 150:
                # Continue searching for even better?
                # Yes, but be careful with time.
                pass

    if best_blocks is None:
        return []
        
    return best_blocks
