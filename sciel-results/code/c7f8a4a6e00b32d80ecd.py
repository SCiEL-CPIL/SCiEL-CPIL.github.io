import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = []
    for sub in itertools.combinations(range(v), t):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        all_t_subsets.append(mask)
        
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        rng = random.Random(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick the minimum uncovered subset to start
            s_mask = min(uncovered)
            
            # Convert mask to sorted list of points
            block = []
            temp_mask = s_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                p = lsb.bit_length() - 1
                block.append(p)
                temp_mask ^= lsb
            block.sort()
            block_set = set(block)
            
            while len(block) < k:
                gains = []
                # Precompute t-1 subsets of current block as masks
                block_masks = []
                for rest in itertools.combinations(block, t - 1):
                    m = 0
                    for p in rest:
                        m |= (1 << p)
                    block_masks.append(m)
                    
                for x in range(v):
                    if x in block_set:
                        continue
                    x_mask = 1 << x
                    gain = 0
                    for bm in block_masks:
                        cand = bm | x_mask
                        if cand in uncovered:
                            gain += 1
                    # Add small noise to break ties and explore
                    score = gain + rng.random() * 0.1
                    gains.append((score, x))
                
                if not gains:
                    break
                    
                # Sort by score descending
                gains.sort(reverse=True)
                max_score = gains[0][0]
                
                # Collect all points with close to max score? 
                # Or just the top one? 
                # If we take the top one, the noise handles ties.
                x = gains[0][1]
                
                block.append(x)
                block_set.add(x)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                uncovered.discard(m)
                
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 131:
            break

    return best_blocks
