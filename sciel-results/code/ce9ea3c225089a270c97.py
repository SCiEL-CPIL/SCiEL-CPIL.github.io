"""Probabilistic covering design with efficient simulated annealing for C(19,8,5)."""
import random
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def get_block_tsubsets(block):
        return [t_subset_index[ts] for ts in itertools.combinations(block, t)]

    # 1. Greedy Initialization
    cov = [0] * num_t_subsets
    blocks = []
    current_uncovered = num_t_subsets
    
    # Fast Greedy:
    for _ in range(1000):
        if current_uncovered == 0:
            break
        if time.time() - start_time > 5:
            break
            
        best_gain = -1
        best_block = None
        
        # Sample 100 candidate blocks
        for _ in range(100):
            anchor_idx = random.randint(0, num_t_subsets - 1)
            anchor = all_t_subsets[anchor_idx]
            block = list(anchor)
            random.shuffle(block)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                # Pick best x from top 3
                best_x = None
                best_local_gain = -1
                for x in candidates[:3]:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_subset_index.get(ts)
                        if idx is not None and cov[idx] == 0:
                            gain += 1
                    if gain > best_local_gain:
                        best_local_gain = gain
                        best_x = x
                        
                if best_x is not None:
                    block.append(best_x)
                    
            block.sort()
            block_tup = tuple(block)
            
            # Calculate total gain for this block
            gain = 0
            for ts in itertools.combinations(block_tup, t):
                idx = t_subset_index[ts]
                if cov[idx] == 0:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block = block_tup
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        for ts in itertools.combinations(best_block, t):
            idx = t_subset_index[ts]
            if cov[idx] == 0:
                current_uncovered -= 1
            cov[idx] += 1

    # 2. Simulated Annealing
    best_uncovered = current_uncovered
    best_blocks = [b[:] for b in blocks]
    
    temperature = 10.0
    cooling = 0.999
    min_temp = 0.1
    
    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break
            
        # Pick a random block to modify
        b_idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[b_idx]
        
        # Generate a new block by swapping one point
        remove_idx = random.randint(0, k - 1)
        old_point = old_block[remove_idx]
        
        candidates = [x for x in range(v) if x not in old_block]
        if not candidates:
            continue
        new_point = random.choice(candidates)
        
        if old_point == new_point:
            continue
            
        new_block = old_block[:]
        new_block[remove_idx] = new_point
        new_block.sort()
        new_block_tup = tuple(new_block)
        
        # Check for duplicates
        is_dup = False
        for i, b in enumerate(blocks):
            if i != b_idx and b == new_block_tup:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Calculate delta
        old_tsubs = get_block_tsubsets(tuple(old_block))
        new_tsubs = get_block_tsubsets(new_block_tup)
        
        # Store current uncovered before move
        unc_before = current_uncovered
        
        # Remove old
        for idx in old_tsubs:
            if cov[idx] == 1:
                current_uncovered += 1
            cov[idx] -= 1
            
        # Add new
        for idx in new_tsubs:
            cov[idx] += 1
            if cov[idx] == 1:
                current_uncovered -= 1
                
        delta = current_uncovered - unc_before
        
        # Metropolis acceptance
        if delta <= 0 or random.random() < pow(2.718281828, -delta / temperature):
            blocks[b_idx] = new_block_tup
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_blocks = [b[:] for b in blocks]
        else:
            # Revert
            for idx in new_tsubs:
                if cov[idx] == 1:
                    current_uncovered += 1
                cov[idx] -= 1
            for idx in old_tsubs:
                cov[idx] += 1
                if cov[idx] == 1:
                    current_uncovered -= 1
            current_uncovered = unc_before

        temperature *= cooling
        
    return [list(b) for b in best_blocks]
