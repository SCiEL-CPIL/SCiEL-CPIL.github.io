import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Evaluator kills at 150s. Use 115s as hard stop for search.
    deadline = start_time + 115
    
    # Precompute t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    def get_block_ids(block):
        # block is a tuple of ints
        return [t_subset_to_id[s] for s in itertools.combinations(block, t)]

    def run_greedy(rng, time_limit):
        uncovered = set(range(num_t_subsets))
        # List for random selection, lazy deletion
        uncovered_list = list(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            if time.time() > time_limit:
                return None # Failed to complete
            
            # Pick a random seed from uncovered
            # Find a valid index in uncovered_list
            seed_id = None
            attempts = 0
            while seed_id is None:
                idx = rng.randrange(len(uncovered_list))
                candidate = uncovered_list[idx]
                if candidate in uncovered:
                    seed_id = candidate
                else:
                    # Lazy removal
                    uncovered_list[idx] = uncovered_list[-1]
                    uncovered_list.pop()
                    if not uncovered_list:
                        break
                attempts += 1
                if attempts > 100: # Safety break
                    break
            
            if seed_id is None:
                break
            
            seed_subset = all_t_subsets[seed_id]
            current_block = list(seed_subset)
            
            # Build the rest of the block
            while len(current_block) < k:
                best_gain = -1
                best_candidates = []
                
                # Points to consider
                for x in range(v):
                    if x in current_block:
                        continue
                    
                    gain = 0
                    # Check t-subsets containing x and t-1 points from current_block
                    if len(current_block) >= t - 1:
                        for combo in itertools.combinations(current_block, t - 1):
                            t_subset = tuple(sorted(combo + (x,)))
                            sid = t_subset_to_id.get(t_subset)
                            if sid is not None and sid in uncovered:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_candidates:
                    next_point = rng.choice(best_candidates)
                    current_block.append(next_point)
                else:
                    # Fallback: pick random point
                    remaining = [x for x in range(v) if x not in current_block]
                    if remaining:
                        current_block.append(rng.choice(remaining))
                    else:
                        break
            
            current_block.sort()
            block_tuple = tuple(current_block)
            blocks.append(block_tuple)
            
            # Update uncovered
            covered_ids = get_block_ids(block_tuple)
            uncovered.difference_update(covered_ids)
            
        return blocks if not uncovered else None

    def remove_redundant(blocks, time_limit):
        if not blocks:
            return blocks
            
        # Precompute IDs for each block
        block_ids_list = [get_block_ids(b) for b in blocks]
        
        coverage_counts = [0] * num_t_subsets
        for ids in block_ids_list:
            for sid in ids:
                coverage_counts[sid] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            new_blocks = []
            new_block_ids = []
            
            for i, block in enumerate(blocks):
                ids = block_ids_list[i]
                essential = False
                for sid in ids:
                    if coverage_counts[sid] == 1:
                        essential = True
                        break
                
                if essential:
                    new_blocks.append(block)
                    new_block_ids.append(ids)
                else:
                    # Remove
                    for sid in ids:
                        coverage_counts[sid] -= 1
                    changed = True
            
            blocks = new_blocks
            block_ids_list = new_block_ids
            
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Baseline run
    rng = random.Random(42)
    res = run_greedy(rng, deadline)
    if res:
        res = remove_redundant(res, deadline)
        best_blocks = res
        best_len = len(res)
        
    # Random restarts
    for seed in range(1, 10000):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed)
        res = run_greedy(rng, deadline)
        if res:
            res = remove_redundant(res, deadline)
            if len(res) < best_len:
                best_blocks = res
                best_len = len(res)
                
    if best_blocks is None:
        # Should not happen, but return empty list to avoid crash
        return []
        
    return [list(b) for b in best_blocks]
