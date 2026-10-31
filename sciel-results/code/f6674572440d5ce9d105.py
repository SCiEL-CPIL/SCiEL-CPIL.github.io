import itertools
import random
import time

def construct(v, k, t):
    # Precompute mapping from t-subset to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = bytearray([1]) * N_t
        blocks = []
        
        while True:
            # Find an uncovered subset
            # Optimization: maintain a list of uncovered indices?
            # For now, scan. N_t is small.
            uncovered_indices = [i for i, val in enumerate(uncovered) if val]
            if not uncovered_indices:
                break
            
            start_idx = random.choice(uncovered_indices)
            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)
            
            # Add k - t points
            while len(block) < k:
                best_gain = -1
                best_points = []
                current_block_set = set(block)
                
                # Precompute combinations of block of size t-1
                if len(block) >= t - 1:
                    block_combos = list(itertools.combinations(block, t - 1))
                else:
                    block_combos = []
                    
                for x in range(v):
                    if x in current_block_set:
                        continue
                    
                    gain = 0
                    for Y in block_combos:
                        new_subset = tuple(sorted(Y + (x,)))
                        idx = subset_to_idx.get(new_subset, -1)
                        if idx != -1 and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                # Pick a point
                if best_points:
                    new_point = random.choice(best_points)
                else:
                    available = [x for x in range(v) if x not in current_block_set]
                    new_point = random.choice(available) if available else 0
                
                block.append(new_point)
                
                # Update uncovered
                # New t-subsets covered are those containing new_point and t-1 elements from the OLD block
                old_block = block[:-1]
                if len(old_block) >= t - 1:
                    new_combos = list(itertools.combinations(old_block, t - 1))
                    for Y in new_combos:
                        new_subset = tuple(sorted(Y + (new_point,)))
                        idx = subset_to_idx.get(new_subset, -1)
                        if idx != -1:
                            uncovered[idx] = 0
                            
            block.sort()
            blocks.append(tuple(block))
            
        return blocks

    def remove_redundant(blocks):
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            
            # Recalculate cover counts for current_blocks
            cover_count = [0] * N_t
            block_subsets_curr = []
            for b in current_blocks:
                subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
                block_subsets_curr.append(subs)
                for idx in subs:
                    cover_count[idx] += 1
            
            for i, b in enumerate(current_blocks):
                is_redundant = True
                for idx in block_subsets_curr[i]:
                    if cover_count[idx] <= 1:
                        is_redundant = False
                        break
                
                if not is_redundant:
                    changed = True
                else:
                    new_blocks.append(b)
            
            current_blocks = new_blocks
        
        return current_blocks

    # Main execution
    best_blocks = None
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # Run multiple restarts
    # Adjust number of restarts based on v
    # For v=19, 5000 restarts should be fast enough
    num_restarts = 5000
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        blocks = greedy_cover(random_seed=i)
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]
