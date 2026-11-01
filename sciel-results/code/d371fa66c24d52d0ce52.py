import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    total_tsubsets = list(itertools.combinations(all_points, t))
    n_tsubsets = len(total_tsubsets)
    
    subset_to_idx = {s: i for i, s in enumerate(total_tsubsets)}
    
    def generate_initial_covering(rng):
        covered_count = [0] * n_tsubsets
        blocks = []
        uncovered = set(range(n_tsubsets))
        
        while uncovered:
            block = rng.sample(all_points, k)
            block.sort()
            
            added_any = False
            for combo in itertools.combinations(block, t):
                idx = subset_to_idx[tuple(combo)]
                if covered_count[idx] == 0:
                    added_any = True
                    covered_count[idx] = 1
                    uncovered.discard(idx)
                else:
                    covered_count[idx] += 1
            
            if added_any:
                blocks.append(block)
            
        return blocks, covered_count

    def local_search(blocks, covered_count, rng, time_limit):
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Try to remove a block
            removed = False
            block_indices = list(range(len(blocks)))
            rng.shuffle(block_indices)
            
            for i in block_indices:
                block = blocks[i]
                removable = True
                for combo in itertools.combinations(block, t):
                    idx = subset_to_idx[tuple(combo)]
                    if covered_count[idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for combo in itertools.combinations(block, t):
                        idx = subset_to_idx[tuple(combo)]
                        covered_count[idx] -= 1
                    blocks.pop(i)
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try to mutate a block
            if not blocks:
                break
                
            # Try a few random mutations
            mutation_success = False
            for _ in range(20):
                block_idx = rng.choice(range(len(blocks)))
                block = blocks[block_idx]
                
                point_idx = rng.choice(range(k))
                old_point = block[point_idx]
                
                available = [p for p in all_points if p not in block]
                if not available:
                    continue
                new_point = rng.choice(available)
                
                new_block = block.copy()
                new_block[point_idx] = new_point
                new_block.sort()
                
                # Check if this mutation is valid
                # We need to ensure no t-subset drops to 0
                # Calculate subsets that lose coverage
                lost = set()
                for combo in itertools.combinations(block, t):
                    if old_point in combo:
                        lost.add(tuple(combo))
                
                # Calculate subsets that gain coverage
                gained = set()
                for combo in itertools.combinations(new_block, t):
                    if new_point in combo:
                        gained.add(tuple(combo))
                
                # A subset is critical if it is in lost but not in gained
                # and its count is 1
                valid = True
                for s in lost:
                    if s not in gained:
                        idx = subset_to_idx[s]
                        if covered_count[idx] == 1:
                            valid = False
                            break
                
                if valid:
                    # Apply mutation
                    # Update counts
                    for s in lost:
                        idx = subset_to_idx[s]
                        covered_count[idx] -= 1
                    for s in gained:
                        idx = subset_to_idx[s]
                        covered_count[idx] += 1
                    
                    blocks[block_idx] = new_block
                    mutation_success = True
                    break
            
            # If no mutation succeeded, we are stuck in a local minimum.
            # We could restart, but the outer loop handles restarts.
            # To avoid infinite loop of failing mutations, we just break out of this restart.
            if not mutation_success:
                # Optionally, we could try a more aggressive move, 
                # but for now, we accept the local minimum for this restart.
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    time_limit = 110
    
    seed = 42
    while time.time() - start_time < time_limit:
        rng = random.Random(seed)
        
        # Phase 1
        blocks, covered_count = generate_initial_covering(rng)
        
        # Phase 2
        # Run local search for a short duration
        # We estimate how many restarts we can do.
        # If we spend 1s on LS, we can do ~100 restarts.
        # The initial covering is fast.
        ls_time = 0.5 # Half a second per restart
        
        blocks = local_search(blocks, covered_count, rng, ls_time)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
    return best_blocks
