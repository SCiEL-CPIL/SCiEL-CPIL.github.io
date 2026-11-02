import itertools
import time
import random
import math

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin under 120s limit
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to its index for O(1) lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute t-subsets for each possible k-subset? No, too many.
    # Instead, helper to get t-subset indices for a given block (list of k points)
    def get_block_tsub_indices(block):
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Greedy Construction
    # We want to start with a reasonably good covering.
    covered = [False] * n_t
    blocks = []
    
    def add_block_greedy(block):
        for idx in get_block_tsub_indices(block):
            covered[idx] = True
        blocks.append(tuple(block))
        
    def greedy_extend(seed_subset, current_block_set, covered_state):
        """Extend current_block_set to size k, maximizing coverage of uncovered t-subsets."""
        block_set = set(current_block_set)
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            # Candidate points
            for x in all_points:
                if x in block_set:
                    continue
                # Calculate gain: how many new t-subsets are covered?
                # A t-subset is covered if it is a subset of block_set U {x}
                # It must contain x and t-1 points from block_set
                gain = 0
                # To be efficient, we can just check all t-subsets containing x?
                # Or check all t-subsets of (block_set U {x}) that include x.
                # Number of such t-subsets is C(k-1, t-1). For k=7, t=4, C(6,3)=20.
                # This is fast.
                temp_set = block_set | {x}
                for sub in itertools.combinations(temp_set, t):
                    if x in sub:
                        idx = t_sub_to_idx[sub]
                        if not covered_state[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
        return sorted(block_set)

    # Initial Greedy Phase
    while True:
        if time.time() > deadline - 60: # Leave time for SA
            break
        # Find first uncovered
        uncovered_idx = None
        for i in range(n_t):
            if not covered[i]:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        seed_subset = all_t_subsets[uncovered_idx]
        new_block = greedy_extend(seed_subset, set(seed_subset), covered)
        add_block_greedy(new_block)
        
    # Ensure full coverage (in case we broke out due to time)
    if any(not c for c in covered):
        # Force complete
        while True:
            uncovered_idx = None
            for i in range(n_t):
                if not covered[i]:
                    uncovered_idx = i
                    break
            if uncovered_idx is None:
                break
            seed_subset = all_t_subsets[uncovered_idx]
            new_block = greedy_extend(seed_subset, set(seed_subset), covered)
            add_block_greedy(new_block)
            if time.time() > deadline - 5:
                break

    if not blocks:
        return []

    # Simulated Annealing
    # State: blocks (list of tuples), cover_count (list of ints)
    # Cost: len(blocks) + LAMBDA * num_uncovered
    
    cover_count = [0] * n_t
    num_uncovered = 0
    for b in blocks:
        for idx in get_block_tsub_indices(b):
            if cover_count[idx] == 0:
                num_uncovered -= 1
            cover_count[idx] += 1
            
    # If we broke early, num_uncovered might be > 0
    # Recalculate num_uncovered from cover_count to be safe
    num_uncovered = sum(1 for c in cover_count if c == 0)
    
    def get_cost(blocks_list, num_unc):
        return len(blocks_list) + 1000 * num_unc

    current_blocks = list(blocks)
    current_cover_count = cover_count
    current_num_uncovered = num_uncovered
    current_cost = get_cost(current_blocks, current_num_uncovered)
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    best_num_uncovered = current_num_uncovered
    
    T = 1.0
    cooling = 0.999
    # We want to run SA for the remaining time
    sa_deadline = deadline - 2
    
    # Helper to apply a move
    def apply_remove(idx):
        nonlocal current_cover_count, current_num_uncovered
        block = current_blocks[idx]
        for t_idx in get_block_tsub_indices(block):
            current_cover_count[t_idx] -= 1
            if current_cover_count[t_idx] == 0:
                current_num_uncovered += 1
        del current_blocks[idx]
        
    def apply_add(block):
        nonlocal current_cover_count, current_num_uncovered
        for t_idx in get_block_tsub_indices(block):
            if current_cover_count[t_idx] == 0:
                current_num_uncovered -= 1
            current_cover_count[t_idx] += 1
        current_blocks.append(tuple(block))
        
    def undo_remove(block, idx):
        # Re-insert block at idx
        nonlocal current_cover_count, current_num_uncovered
        current_blocks.insert(idx, block)
        for t_idx in get_block_tsub_indices(block):
            if current_cover_count[t_idx] == 0:
                current_num_uncovered -= 1
            current_cover_count[t_idx] += 1
            
    def undo_add(block):
        nonlocal current_cover_count, current_num_uncovered
        # Assumes block is at end
        current_blocks.pop()
        for t_idx in get_block_tsub_indices(block):
            current_cover_count[t_idx] -= 1
            if current_cover_count[t_idx] == 0:
                current_num_uncovered += 1

    # Pre-generate some random blocks for "Add" moves when valid?
    # Or just generate on the fly.
    
    while time.time() < sa_deadline:
        if T < 1e-6:
            break
            
        move_type = random.random()
        if move_type < 0.5:
            # Remove
            if not current_blocks:
                continue
            idx = random.randint(0, len(current_blocks) - 1)
            block_to_remove = current_blocks[idx]
            
            apply_remove(idx)
            new_cost = get_cost(current_blocks, current_num_uncovered)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(current_blocks)
                    best_num_uncovered = current_num_uncovered
            else:
                undo_remove(block_to_remove, idx)
        else:
            # Add
            # Choose a t-subset to base the block on
            # If there are uncovered, pick one of them to fix
            # Otherwise, pick random
            if current_num_uncovered > 0:
                # Find an uncovered one
                # To avoid O(N) scan every time, we could maintain a list, but N is small (3876)
                # Random scan is fine
                for _ in range(10):
                    idx = random.randint(0, n_t - 1)
                    if current_cover_count[idx] == 0:
                        seed_sub = all_t_subsets[idx]
                        break
                else:
                    seed_sub = all_t_subsets[random.randint(0, n_t - 1)]
            else:
                seed_sub = all_t_subsets[random.randint(0, n_t - 1)]
                
            # Greedily extend seed_sub to a block
            # Use a local covered view? No, just extend based on current cover_count
            # We want to maximize coverage of *currently uncovered* subsets
            block_set = set(seed_sub)
            while len(block_set) < k:
                best_x = None
                best_gain = -1
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    temp_set = block_set | {x}
                    for sub in itertools.combinations(temp_set, t):
                        if x in sub:
                            idx = t_sub_to_idx[sub]
                            if current_cover_count[idx] == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block_set.add(best_x)
            
            new_block = tuple(sorted(block_set))
            
            apply_add(new_block)
            new_cost = get_cost(current_blocks, current_num_uncovered)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(current_blocks)
                    best_num_uncovered = current_num_uncovered
            else:
                undo_add(new_block)
                
        T *= cooling
        
    # Return the best valid solution found
    # Ensure best_blocks is valid
    # If best_num_uncovered > 0, we should probably return current_blocks if it's valid, 
    # or re-verify best_blocks.
    
    # Verification of best_blocks
    cover_check = [0] * n_t
    for b in best_blocks:
        for idx in get_block_tsub_indices(b):
            cover_check[idx] += 1
            
    if any(c == 0 for c in cover_check):
        # Fallback to current state if it is valid
        cover_check_cur = [0] * n_t
        for b in current_blocks:
            for idx in get_block_tsub_indices(b):
                cover_check_cur[idx] += 1
        if not any(c == 0 for c in cover_check_cur):
            best_blocks = current_blocks
        else:
            # If both invalid (shouldn't happen if init was valid and we track correctly),
            # return a valid greedy solution
            # Re-run greedy to be safe?
            # For now, return best_blocks and hope, or fix it.
            # Let's just fix best_blocks by adding blocks for uncovered
            for i in range(n_t):
                if cover_check[i] == 0:
                    seed = all_t_subsets[i]
                    b_set = set(seed)
                    while len(b_set) < k:
                        # Add random points
                        pt = random.choice(all_points)
                        if pt not in b_set:
                            b_set.add(pt)
                    b_tuple = tuple(sorted(b_set))
                    best_blocks.append(b_tuple)
                    for idx in get_block_tsub_indices(b_tuple):
                        cover_check[idx] += 1

    return [list(b) for b in best_blocks]
