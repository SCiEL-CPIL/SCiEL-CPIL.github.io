import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Simulated Annealing.
    
    Strategy:
    1. Generate an initial valid cover using a fast greedy algorithm.
    2. Perform Simulated Annealing (SA) with the following moves:
       - Remove a block if it doesn't invalidate the cover (Cost -1).
       - Replace a block with a new random block (Cost 0, explores neighborhood).
       - Add a random block (Cost +1, allows escape from local minima).
    3. The goal is to minimize the number of blocks.
    
    This differs from the current greedy+reduction by allowing temporary increases in cost
    and block swapping to escape local minima where no single block is redundant.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to get t-subsets of a block
    def get_block_tsubs(block):
        return [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    # --- Phase 1: Initial Greedy Cover ---
    def greedy_initial():
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        # Randomize point order for diversity
        points = list(range(v))
        
        while uncovered_set:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in points if x not in block_points]
            random.shuffle(remaining_points)
            
            # Greedily add points to maximize coverage of uncovered subsets
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    block_points.add(best_point)
                    remaining_points.remove(best_point)
                else:
                    if remaining_points:
                        p = remaining_points.pop(0)
                        block_points.add(p)
            
            block = tuple(sorted(block_points))
            
            # Avoid duplicate blocks
            if block not in blocks:
                blocks.append(block)
                for ts_idx in get_block_tsubs(block):
                    if ts_idx in uncovered_set:
                        uncovered_set.remove(ts_idx)
        
        return blocks
    
    # Start with a greedy cover
    random.seed(42)
    current_blocks = greedy_initial()
    
    # Initialize cover counts
    cover_count = [0] * num_t_subsets
    for b in current_blocks:
        for ts_idx in get_block_tsubs(b):
            cover_count[ts_idx] += 1
    
    best_blocks = [list(b) for b in current_blocks]
    best_len = len(current_blocks)
    
    # --- Phase 2: Simulated Annealing ---
    # Temperature schedule
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # Precompute all possible k-subsets? C(19,8) = 75582. This is small enough to precompute.
    all_k_blocks = list(itertools.combinations(range(v), k))
    # Map block to index for fast lookup? Or just use tuples.
    # Let's precompute the t-subsets for every possible block to speed up evaluation.
    # block_tsubs_cache[block_tuple] = list of t-subset indices
    block_tsubs_cache = {}
    for b in all_k_blocks:
        block_tsubs_cache[b] = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
    
    # Current state
    # We maintain a list of blocks and a set of blocks for O(1) lookup
    current_block_set = set(current_blocks)
    
    # Time budget for SA
    sa_deadline = deadline - 5.0
    
    while time.time() < sa_deadline and T > T_min:
        T *= cooling_rate
        
        # Choose a move type
        # 1. Remove (if valid)
        # 2. Replace (swap one block for another)
        # 3. Add (add a random block)
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try Remove
            if not current_blocks:
                continue
            i = random.randint(0, len(current_blocks) - 1)
            b = current_blocks[i]
            ts_indices = block_tsubs_cache[b]
            
            # Check if removal is valid
            removable = True
            for idx in ts_indices:
                if cover_count[idx] == 1:
                    removable = False
                    break
            
            if removable:
                # Accept removal (Cost -1)
                for idx in ts_indices:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                current_block_set.remove(b)
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
                    
        elif move_type < 0.8:
            # Try Replace
            if not current_blocks:
                continue
            i = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[i]
            
            # Pick a new block
            b_new = random.choice(all_k_blocks)
            
            if b_new == b_old:
                continue
            if b_new in current_block_set:
                continue
                
            ts_old = block_tsubs_cache[b_old]
            ts_new = block_tsubs_cache[b_new]
            
            # Calculate change in coverage
            # We need to ensure that after removing b_old and adding b_new, all t-subsets are still covered.
            # This is a strict constraint move. We don't allow temporary invalidity in this simple SA variant 
            # because checking validity is expensive if we allow it. 
            # Instead, we only accept if the new state is valid.
            
            # Check if b_new covers any t-subset that is ONLY covered by b_old
            valid_swap = True
            for idx in ts_old:
                if cover_count[idx] == 1:
                    # This t-subset is only covered by b_old. 
                    # b_new must cover it.
                    if idx not in ts_new:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform swap
                for idx in ts_old:
                    cover_count[idx] -= 1
                for idx in ts_new:
                    cover_count[idx] += 1
                    
                current_blocks[i] = b_new
                current_block_set.remove(b_old)
                current_block_set.add(b_new)
                
                # Cost is same, but structure changed. 
                # This might enable future removals.
                # We don't update best_len here as length didn't change.
                
        else:
            # Try Add
            # Add a random block not already present
            b_new = random.choice(all_k_blocks)
            if b_new in current_block_set:
                continue
                
            # Cost increases by 1. Accept with probability exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                for idx in block_tsubs_cache[b_new]:
                    cover_count[idx] += 1
                current_blocks.append(b_new)
                current_block_set.add(b_new)
    
    # Final Reduction Pass on the best result found
    # It's possible the SA found a smaller set, but we should try to reduce it further
    # using the standard reduction on the best_blocks found
    if best_blocks:
        # Convert to tuples for internal processing
        final_blocks = [tuple(b) for b in best_blocks]
        final_block_set = set(final_blocks)
        
        # Recompute cover counts for final_blocks
        final_cover_count = [0] * num_t_subsets
        for b in final_blocks:
            for idx in block_tsubs_cache[b]:
                final_cover_count[idx] += 1
        
        # Simple greedy reduction
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            indices = list(range(len(final_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(final_blocks):
                    continue
                b = final_blocks[i]
                removable = True
                for idx in block_tsubs_cache[b]:
                    if final_cover_count[idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for idx in block_tsubs_cache[b]:
                        final_cover_count[idx] -= 1
                    final_blocks.pop(i)
                    final_block_set.remove(b)
                    changed = True
                    break
        
        if len(final_blocks) < len(best_blocks):
            best_blocks = [list(b) for b in final_blocks]
            
    return best_blocks
