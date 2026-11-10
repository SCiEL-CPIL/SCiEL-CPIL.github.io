import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_index = {trip: i for i, trip in enumerate(all_triples)}
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_blocks:
        covered = []
        for trip in itertools.combinations(block, t):
            covered.append(triple_index[trip])
        block_triples.append(covered)
        
    # Precompute bitmask for each block for fast intersection and counting
    # n_triples is ~2024, fits in Python int.
    block_masks = [0] * n_blocks
    for i, trips in enumerate(block_triples):
        m = 0
        for t_idx in trips:
            m |= (1 << t_idx)
        block_masks[i] = m

    # 2. Greedy Initialization
    def greedy_init(rng):
        uncovered_mask = (1 << n_triples) - 1
        selected = []
        
        while uncovered_mask:
            best_gain = -1
            best_block = -1
            
            # Randomized greedy: scan all blocks, pick best gain.
            # To add diversity, we can randomize the order or tie-breaking.
            # Here we use a simple scan with random tie-breaking.
            # Optimization: If we find a block covering all remaining uncovered, we can stop?
            # No, we need to cover ALL. But if gain == bit_count(uncovered_mask), we are done.
            
            # Iterate through blocks. 
            # To make it faster, we can skip blocks that don't intersect uncovered_mask?
            # block_masks[i] & uncovered_mask
            # If 0, gain is 0.
            
            # We want to maximize gain.
            # We can just iterate all.
            # 346k blocks.
            # In Python, this loop might take ~0.1-0.2s per iteration if optimized?
            # 70 iterations * 0.2s = 14s. Acceptable.
            
            # To speed up, we can use a list of indices and shuffle? No, we need max.
            # We can use a priority queue? 
            # But gains change as uncovered_mask changes.
            
            # Let's just scan.
            # Optimization: pre-filter blocks that are "promising"? Hard to do dynamically.
            
            # We can break early if we find a block that covers everything?
            # Max possible gain is uncovered_mask.bit_count().
            max_possible = uncovered_mask.bit_count()
            
            for i in range(n_blocks):
                inter = block_masks[i] & uncovered_mask
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        if best_gain == max_possible:
                            break # Can't do better
                    elif gain == best_gain and gain > 0:
                        # Random tie-breaking
                        if rng.random() < 0.1:
                            best_block = i
                            # Don't break, keep looking for strictly better? 
                            # Or just accept this one? 
                            # Standard greedy picks first max. 
                            # Randomized greedy might pick a different max.
                            # Let's just keep the first max found to be deterministic-ish 
                            # but rely on initial shuffle? No, we scan 0..N.
                            # Let's just pick the first one with max gain to be fast.
                            # The randomness comes from multiple restarts or SA.
                            pass
            
            if best_block == -1:
                break # Should not happen
                
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected

    # Run a few greedy restarts to find a good initial solution
    best_initial = None
    best_initial_len = float('inf')
    
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        rng = random.Random()
        sel = greedy_init(rng)
        if len(sel) < best_initial_len:
            best_initial_len = len(sel)
            best_initial = sel
            
    if best_initial is None:
        # Fallback: single greedy
        best_initial = greedy_init(random.Random())
        
    # 3. Simulated Annealing / Local Search
    # Objective: Minimize (number_of_blocks + penalty * uncovered)
    # We want to minimize block count.
    
    current_set = set(best_initial)
    
    # Coverage counts
    cover_count = [0] * n_triples
    for idx in current_set:
        for t_idx in block_triples[idx]:
            cover_count[t_idx] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    # Cost function:
    # If valid: cost = len(current_set)
    # If invalid: cost = len(current_set) + 1000 * uncovered_count
    # We want to drive uncovered_count to 0 first, then minimize size.
    
    def calc_cost():
        if uncovered_count == 0:
            return len(current_set)
        else:
            return len(current_set) + 1000 * uncovered_count

    current_cost = calc_cost()
    best_set = set(current_set)
    best_cost = current_cost
    
    # SA Parameters
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    all_indices = list(range(n_blocks))
    
    while time.time() < deadline - 2:
        if T < min_T:
            # Restart with higher T? Or just break.
            # Let's restart with a new T to escape local minima.
            T = 5.0
            # Maybe shuffle current_set? No, keep it.
            # Or re-initialize from best_set?
            # Let's just continue, T will cool down again.
            
        # Choose a move
        move_type = random.random()
        
        delta_cost = 0
        old_block = -1
        new_block = -1
        valid_move = True
        
        if move_type < 0.4:
            # Add a block
            new_block = random.choice(all_indices)
            if new_block in current_set:
                valid_move = False
            else:
                # Calculate delta
                for t_idx in block_triples[new_block]:
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
                        delta_cost -= 1000
                    cover_count[t_idx] += 1
                delta_cost += 1 # Size increases
                current_set.add(new_block)
                
        elif move_type < 0.8:
            # Remove a block
            if not current_set:
                valid_move = False
            else:
                old_block = random.choice(list(current_set))
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1
                        delta_cost += 1000
                delta_cost -= 1 # Size decreases
                current_set.remove(old_block)
                
        else:
            # Swap
            if not current_set:
                valid_move = False
            else:
                old_block = random.choice(list(current_set))
                new_block = random.choice(all_indices)
                if new_block in current_set:
                    valid_move = False
                else:
                    # Remove old
                    for t_idx in block_triples[old_block]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            uncovered_count += 1
                            delta_cost += 1000
                    # Add new
                    for t_idx in block_triples[new_block]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                            delta_cost -= 1000
                        cover_count[t_idx] += 1
                    # Size change: 0
                    delta_cost += 0
                    current_set.remove(old_block)
                    current_set.add(new_block)

        if not valid_move:
            continue
            
        new_cost = current_cost + delta_cost
        
        # Metropolis Acceptance
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
        else:
            # Reject: Undo
            if move_type < 0.4:
                # Undo Add
                current_set.remove(new_block)
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1
            elif move_type < 0.8:
                # Undo Remove
                current_set.add(old_block)
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
            else:
                # Undo Swap
                current_set.remove(new_block)
                current_set.add(old_block)
                # Re-add old coverage
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
                # Re-remove new coverage
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1

        T *= cooling_rate

    # Return best valid solution
    # Ensure best_set is valid? 
    # The cost function penalizes uncovered, so best_cost should correspond to a valid set 
    # if we found one. 
    # If best_cost still has penalty, we might return an invalid set.
    # We should check validity of best_set.
    
    # Verify best_set
    final_cover = [0] * n_triples
    for idx in best_set:
        for t_idx in block_triples[idx]:
            final_cover[t_idx] += 1
            
    if any(c == 0 for c in final_cover):
        # Fallback to current_set if it's valid, or greedy
        current_cover = [0] * n_triples
        for idx in current_set:
            for t_idx in block_triples[idx]:
                current_cover[t_idx] += 1
        if not any(c == 0 for c in current_cover):
            best_set = current_set
        else:
            # Return greedy
            best_set = set(best_initial)

    final_blocks = [list(all_blocks[i]) for i in best_set]
    return final_blocks
