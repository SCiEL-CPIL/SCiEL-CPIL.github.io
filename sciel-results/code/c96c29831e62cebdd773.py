import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in all_triples ]
    num_triples = len(all_triples)
    
    # Precompute conflicts
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(i+1, num_triples):
            mj = triple_masks[j]
            inter = mi & mj
            if inter & (inter - 1):
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)

    # Precompute masks by size for exact alpha check
    masks_by_size = {k: [] for k in range(1, N+1)}
    for mask in range(1 << N):
        k = bin(mask).count('1')
        if k > 0:
            masks_by_size[k].append(mask)
            
    def get_active_masks(active_indices):
        return [triple_masks[i] for i in active_indices]

    def calc_greedy_alpha(active_masks):
        if not active_masks:
            return N
        points_mask = (1 << N) - 1
        # To make it deterministic-ish but fast, we shuffle the triples list copy?
        # No, just iterate.
        # We want to remove points to hit all triples.
        # This is a Set Cover / Hitting Set heuristic.
        # Greedy: Pick triple with most points in current set? 
        # Or just iterate and remove one point from any active triple.
        
        # Better heuristic:
        # While there is an active triple fully contained in points_mask:
        #   Pick one such triple.
        #   Remove a point from it.
        
        # To avoid bias, we can randomize the order of triples checked.
        # But creating a shuffled list every call is slow.
        # We can just iterate.
        
        changed = True
        # Limit iterations to prevent infinite loops (shouldn't happen)
        iter_count = 0
        while changed and iter_count < 100:
            changed = False
            iter_count += 1
            for tm in active_masks:
                if (points_mask & tm) == tm:
                    # Find a point to remove
                    # Prefer points that appear in many other active triples?
                    # That's expensive. Just pick random.
                    pts = [i for i in range(N) if (tm >> i) & 1]
                    p = random.choice(pts)
                    points_mask &= ~(1 << p)
                    changed = True
                    # break to restart with new state? 
                    # Or continue? Continuing is faster.
                    # If we continue, we might remove multiple points from the same triple? 
                    # No, once one point is removed, (points_mask & tm) != tm.
                    # So we won't hit the same triple again in this pass.
        return bin(points_mask).count('1')

    def calc_exact_alpha(active_masks, lower_bound=1):
        if not active_masks:
            return N
        # Check from N down to lower_bound
        # Actually, we know greedy gave us a lower bound LB.
        # We want to know if there is an independent set of size > LB.
        # So we check sizes N, N-1, ..., LB+1.
        # If we find one, return that size.
        # If we find none, return LB.
        
        # Optimization: If LB is close to N, check few sizes.
        # If LB is small, check many.
        
        # We can stop early if we find an independent set.
        for k in range(N, lower_bound, -1):
            for mask in masks_by_size[k]:
                is_ind = True
                for tm in active_masks:
                    if (tm & mask) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return lower_bound

    def is_valid_add(active_mask_int, idx):
        return (conflict_bits[idx] & active_mask_int) == 0

    def generate_initial():
        active_mask_int = 0
        active_indices = []
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if is_valid_add(active_mask_int, idx):
                active_mask_int |= (1 << idx)
                active_indices.append(idx)
        return active_indices, active_mask_int

    best_solution_indices = []
    best_exact_alpha = N + 1
    best_greedy_alpha = N + 1
    
    # Initial
    curr_indices, curr_mask_int = generate_initial()
    curr_masks = get_active_masks(curr_indices)
    curr_g_alpha = calc_greedy_alpha(curr_masks)
    
    # Check exact for initial
    curr_exact = calc_exact_alpha(curr_masks, curr_g_alpha)
    if curr_exact < best_exact_alpha:
        best_exact_alpha = curr_exact
        best_greedy_alpha = curr_g_alpha
        best_solution_indices = list(curr_indices)
        
    if best_exact_alpha <= 5:
        return [list(all_triples[i]) for i in best_solution_indices]

    temperature = 1.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Generate neighbor
        # Strategy: Swap one triple
        if curr_indices:
            remove_idx_pos = random.randint(0, len(curr_indices) - 1)
            remove_idx = curr_indices[remove_idx_pos]
            
            # Find a candidate to add
            # Pick random inactive triple
            # We need to check if it conflicts with (curr_mask_int without remove_idx)
            
            # Try a few random candidates
            added = False
            for _ in range(20): # Try 20 random swaps
                add_idx = random.randint(0, num_triples - 1)
                if add_idx in curr_indices:
                    continue
                # Check conflict with current set minus remove_idx
                # new_mask_int = (curr_mask_int & ~(1 << remove_idx))
                # if conflict_bits[add_idx] & new_mask_int == 0:
                #    valid
                
                new_mask_int = curr_mask_int & ~(1 << remove_idx)
                if (conflict_bits[add_idx] & new_mask_int) == 0:
                    # Valid swap
                    new_indices = curr_indices.copy()
                    new_indices[remove_idx_pos] = add_idx
                    new_mask_int = new_mask_int | (1 << add_idx)
                    
                    new_masks = get_active_masks(new_indices)
                    new_g_alpha = calc_greedy_alpha(new_masks)
                    
                    # Acceptance
                    delta = new_g_alpha - curr_g_alpha
                    if delta < 0 or (temperature > 0.01 and random.random() < pow(2.71828, -delta / temperature)):
                        curr_indices = new_indices
                        curr_mask_int = new_mask_int
                        curr_masks = new_masks
                        curr_g_alpha = new_g_alpha
                        added = True
                        break
                        
            if not added:
                # Just remove? Or skip?
                # Let's try just removing to reduce density if stuck
                # Or just skip this step
                pass
        else:
            # Add a random triple
            add_idx = random.randint(0, num_triples - 1)
            if is_valid_add(curr_mask_int, add_idx):
                curr_indices.append(add_idx)
                curr_mask_int |= (1 << add_idx)
                curr_masks = get_active_masks(curr_indices)
                curr_g_alpha = calc_greedy_alpha(curr_masks)

        # Check improvement
        if curr_g_alpha < best_greedy_alpha:
            best_greedy_alpha = curr_g_alpha
            no_improve_count = 0
            # Run exact check
            curr_exact = calc_exact_alpha(curr_masks, curr_g_alpha)
            if curr_exact < best_exact_alpha:
                best_exact_alpha = curr_exact
                best_solution_indices = list(curr_indices)
                if best_exact_alpha <= 5:
                    break
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart
                curr_indices, curr_mask_int = generate_initial()
                curr_masks = get_active_masks(curr_indices)
                curr_g_alpha = calc_greedy_alpha(curr_masks)
                no_improve_count = 0
                temperature = 1.0

        temperature *= cooling

    return [list(all_triples[i]) for i in best_solution_indices]
