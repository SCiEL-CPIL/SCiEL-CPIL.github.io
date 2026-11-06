import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MASK = (1 << 201) - 1
    TARGET = 200
    TARGET_BIT = 1 << TARGET
    
    def compute_reachable(elements):
        """Compute bitset of reachable sums for a set of elements."""
        reachable = 1
        for x in elements:
            reachable = (reachable | (reachable << x)) & MASK
        return reachable

    def is_valid(elements, reachable=None):
        """Check if subset summing to 200 exists."""
        if reachable is None:
            reachable = compute_reachable(elements)
        return not (reachable & TARGET_BIT)

    def greedy_construct(order):
        """Greedy construction given an order."""
        selected = []
        reachable = 1
        for x in order:
            new_reachable = (reachable | (reachable << x)) & MASK
            if not (new_reachable & TARGET_BIT):
                selected.append(x)
                reachable = new_reachable
        return selected

    # 1. Generate initial seeds using different greedy orderings
    best_set = []
    best_len = 0
    
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(2, 101, 2)) + list(range(1, 100, 2)),
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(2, 101, 2))
    ]
    
    # Add a few random orders
    for _ in range(5):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        candidate = greedy_construct(order)
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_set = candidate

    # If we didn't find anything (shouldn't happen), fallback
    if not best_set:
        best_set = [100]
        best_len = 1

    # 2. Local Search: Add, then Swap-to-Enable-Add
    # We represent the set as a list and a set for O(1) lookup
    current_set = set(best_set)
    current_list = list(best_set)
    current_reachable = compute_reachable(current_list)
    
    stuck_counter = 0
    max_stuck = 50  # If stuck for 50 full passes, try perturbation

    while time.time() < deadline:
        improved = False
        excluded = [x for x in range(1, 101) if x not in current_set]
        
        # Phase 1: Try to add any excluded element
        for x in excluded:
            if time.time() > deadline:
                break
            new_reachable = (current_reachable | (current_reachable << x)) & MASK
            if not (new_reachable & TARGET_BIT):
                # Can add x
                current_set.add(x)
                current_list.append(x)
                current_reachable = new_reachable
                improved = True
                # Restart exclusion list check as set changed
                excluded = [x for x in range(1, 101) if x not in current_set]
                # Break to restart outer loop logic with new state
                break
        
        if improved:
            stuck_counter = 0
            continue
        
        # Phase 2: Try swaps that enable an add
        # If we swap y out and x in, we get S'. 
        # If S' is valid, check if we can add z to S'.
        # If yes, we do Swap(y->x) and Add(z). Net gain +1.
        for x in excluded:
            if time.time() > deadline:
                break
            for y in list(current_list):
                if time.time() > deadline:
                    break
                if y == x:
                    continue
                
                # Create candidate set S' = (S - {y}) + {x}
                # We need to check if S' is valid.
                # Recomputing reachable for S' is expensive if done naively for every y.
                # Optimization: 
                # We can't easily "remove" y from bitset.
                # So we recompute.
                # To save time, we only recompute if we think it might be worth it?
                # No, just recompute. It's fast.
                
                # Optimization: Precompute reachable for S - {y}? 
                # We can do this once per y per pass? 
                # There are ~90 y's. 90 recomputes per pass is fine.
                
                # Let's optimize by computing reachable for S-y once per y.
                # But we are inside x loop. 
                # Let's restructure: 
                # For each y in S:
                #    R_y = reachable of S - {y}
                #    For each x in Excluded:
                #       R_yx = R_y | (R_y << x)
                #       if valid(R_yx):
                #           For each z in Excluded \ {x}:
                #               if valid(R_yx + z):
                #                   Swap y->x, Add z.
                
                pass # Placeholder, will implement below

        # Restructured Phase 2 for efficiency
        # Precompute reachable for S - {y} for all y in S? 
        # That's |S| recomputes. |S| ~ 90.
        # Then for each y, iterate x, then z.
        
        pass_found = False
        # We only do this if Phase 1 failed
        if not improved:
            # Compute reachable for S without each y
            # This is O(|S|^2 * cost_shift) if done naively per y.
            # Better: Just iterate y, compute R_y, then iterate x, z.
            
            for y in list(current_list):
                if time.time() > deadline:
                    break
                
                # Compute reachable of S \ {y}
                # We can do this by iterating all elements in S except y
                r_y = 1
                for z in current_list:
                    if z != y:
                        r_y = (r_y | (r_y << z)) & MASK
                
                # Now try adding x
                for x in excluded:
                    if time.time() > deadline:
                        break
                    r_yx = (r_y | (r_y << x)) & MASK
                    if r_yx & TARGET_BIT:
                        continue
                    
                    # S' = (S - {y}) + {x} is valid.
                    # Can we add z?
                    for z in excluded:
                        if z == x:
                            continue
                        r_yxz = (r_yx | (r_yx << z)) & MASK
                        if not (r_yxz & TARGET_BIT):
                            # Success! Swap y->x, Add z.
                            # Update state
                            current_set.remove(y)
                            current_set.add(x)
                            current_set.add(z)
                            
                            current_list.remove(y)
                            current_list.append(x)
                            current_list.append(z)
                            
                            current_reachable = r_yxz
                            improved = True
                            pass_found = True
                            break
                    if pass_found:
                        break
                if pass_found:
                    break
            
            if improved:
                stuck_counter = 0
                continue
        
        # If no improvement in Phase 1 or 2
        if not improved:
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perturbation: Remove a random element to escape local optimum
                if current_list:
                    # Remove a random element
                    idx = random.randint(0, len(current_list) - 1)
                    rem_elem = current_list.pop(idx)
                    current_set.remove(rem_elem)
                    
                    # Recompute reachable
                    current_reachable = compute_reachable(current_list)
                    stuck_counter = 0
                else:
                    # Should not happen, but if set is empty, break
                    break
            else:
                # If we are stuck but haven't perturbed yet, 
                # we might just be at a local optimum.
                # To avoid infinite loop of checking the same state,
                # we can break if we've checked everything and found nothing.
                # Since the state doesn't change, we will find nothing again.
                # So we should break or perturb immediately.
                # Let's perturb immediately if no improvement.
                if current_list:
                    idx = random.randint(0, len(current_list) - 1)
                    rem_elem = current_list.pop(idx)
                    current_set.remove(rem_elem)
                    current_reachable = compute_reachable(current_list)
                    stuck_counter = 0
                else:
                    break

    # Final verification
    if not is_valid(current_list):
        # Fallback to a known valid small set
        return [100]
        
    return current_list

