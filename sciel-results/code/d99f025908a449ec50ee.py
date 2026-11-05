import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    max_val = 284
    
    # Precompute all possible sums? No, max sum is 284+284=568.
    # We can use a set or a bitmask for sums.
    # Bitmask is faster for checking collisions.
    
    def is_valid_addition(x, current_mask, current_elements):
        # Check x + x
        if (current_mask >> (x + x)) & 1:
            return False
        # Check x + y for y in current_elements
        for y in current_elements:
            if (current_mask >> (x + y)) & 1:
                return False
        return True

    def get_new_mask(x, current_mask, current_elements):
        new_mask = current_mask | (1 << (x + x))
        for y in current_elements:
            new_mask |= (1 << (x + y))
        return new_mask

    def greedy_build(seed):
        random.seed(seed)
        current_elements = list(fixed)
        current_mask = 0
        for i in range(len(current_elements)):
            for j in range(i, len(current_elements)):
                current_mask |= (1 << (current_elements[i] + current_elements[j]))
        
        candidates = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, current_mask, current_elements):
                current_elements.append(x)
                current_mask = get_new_mask(x, current_mask, current_elements[:-1])
        
        return current_elements, current_mask

    def local_search(initial_elements, initial_mask, seed):
        random.seed(seed)
        best_elements = list(initial_elements)
        best_mask = initial_mask
        best_size = len(initial_elements)
        
        current_elements = list(initial_elements)
        current_mask = initial_mask
        
        # We will iterate until time runs out or we stop improving
        # A simple strategy: 
        # 1. Try to add any valid element.
        # 2. If none, remove one element and restart adding.
        
        while time.time() < deadline - 5:
            # Phase 1: Try to add elements
            added_any = False
            # We need to check all candidates not in current_elements
            # To be efficient, we can maintain a list of candidates
            # But checking all 280 candidates is fast enough (280 * 20 ops = 5600 ops)
            
            # Create a list of candidates to try
            # We can randomize the order to avoid bias
            candidates_to_try = [x for x in range(1, max_val + 1) if x not in set(current_elements)]
            random.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                if time.time() > deadline - 5:
                    break
                if is_valid_addition(x, current_mask, current_elements):
                    current_elements.append(x)
                    current_mask = get_new_mask(x, current_mask, current_elements[:-1])
                    added_any = True
                    # Update best if we found a larger set
                    if len(current_elements) > best_size:
                        best_size = len(current_elements)
                        best_elements = list(current_elements)
                        best_mask = current_mask
            
            if not added_any:
                # We are stuck in a local maximum.
                # Perform a "kick": remove an element.
                # Which element to remove?
                # Heuristic: Remove an element that blocks the most other elements.
                
                # Calculate "blocking score" for each non-fixed element
                # For each element y in current_elements (not fixed):
                #   For each candidate x not in current_elements:
                #     If x is invalid ONLY because of y (and maybe x+x), then y blocks x.
                #     Actually, if x is invalid because of collision with y, removing y might make it valid.
                
                # This is O(N * M * N) which might be slow.
                # N ~ 20, M ~ 260. 20 * 260 * 20 = 104,000 ops. This is fine.
                
                block_scores = {y: 0 for y in current_elements if y not in fixed_set}
                
                # We need to check which elements are blocking which candidates.
                # A candidate x is blocked if there exists a collision.
                # If we remove y, does x become valid?
                # x is valid if no sums collide.
                # Collision sums for x: {x+x, x+y1, x+y2, ...}
                # If we remove y, the set of sums in current_mask reduces.
                
                # Simpler heuristic: Random removal of a non-fixed element.
                # Or: Remove the element with the largest value? No.
                # Let's try random removal first, it's simple and often works well with enough restarts.
                # To make it slightly smarter, we can prefer removing elements that are "central" or have high degree.
                
                removable = [y for y in current_elements if y not in fixed_set]
                if not removable:
                    break # Can't remove fixed elements, and if we only have fixed, we are stuck.
                
                # Let's pick a random removable element
                y_to_remove = random.choice(removable)
                
                # Remove y_to_remove
                current_elements.remove(y_to_remove)
                
                # Recalculate mask
                current_mask = 0
                for i in range(len(current_elements)):
                    for j in range(i, len(current_elements)):
                        current_mask |= (1 << (current_elements[i] + current_elements[j]))
                        
        return best_elements, best_mask

    # Strategy:
    # 1. Run several greedy builds to get good starting points.
    # 2. Run local search from the best greedy results.
    
    best_overall = list(fixed)
    best_mask_overall = 0
    for i in range(len(fixed)):
        for j in range(i, len(fixed)):
            best_mask_overall |= (1 << (fixed[i] + fixed[j]))
    
    # Run multiple greedy restarts
    greedy_results = []
    for i in range(50):
        if time.time() > deadline - 20:
            break
        elems, mask = greedy_build(i)
        if len(elems) > len(best_overall):
            best_overall = elems
            best_mask_overall = mask
        greedy_results.append((elems, mask))
        
    # Sort greedy results by size
    greedy_results.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Run local search on top greedy results
    # We have about 20s left for local search
    ls_budget = deadline - time.time() - 5
    if ls_budget > 5:
        # Run local search on the best few greedy solutions
        for elems, mask in greedy_results[:10]:
            if time.time() > deadline - 5:
                break
            # Allocate some time to each LS run
            # Or just let it run until deadline?
            # Better to have a sub-deadline for each LS run to allow multiple attempts?
            # Local search can be fast.
            
            # We'll just run one long local search on the best greedy solution
            # and maybe a few shorter ones on others?
            # Let's just run local search on the single best greedy solution found so far.
            # If we have time, we can restart local search with different random seeds.
            
            pass

    # Actually, let's just run local search on the best greedy solution found.
    # And if we have time, run it again with different seeds?
    # The local_search function above runs until deadline.
    # So we should only call it once, or manage time internally.
    
    # Let's modify the strategy:
    # 1. Greedy to find best start.
    # 2. Call local_search which runs until deadline.
    
    if greedy_results:
        best_elems, best_mask = greedy_results[0]
        # Run local search
        final_elems, final_mask = local_search(best_elems, best_mask, 12345)
        
        if len(final_elems) > len(best_overall):
            best_overall = final_elems
    
    return best_overall
