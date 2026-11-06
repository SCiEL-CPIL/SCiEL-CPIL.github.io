<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    
    # Greedy construction: build a 4-AP-free set by iterating through [1,512]
    # and adding each number if it doesn't complete a 4-term AP with already selected numbers.
    # This is the seed algorithm, which produces a valid 4-AP-free set.
    
    def greedy_construction():
        selected = set()
        for x in range(1, 513):
            # Check if adding x creates a 4-term AP
            # A 4-term AP: a, a+d, a+2d, a+3d
            # x could be:
            # 1. The last term: x-3d, x-2d, x-d, x -> need x-d, x-2d, x-3d in selected
            # 2. The third term: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in selected (but x+d > x, not yet selected)
            #    Actually since we process in order, x+d hasn't been considered yet
            # 3. The second term: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d (x+d, x+2d not yet selected)
            # 4. The first term: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d (none yet selected)
            # So only case 1 matters: check if there exists d>0 such that x-d, x-2d, x-3d all in selected
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected
    
    # Start with greedy
    selected = greedy_construction()
    best_set = set(selected)
    best_size = len(best_set)
    
    # Try to improve by local search: try to swap out elements to allow more to be added
    # Or try different orderings
    
    # Try reverse greedy: process from 512 down to 1
    def greedy_reverse():
        selected = set()
        for x in range(512, 0, -1):
            # Check if adding x creates a 4-term AP with selected elements
            # x could be first term: x, x+d, x+2d, x+3d where x+d, x+2d, x+3d in selected
            # x could be second term: x-d, x, x+d, x+2d where x-d, x+d, x+2d in selected
            # x could be third term: x-2d, x-d, x, x+d where x-2d, x-d, x+d in selected
            # x could be last term: x-3d, x-2d, x-d, x where x-3d, x-2d, x-d in selected
            valid = True
            # Case: x is first term
            for d in range(1, (512 - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is second term
            for d in range(1, x // 1 + 1):
                if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected and (x + 2*d) <= 512:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is third term
            for d in range(1, x // 2 + 1):
                if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected and (x + d) <= 512:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is last term
            for d in range(1, x // 3 + 1):
                if (x - 3*d) in selected and (x - 2*d) in selected and (x - d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected
    
    # Try random orderings
    import random
    
    def greedy_random_order():
        order = list(range(1, 513))
        random.shuffle(order)
        selected = set()
        for x in order:
            # Check all positions x could be in a 4-term AP
            valid = True
            # x is last term: need x-d, x-2d, x-3d in selected
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # x is first term: need x+d, x+2d, x+3d in selected
            for d in range(1, (512 - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # x is second term: need x-d, x+d, x+2d in selected
            for d in range(1, x + 1):
                if (x - d) >= 1 and (x + 2*d) <= 512:
                    if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                        valid = False
                        break
            if not valid:
                continue
            # x is third term: need x-2d, x-d, x+d in selected
            for d in range(1, x // 2 + 1):
                if (x - 2*d) >= 1 and (x + d) <= 512:
                    if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                        valid = False
                        break
            if valid:
                selected.add(x)
        return selected
    
    # Try reverse greedy
    rev_set = greedy_reverse()
    if len(rev_set) > best_size:
        best_set = set(rev_set)
        best_size = len(best_set)
    
    # Try several random orderings if time permits
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        rand_set = greedy_random_order()
        if len(rand_set) > best_size:
            best_set = set(rand_set)
            best_size = len(best_set)
        attempts += 1
        # Early stop if we've tried many and not improving
        if attempts > 200:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    N = 512
    
    # Precompute all 4-term APs for fast checking
    # For each element x, store the set of "conflicting triples" that would be completed
    # Actually, let's use a different approach: for each x, precompute which triples (a,b,c) 
    # would form a 4-AP with x
    
    # For fast checking, we'll use a boolean array and precomputed conflict lists
    # conflict[x] = list of (y, z, w) such that {x, y, z, w} forms a 4-AP
    # When we add x, we need to check if any y, z, w are all already selected
    
    # Better approach: use numpy for vectorized operations where possible
    
    def is_valid_addition(x, selected_mask):
        """Check if x can be added to selected_mask without creating a 4-AP."""
        # x as last term: need x-d, x-2d, x-3d in selected
        for d in range(1, (x - 1) // 3 + 1):
            if selected_mask[x-d] and selected_mask[x-2*d] and selected_mask[x-3*d]:
                return False
        # x as first term: need x+d, x+2d, x+3d in selected
        for d in range(1, (N - x) // 3 + 1):
            if selected_mask[x+d] and selected_mask[x+2*d] and selected_mask[x+3*d]:
                return False
        # x as second term: need x-d, x+d, x+2d in selected
        for d in range(1, min(x, N - 2*x) + 1):
            if d >= 1 and x - d >= 1 and x + 2*d <= N:
                if selected_mask[x-d] and selected_mask[x+d] and selected_mask[x+2*d]:
                    return False
        # x as third term: need x-2d, x-d, x+d in selected
        for d in range(1, min(x // 2, (N - x) // 1) + 1):
            if x - 2*d >= 1 and x + d <= N:
                if selected_mask[x-2*d] and selected_mask[x-d] and selected_mask[x+d]:
                    return False
        return True
    
    def greedy_with_order(order):
        """Run greedy algorithm with given order."""
        selected_mask = np.zeros(N + 1, dtype=bool)
        selected_count = 0
        for x in order:
            if is_valid_addition(x, selected_mask):
                selected_mask[x] = True
                selected_count += 1
        return selected_mask, selected_count
    
    # Deterministic greedy (ascending)
    order_asc = list(range(1, N + 1))
    mask_asc, size_asc = greedy_with_order(order_asc)
    best_mask = mask_asc.copy()
    best_size = size_asc
    
    # Deterministic greedy (descending)
    order_desc = list(range(N, 0, -1))
    mask_desc, size_desc = greedy_with_order(order_desc)
    if size_desc > best_size:
        best_mask = mask_desc.copy()
        best_size = size_desc
    
    # Try various random orderings
    rng = random.Random(42)
    attempts = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        # Generate different random orders
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        mask, size = greedy_with_order(order)
        
        if size > best_size:
            best_mask = mask.copy()
            best_size = size
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        attempts += 1
        
        # Also try some structured random orders
        if attempts % 10 == 0:
            # Try a "mixed" order: interleave low and high numbers
            low = list(range(1, N // 2 + 1))
            high = list(range(N // 2 + 1, N + 1))
            rng.shuffle(low)
            rng.shuffle(high)
            mixed = []
            for i in range(max(len(low), len(high))):
                if i < len(low):
                    mixed.append(low[i])
                if i < len(high):
                    mixed.append(high[i])
            mask_m, size_m = greedy_with_order(mixed)
            if size_m > best_size:
                best_mask = mask_m.copy()
                best_size = size_m
                no_improve_count = 0
        
        if no_improve_count >= max_no_improve:
            break
    
    # Try local improvement: try to add elements by removing others
    # This is a swap-based local search
    if time.time() < deadline - 10:
        # Try to improve by trying to add each excluded element
        # and removing minimal conflicting elements
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            excluded = [x for x in range(1, N + 1) if not best_mask[x]]
            rng.shuffle(excluded)
            
            for x in excluded:
                if time.time() > deadline - 10:
                    break
                
                # Try to add x by removing conflicting elements
                # Find all 4-APs that would be formed by adding x
                conflicting = set()
                
                # x as last term
                for d in range(1, (x - 1) // 3 + 1):
                    if best_mask[x-d] and best_mask[x-2*d] and best_mask[x-3*d]:
                        conflicting.add((x-d, x-2*d, x-3*d))
                
                # x as first term
                for d in range(1, (N - x) // 3 + 1):
                    if best_mask[x+d] and best_mask[x+2*d] and best_mask[x+3*d]:
                        conflicting.add((x+d, x+2*d, x+3*d))
                
                # x as second term
                for d in range(1, min(x, N - 2*x) + 1):
                    if d >= 1 and x - d >= 1 and x + 2*d <= N:
                        if best_mask[x-d] and best_mask[x+d] and best_mask[x+2*d]:
                            conflicting.add((x-d, x+d, x+2*d))
                
                # x as third term
                for d in range(1, min(x // 2, (N - x) // 1) + 1):
                    if x - 2*d >= 1 and x + d <= N:
                        if best_mask[x-2*d] and best_mask[x-d] and best_mask[x+d]:
                            conflicting.add((x-2*d, x-d, x+d))
                
                if not conflicting:
                    # Can add x directly
                    best_mask[x] = True
                    best_size += 1
                    improved = True
                elif len(conflicting) <= 1:
                    # Only need to remove one element
                    for conflict in conflicting:
                        # Remove one element from the conflict
                        # Try removing each and see if it works
                        for elem in conflict:
                            if best_mask[elem]:
                                # Tentatively remove elem and add x
                                best_mask[elem] = False
                                best_mask[x] = True
                                # Check if this creates new conflicts
                                # For simplicity, just accept if size doesn't decrease
                                if best_size + 1 > best_size:
                                    best_size += 1
                                    improved = True
                                else:
                                    # Revert
                                    best_mask[x] = False
                                    best_mask[elem] = True
                                break
                        if improved:
                            break
    
    return sorted([x for x in range(1, N + 1) if best_mask[x]])
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
