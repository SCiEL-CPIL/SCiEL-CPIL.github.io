import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Helper to compute all sums a+b for a<=b in the set
    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    # Check if a set is a valid Sidon set
    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    # Check if a set is inclusion-maximal
    # Returns list of missing elements that can be added
    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if adding x creates a collision
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            # If any new sum is in existing sums, collision
            if new_sums & sums:
                continue
            
            # Internal collisions in new_sums are impossible if selected is distinct.
            extendable.append(x)
        return extendable

    # Objective function for minimization
    def objective(selected):
        if not selected:
            return float('inf')
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        violations = expected - len(sums)
        
        if violations > 0:
            return violations * 1000 + len(selected)
            
        extendable = get_extendable(selected, sums)
        non_max_count = len(extendable)
        
        if non_max_count > 0:
            return non_max_count * 100 + len(selected)
            
        return len(selected)

    def greedy_blocking(seed_set):
        """
        Start with a seed set, then iteratively add the element that 'blocks' the most other elements.
        'Blocking' an element y means that adding x to the current set makes it impossible to add y 
        without violating the Sidon property.
        We want to maximize the number of blocked elements to reach maximality with fewest additions.
        """
        current = list(seed_set)
        current_set = set(current)
        current_sums = get_sums(current)
        
        while True:
            # Check if maximal
            extendable = get_extendable(current, current_sums)
            if not extendable:
                break
            
            # For each candidate x in extendable, calculate how many other elements it blocks.
            # An element y is blocked by x if adding x prevents adding y.
            # This is complex to compute exactly for all pairs.
            # Approximation: Calculate the number of 'conflicts' x creates with the *potential* sums.
            # A simpler proxy: Add x, see how many elements from the *previous* extendable list are no longer extendable.
            
            best_candidate = None
            best_blocks = -1
            best_new_set = None
            best_new_sums = None
            
            # To save time, only evaluate a subset of candidates if there are many
            candidates = extendable
            if len(candidates) > 20:
                # Heuristic: Prefer candidates that are "central" or have high density?
                # Random sampling of top candidates might be better than sorting by value
                random.shuffle(candidates)
                candidates = candidates[:20]
            
            for x in candidates:
                new_set = current + [x]
                new_set.sort()
                
                # Check if valid (it is by definition of extendable)
                # Calculate new sums
                # Optimization: Only add x's sums
                new_sums = set(current_sums)
                new_sums.add(2 * x)
                for y in current:
                    new_sums.add(x + y)
                
                # How many elements are now blocked?
                # An element z is blocked if z was in `extendable` (or was extendable before x was added)
                # but is NOT in the new extendable list.
                # We don't need the full new extendable list, just check the ones we care about.
                # Actually, we want to minimize the *remaining* extendable count.
                # So we want to minimize len(get_extendable(new_set, new_sums)).
                
                # This is expensive. Let's estimate.
                # If x is added, any z such that (z + y) in new_sums or 2z in new_sums is blocked.
                # We can count how many z in [1..N] \ current_set are now invalid.
                
                # Fast check: 
                # A z is invalid if 2z in new_sums OR any (z+y) in new_sums for y in new_set.
                
                # We can just compute the new extendable count.
                # Since N=256, this loop is 256 * (size of new_set).
                # Size is small (<30). 256 * 30 = 7680 ops.
                # Candidates ~20. Total 150k ops per iteration.
                # Iterations ~20. Total 3M ops. Fast enough.
                
                new_extendable_count = 0
                new_set_set = set(new_set)
                for z in range(1, N + 1):
                    if z in new_set_set:
                        continue
                    # Check if z can be added to new_set
                    # Sums to check: 2z, z+y for y in new_set
                    # If any in new_sums, blocked.
                    blocked = False
                    if 2 * z in new_sums:
                        blocked = True
                    else:
                        for y in new_set:
                            if z + y in new_sums:
                                blocked = True
                                break
                    if not blocked:
                        new_extendable_count += 1
                
                blocks = len(extendable) - new_extendable_count
                # We want to maximize blocks (minimize remaining extendable)
                # Tie breaker: prefer smaller x? Or random?
                # To encourage diversity, maybe random tie break or smallest x
                if blocks > best_blocks:
                    best_blocks = blocks
                    best_candidate = x
                    best_new_set = new_set
                    best_new_sums = new_sums
            
            if best_candidate is None:
                break
            
            current = best_new_set
            current_set = set(current)
            current_sums = best_new_sums
            
            if time.time() > deadline:
                break
                
        return current

    best_set = None
    best_obj = float('inf')
    
    # Strategy: Try multiple random seeds, then apply greedy blocking.
    # Also try "empty" seed? No, must be non-empty.
    # Seeds: Small random Sidon sets.
    
    def random_valid_sidon(k_target=None):
        selected = []
        sums = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if k_target and len(selected) >= k_target:
                break
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) != len(added):
                continue
            if sums.intersection(added):
                continue
            selected.append(x)
            sums.update(added)
            
        return selected

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate a small random seed
        # Smaller seeds give more room for greedy to work, but might result in larger final sets?
        # Actually, we want to minimize final size.
        # If we start with a large seed, we might already be close to maximal?
        # Greedy blocking adds elements. So final size >= seed size.
        # So we should start with SMALL seeds to allow the greedy to pick the "most efficient" blockers.
        # But if the seed is too small, it might take many steps to become maximal.
        
        seed_size = random.randint(1, 5)
        seed = random_valid_sidon(seed_size)
        
        if not seed:
            continue
            
        result = greedy_blocking(seed)
        
        if not result:
            continue
            
        obj = objective(result)
        
        if obj < best_obj:
            best_obj = obj
            best_set = list(result)
            
            # If we found a very good one, maybe stop?
            # Theoretical lower bound ~16.
            if best_obj <= 18:
                # Could try to improve further, but let's see if we can get lower.
                # Don't break immediately, keep searching for a bit.
                pass

    if best_set is None:
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
        
    return best_set
# EVOLVE-BLOCK-END
