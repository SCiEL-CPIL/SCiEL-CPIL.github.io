"""Optimized subset-sum divisibility search with Simulated Annealing / Random Restarts."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums(nums):
    """Generate all non-empty subset sums as a set."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sums

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        return False
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity_fast(nums):
    """Faster validity check with early termination."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        return False
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    min_sum = sums_list[0]
    if max_sum < 2 * min_sum:
        return True
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _try_add(current, candidate, deadline):
    """Try adding candidate to current set, return True if valid."""
    if time.time() > deadline:
        return False
    trial = current + [candidate]
    return _check_validity_fast(trial)

def _can_add_incremental(elements, x, current_sums):
    """Check if x can be added to elements without violating constraints.
    Returns (ok, new_sums) where new_sums is the updated set of subset sums."""
    # New sums are: current_sums ∪ {s + x for s in current_sums} ∪ {x}
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    
    # Check distinctness: number of sums must be 2^(len(elements)+1) - 1
    expected = (1 << (len(elements) + 1)) - 1
    if len(new_sums) != expected:
        return False, None
    
    # Check divisibility: no two distinct values divide each other
    sums_list = sorted(new_sums)
    n = len(sums_list)
    for i in range(n):
        si = sums_list[i]
        for j in range(i + 1, n):
            sj = sums_list[j]
            if sj % si == 0:
                return False, None
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with a known valid set
    best = [128, 131, 134, 137, 140, 143, 146]
    
    # Verify initial best
    if not _check_validity(best):
        best = [100, 103, 106]
    
    all_candidates = list(range(1, 257))
    
    # Helper: Greedily extend a set with available candidates using incremental sums
    def greedy_extend_incremental(current, candidates, deadline):
        """Try to add as many candidates as possible to current set using incremental sums."""
        current = current[:]
        if not current:
            current_sums = set()
        else:
            current_sums = _subset_sums(current)
        
        for num in candidates:
            if time.time() > deadline:
                break
            if num in current:
                continue
            ok, new_sums = _can_add_incremental(current, num, current_sums)
            if ok:
                current.append(num)
                current_sums = new_sums
        return current

    # Strategy 1: Initial Greedy Construction with incremental sums
    # Try multiple random orderings to find a good starting point
    for trial in range(80):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(trial * 7919)
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        
        current = []
        for num in candidates:
            if len(current) >= 15:  # Cap to save time
                break
            if _try_add(current, num, deadline):
                current.append(num)
        
        if len(current) > len(best):
            best = current[:]

    # Strategy 2: Simulated Annealing / Random Perturbation + Greedy Refinement
    # Use incremental sums for speed
    best_size = len(best)
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        if no_improve_count > 150:
            break
            
        rng = random.Random(int(time.time() * 1000) + len(best))
        
        # Create a perturbed version of the best set
        move_type = rng.random()
        
        perturbed = best[:]
        
        if move_type < 0.5 and len(perturbed) > 1:
            # Remove 1 or 2 elements
            num_remove = 1 if rng.random() < 0.7 else 2
            indices_to_remove = rng.sample(range(len(perturbed)), min(num_remove, len(perturbed)))
            for idx in sorted(indices_to_remove, reverse=True):
                perturbed.pop(idx)
        elif move_type < 0.8 and len(perturbed) > 0:
            # Swap one element
            idx = rng.randint(0, len(perturbed) - 1)
            old_val = perturbed[idx]
            new_val = rng.choice(all_candidates)
            if new_val not in perturbed:
                perturbed[idx] = new_val
        else:
            # Keep best set for pure greedy refinement with different candidate orders
            pass
        
        # Greedily extend the perturbed set using incremental sums
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        
        extended = greedy_extend_incremental(perturbed, candidates, deadline)
        
        if len(extended) > best_size:
            best = extended
            best_size = len(best)
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final validation
    if not _check_validity(best):
        # Fallback to a definitely valid small set
        best = [100, 101, 103]
        if not _check_validity(best):
            best = [100, 101]
    
    return list(best)
# EVOLVE-BLOCK-END
