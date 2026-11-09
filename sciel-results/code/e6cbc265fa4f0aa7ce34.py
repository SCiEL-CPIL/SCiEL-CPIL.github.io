"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _check_divisibility(sums_set):
    """Check if any two sums in the set divide one another."""
    sums_list = sorted(sums_set)
    max_sum = sums_list[-1]
    for i, s in enumerate(sums_list):
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    return _check_divisibility(sums)

def _check_validity_fast(nums):
    """Fast validity check with early termination."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    return _check_divisibility(sums)

def _greedy_construct(candidates, max_size=12, deadline=None):
    """Greedy construction: add candidates one by one if valid."""
    current = []
    for num in candidates:
        if len(current) >= max_size:
            break
        if deadline and time.time() > deadline:
            break
        trial = current + [num]
        if _check_validity_fast(trial):
            current = trial
    return current

def _simulate_annealing(initial_set, all_candidates, deadline):
    """Simulated annealing to find larger valid sets."""
    current = initial_set[:]
    best = current[:]
    best_size = len(current)
    
    n_iter = 0
    max_iter = 5000
    
    # Cooling schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and n_iter < max_iter:
        n_iter += 1
        T = max(T * cooling, T_min)
        
        # Choose a random mutation
        mutation_type = random.random()
        
        if mutation_type < 0.4 and len(current) < 30:
            # Add a random element
            if current:
                available = [x for x in all_candidates if x not in current]
                if available:
                    new_elem = random.choice(available)
                    trial = current + [new_elem]
            else:
                trial = [random.choice(all_candidates)]
        elif mutation_type < 0.7:
            # Remove a random element
            if current:
                idx = random.randint(0, len(current) - 1)
                trial = current[:idx] + current[idx+1:]
            else:
                continue
        elif mutation_type < 0.9:
            # Swap an element
            if current:
                idx = random.randint(0, len(current) - 1)
                available = [x for x in all_candidates if x not in current]
                if available:
                    new_elem = random.choice(available)
                    trial = current[:]
                    trial[idx] = new_elem
                else:
                    continue
            else:
                continue
        else:
            # Replace two elements
            if len(current) >= 2:
                i, j = random.sample(range(len(current)), 2)
                available = [x for x in all_candidates if x not in current]
                if len(available) >= 2:
                    new_elems = random.sample(available, 2)
                    trial = current[:]
                    trial[i] = new_elems[0]
                    trial[j] = new_elems[1]
                else:
                    continue
            else:
                continue
        
        # Check validity
        if _check_validity_fast(trial):
            # Accept the move
            current = trial
            if len(current) > best_size:
                best = current[:]
                best_size = len(current)
        else:
            # Reject (for simplicity, no probabilistic acceptance of worse states
            # since we're optimizing for size, not a continuous function)
            pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    all_candidates = list(range(1, 257))
    
    # Phase 1: Greedy construction with multiple strategies
    best = [31, 32, 34]
    
    # Try different candidate orderings
    strategies = [
        sorted(all_candidates, reverse=True),  # Largest first
        sorted(all_candidates),                # Smallest first
        list(range(200, 257)) + list(range(100, 200)),  # High range
        list(range(150, 257)) + list(range(50, 150)),   # Mid-high
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 15:
            break
        result = _greedy_construct(strategy, max_size=15, deadline=deadline - 10)
        if len(result) > len(best):
            best = result
    
    # Phase 2: Simulated annealing from the best greedy result
    if time.time() < deadline - 5:
        sa_result = _simulate_annealing(best, all_candidates, deadline - 2)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Phase 3: Final local search - try to add one more element
    if time.time() < deadline - 2:
        for num in sorted(all_candidates, reverse=True):
            if num in best:
                continue
            trial = best + [num]
            if _check_validity_fast(trial):
                best = trial
                break
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
