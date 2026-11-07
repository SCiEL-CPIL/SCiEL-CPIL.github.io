"""
Strategy: Simulated Annealing on the set of elements, minimizing divisibility violations.
Instead of greedy construction which gets stuck in local optima, we start with a random set
and perform random moves (add, remove, swap) while accepting worse states with a probability
based on temperature. The objective is to minimize the number of divisibility conflicts
among subset sums.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _count_violations(sums, total):
    """Count how many pairs (a, b) in sums exist such that a < b and b % a == 0."""
    if not sums:
        return 0
    sums_list = sorted(sums)
    violations = 0
    for i, s in enumerate(sums_list):
        if s == 0:
            continue
        # Check multiples of s in the set
        # Optimization: only check if 2*s <= total
        if 2 * s > total:
            continue
        for j in range(i + 1, len(sums_list)):
            t = sums_list[j]
            if t > total:
                break
            if t % s == 0:
                violations += 1
    return violations

def _is_valid(sums, total):
    """Check if the set of sums has no divisibility relations."""
    return _count_violations(sums, total) == 0

def _get_violation_count(S):
    """Calculate the number of violations for a given set S."""
    sums = _subset_sums(S)
    total = sum(S)
    return _count_violations(sums, total)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    pool = list(range(1, 257))
    rng = random.Random(42)
    
    best_set = []
    best_score = float('inf') # Minimize violations
    
    # Initialize with a random set of medium-sized numbers to avoid small factor issues
    # Larger numbers generally have fewer small factors, reducing divisibility chances
    initial_size = 8
    initial_pool = pool[100:256]
    rng.shuffle(initial_pool)
    current_set = initial_pool[:initial_size]
    
    # Calculate initial state
    current_sums = _subset_sums(current_set)
    current_total = sum(current_set)
    current_violations = _count_violations(current_sums, current_total)
    
    if current_violations == 0:
        best_set = current_set[:]
        best_score = 0
    
    # Simulated Annealing Parameters
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will iterate until time runs out
    while time.time() < deadline:
        if T <= T_min:
            # Reheat if we get stuck or just restart with new random
            # Actually, let's just break and do a fresh random restart if T is too low
            # Or keep going with a very small T
            if T < 0.1:
                break
        
        # Choose a move: Add, Remove, or Swap
        move_type = rng.random()
        
        new_set = current_set[:]
        
        if move_type < 0.3 and len(current_set) < 30:
            # Add a random element
            available = [x for x in pool if x not in current_set]
            if not available:
                continue
            # Bias towards larger numbers
            available.sort(key=lambda x: -x)
            # Pick from top 50%
            half = len(available) // 2
            x = rng.choice(available[:half])
            new_set.append(x)
            
        elif move_type < 0.6 and len(current_set) > 1:
            # Remove a random element
            idx = rng.randrange(len(current_set))
            new_set.pop(idx)
            
        else:
            # Swap: Remove one, Add another
            if len(current_set) == 0:
                continue
            idx = rng.randrange(len(current_set))
            removed = new_set.pop(idx)
            
            available = [x for x in pool if x not in new_set]
            if not available:
                new_set.append(removed)
                continue
            available.sort(key=lambda x: -x)
            half = len(available) // 2
            x = rng.choice(available[:half])
            new_set.append(x)
        
        # Calculate new state
        new_sums = _subset_sums(new_set)
        new_total = sum(new_set)
        new_violations = _count_violations(new_sums, new_total)
        
        # Acceptance criterion
        delta = new_violations - current_violations
        
        if delta < 0:
            # Improvement
            current_set = new_set
            current_sums = new_sums
            current_total = new_total
            current_violations = new_violations
            
            if current_violations < best_score:
                best_score = current_violations
                best_set = current_set[:]
                
            # If we found a valid set, try to grow it?
            # SA will naturally try to add more if it reduces violations (which is impossible if already 0)
            # So if violations is 0, we should bias towards adding elements that keep it 0.
            # But SA handles this: adding an element that causes 0 violations is accepted (delta 0).
            # Adding an element that causes >0 violations is rejected (delta > 0, prob low).
            
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            # If delta is 0, always accept to allow exploration
            if delta == 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / T)
                
            if rng.random() < accept_prob:
                current_set = new_set
                current_sums = new_sums
                current_total = new_total
                current_violations = new_violations
        
        # Cool down
        T *= cooling_rate
        
        # If we have a valid set (violations == 0), we can try to be more aggressive with adding
        # by not cooling down as fast or by specifically targeting additions
        if current_violations == 0:
             # If valid, we want to maximize size.
             # Let's force a bias: if valid, try adding a random element that keeps it valid.
             # This is a local search step within SA.
             pass
             
    # Final check
    if best_score == 0:
        return [int(x) for x in best_set]
    
    # If no valid set found, return the best we have (even if invalid? No, task says invalid is 0 score)
    # But we should try to return a valid set if possible.
    # If best_score > 0, it means we never found a valid set.
    # Let's do a quick greedy repair on the best_set found
    # Actually, if we never found a valid set, returning an invalid one scores 0.
    # It is better to return a small valid set if we can find one quickly.
    
    # Fallback: Greedy construction
    greedy_set = []
    greedy_sums = set()
    greedy_total = 0
    # Try to build a small valid set
    for x in sorted(pool, reverse=True):
        if len(greedy_set) >= 10: break
        test_set = greedy_set + [x]
        test_sums = _subset_sums(test_set)
        test_total = sum(test_set)
        if _is_valid(test_sums, test_total):
            greedy_set = test_set
            greedy_sums = test_sums
            greedy_total = test_total
            
    if len(greedy_set) > 0:
        return [int(x) for x in greedy_set]
        
    # Last resort: return a single element (always valid)
    return [256]
# EVOLVE-BLOCK-END
