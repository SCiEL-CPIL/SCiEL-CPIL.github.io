"""Optimized 3-AP-free set with maximum reciprocal weight."""
import time
import random
from fractions import Fraction

N = 256

def has_3ap(selected_set):
    """Check if a set contains any 3-term arithmetic progression."""
    for i, a in enumerate(selected_set):
        for b in selected_set[i+1:]:
            c = 2*b - a
            if c in selected_set:
                return True
    return False

def reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def reciprocal_sum_float(selected):
    """Compute float reciprocal sum for fast comparison."""
    return sum(1.0/x for x in selected)

def is_3ap_free(selected_set):
    """Check 3-AP freeness using set for O(1) lookup."""
    for a in selected_set:
        for b in selected_set:
            if a < b:
                c = 2*b - a
                if c in selected_set:
                    return False
    return True

def greedy_initial():
    """Greedy: add numbers in order of decreasing 1/x (i.e., ascending x) if no 3-AP formed."""
    selected = set()
    for x in range(1, N+1):
        # Check if adding x creates a 3-AP
        ok = True
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2*b - a
                    if c == x:
                        ok = False
                        break
                elif b < a:
                    c = 2*a - b
                    if c == x:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            selected.add(x)
    return selected

def local_search(selected_set, time_limit):
    """Local search with simulated annealing to maximize reciprocal sum."""
    start_time = time.time()
    best_set = frozenset(selected_set)
    best_val = reciprocal_sum_float(best_set)
    current_set = set(selected_set)
    current_val = best_val
    
    # Precompute: for each x, which pairs (a,b) would form a 3-AP with x
    # x is in a 3-AP if there exist a,b in set such that:
    #   x = a + d, a + 2d for some d (x is the end)
    #   x = a, a+d, a+2d (x is the start)  
    #   x = a-d, a, a+d (x is the middle)
    
    def can_add(x, s):
        """Check if x can be added to set s without creating a 3-AP."""
        # x as end: need a, a+d where 2(a+d) - a = x => a + 2d = x, d = x-a, so a + 2(x-a) = x => 2x - a = x => a = x... 
        # Let's just check all pairs
        for a in s:
            for b in s:
                if a == b:
                    continue
                # Check if {a, b, x} forms a 3-AP in some order
                # Sort the three
                triple = sorted([a, b, x])
                if triple[1] - triple[0] == triple[2] - triple[1] and triple[0] != triple[2]:
                    return False
        return True
    
    def can_remove(x, s):
        """Removing always maintains 3-AP freeness."""
        return True
    
    # Build a faster check: for a given x and set s, check if adding x creates a 3-AP
    # A 3-AP involving x: (x-2d, x-d, x), (x-d, x, x+d), (x, x+d, x+2d) for some d>=1
    def can_add_fast(x, s):
        """Check if x can be added without creating a 3-AP. O(|s|) per x."""
        for d in range(1, N):
            # x is the largest: (x-2d, x-d, x)
            if x - 2*d >= 1 and x - 2*d in s and x - d in s:
                return False
            # x is the middle: (x-d, x, x+d)
            if x - d >= 1 and x + d <= N and x - d in s and x + d in s:
                return False
            # x is the smallest: (x, x+d, x+2d)
            if x + 2*d <= N and x + d in s and x + 2*d in s:
                return False
        return True
    
    # Try moves: add, remove, swap
    all_numbers = set(range(1, N+1))
    
    for iteration in range(1000000):
        if time.time() - start_time > time_limit:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a number
            candidates = [x for x in all_numbers if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add_fast(x, current_set):
                new_set = current_set | {x}
                new_val = current_val + 1.0/x
                if new_val > current_val:
                    current_set = new_set
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_set = frozenset(current_set)
        
        elif move_type < 0.7:
            # Try to remove a number
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            new_val = current_val - 1.0/x
            # Only accept if it helps (it won't directly, but might enable better adds later)
            # Use simulated annealing: accept with probability exp(-(delta)/T)
            delta = new_val - current_val  # negative
            # Simple: only accept removals if we're stuck, or use low probability
            if random.random() < 0.05:  # low probability removal to explore
                current_set = new_set
                current_val = new_val
        
        else:
            # Try swap: remove x, add y
            if not current_set:
                continue
            x = random.choice(list(current_set))
            candidates = [y for y in all_numbers if y not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            # Check if removing x and adding y maintains 3-AP freeness
            temp_set = current_set - {x}
            if can_add_fast(y, temp_set):
                new_set = temp_set | {y}
                new_val = current_val - 1.0/x + 1.0/y
                if new_val > current_val:
                    current_set = new_set
                    current_val = new_val
                    if current_val > best_val:
                        best_val = current_val
                        best_set = frozenset(current_set)
                elif random.random() < 0.01:
                    current_set = new_set
                    current_val = new_val
    
    return set(best_set)

def construct():
    """Construct a 3-AP-free subset of [1,256] maximizing reciprocal sum."""
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    # Start with greedy solution
    current = greedy_initial()
    current_val = reciprocal_sum_float(current)
    best = frozenset(current)
    best_val = current_val
    
    # Multiple rounds of local search with different random seeds
    random.seed(42)
    
    while time.time() - start_time < deadline:
        # Run local search for a chunk of time
        chunk_time = min(5.0, deadline - (time.time() - start_time))
        if chunk_time < 1.0:
            break
        
        result = local_search(set(best), chunk_time)
        result_val = reciprocal_sum_float(result)
        
        if result_val > best_val:
            best_val = result_val
            best = frozenset(result)
        
        # Also try from the greedy solution with different perturbations
        if time.time() - start_time < deadline - 5.0:
            perturbed = set(best)
            # Remove a few random elements
            to_remove = random.sample(list(perturbed), min(3, len(perturbed)))
            for x in to_remove:
                perturbed.discard(x)
            result2 = local_search(perturbed, min(3.0, deadline - (time.time() - start_time)))
            result2_val = reciprocal_sum_float(result2)
            if result2_val > best_val:
                best_val = result2_val
                best = frozenset(result2)
    
    # Final polish: try all single adds/removes on the best set
    final_set = set(best)
    all_numbers = set(range(1, N+1))
    
    # Try adding any single number
    for x in sorted(all_numbers - final_set, key=lambda z: 1.0/z, reverse=True):
        if can_add_final(x, final_set):
            new_set = final_set | {x}
            if reciprocal_sum_float(new_set) > reciprocal_sum_float(final_set):
                final_set = new_set
    
    # Try removing any single number if it allows adding a better one
    for x in list(final_set):
        temp = final_set - {x}
        best_add = None
        best_gain = 0
        for y in sorted(all_numbers - temp, key=lambda z: 1.0/z, reverse=True):
            if can_add_final(y, temp):
                gain = 1.0/y - 1.0/x
                if gain > best_gain:
                    best_gain = gain
                    best_add = y
        if best_add is not None and best_gain > 0:
            final_set = temp | {best_add}
    
    return sorted(final_set)

def can_add_final(x, s):
    """Fast check if x can be added to set s without creating a 3-AP."""
    for d in range(1, N):
        if x - 2*d >= 1 and x - 2*d in s and x - d in s:
            return False
        if x - d >= 1 and x + d <= N and x - d in s and x + d in s:
            return False
        if x + 2*d <= N and x + d in s and x + 2*d in s:
            return False
    return True
