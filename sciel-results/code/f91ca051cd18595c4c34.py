"""Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256]."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def has_3ap_with_x(s, x):
    """Check if adding x to set s creates a 3-AP. x should not be in s."""
    # x is middle: a + b = 2x for a, b in s, a != b
    two_x = 2 * x
    for a in s:
        b = two_x - a
        if b != a and b in s:
            return True
    # x is max: a, b, x => x = 2b - a => a = 2b - x. Need a in s, a < b.
    # x is min: x, a, b => b = 2a - x. Need a in s, a > x.
    # Combining: for any a in s, check if 2a - x is in s.
    # If 2a - x == x, then a = x, impossible.
    # If 2a - x == a, then a = x, impossible.
    # So we just check if 2a - x is in s and 2a - x != x (which is always true if x not in s)
    # Wait, if 2a-x is in s, does it form an AP with x?
    # Case 1: x, a, 2a-x. AP if a-x = (2a-x)-a = a-x. Yes.
    #   Valid if x, a, 2a-x are distinct.
    #   Distinctness: a!=x (true), 2a-x!=x => a!=x (true), 2a-x!=a => a!=x (true).
    #   So if 2a-x in s, we have a 3-AP {x, a, 2a-x}.
    #   Wait, is this correct?
    #   If x=1, a=2, 2a-x=3. AP: 1,2,3. Yes.
    #   If x=1, a=3, 2a-x=5. AP: 1,3,5. Yes.
    #   What if x is the max? e.g. 1, 2, 3. x=3. a=2. 2a-x = 4-3=1. 1 in s. AP: 3,2,1. Yes.
    #   So checking if 2a-x in s covers both x as min and x as max.
    for a in s:
        b = 2 * a - x
        if b in s:
            return True
    return False

def reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def is_3ap_free(selected_set):
    """Check if a set is 3-AP-free."""
    s = selected_set
    for x in s:
        if has_3ap_with_x(s - {x}, x):
            return False
    return True

def greedy_construct():
    """Greedy: include x if it doesn't create a 3-AP with previously included numbers."""
    selected = set()
    for x in range(1, N + 1):
        if not has_3ap_with_x(selected, x):
            selected.add(x)
    return selected

def local_search(initial_set, time_limit):
    """Local search to improve the 3-AP-free set."""
    start_time = time.time()
    current = set(initial_set)
    current_score = reciprocal_sum(current)
    best = set(current)
    best_score = current_score
    
    all_candidates = set(range(1, N + 1))
    
    # Try to add elements
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try adding each element not in current, sorted by value (smallest first)
        # Smallest x has largest 1/x, so prioritize those
        for x in sorted(all_candidates - current):
            if time.time() - start_time > time_limit:
                break
            if not has_3ap_with_x(current, x):
                current.add(x)
                current_score += Fraction(1, x)
                improved = True
                if current_score > best_score:
                    best = set(current)
                    best_score = current_score
        
        if not improved:
            break
    
    # Try swaps: remove one element, try to add another
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try removing each element, sorted by value (largest first, since 1/x is smallest)
        for x in sorted(current, reverse=True):
            if time.time() - start_time > time_limit:
                break
            
            reduced = current - {x}
            reduced_score = current_score - Fraction(1, x)
            
            # Try to add elements to reduced, sorted by value (smallest first)
            for y in sorted(all_candidates - reduced):
                if time.time() - start_time > time_limit:
                    break
                if y == x:
                    continue
                if not has_3ap_with_x(reduced, y):
                    new_score = reduced_score + Fraction(1, y)
                    if new_score > current_score:
                        current = reduced | {y}
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best = set(current)
                            best_score = current_score
                        break  # Found improvement, move to next x
        
        if not improved:
            break
    
    # Try 2-swap: remove two elements, try to add two others
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        items = sorted(current, reverse=True)  # Remove largest (lowest value) first
        
        for i in range(len(items)):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, len(items)):
                if time.time() - start_time > time_limit:
                    break
                
                rem1, rem2 = items[i], items[j]
                reduced = current - {rem1, rem2}
                reduced_score = current_score - Fraction(1, rem1) - Fraction(1, rem2)
                
                # Try to add two elements to reduced
                candidates = sorted(all_candidates - reduced)
                for y in candidates:
                    if time.time() - start_time > time_limit:
                        break
                    if y == rem1 or y == rem2:
                        continue
                    if not has_3ap_with_x(reduced, y):
                        temp_set = reduced | {y}
                        temp_score = reduced_score + Fraction(1, y)
                        
                        for z in candidates:
                            if time.time() - start_time > time_limit:
                                break
                            if z <= y:
                                continue
                            if z == rem1 or z == rem2:
                                continue
                            if not has_3ap_with_x(temp_set, z):
                                new_score = temp_score + Fraction(1, z)
                                if new_score > current_score:
                                    current = temp_set | {z}
                                    current_score = new_score
                                    improved = True
                                    if current_score > best_score:
                                        best = set(current)
                                        best_score = current_score
                                    break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    greedy_set = greedy_construct()
    greedy_score = reciprocal_sum(greedy_set)
    
    # Phase 2: Local search
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 2.0:
        improved_set = local_search(greedy_set, remaining_time - 1.0)
        improved_score = reciprocal_sum(improved_set)
        if improved_score > greedy_score:
            greedy_set = improved_set
            greedy_score = improved_score
    
    # Convert to sorted list
    result = sorted(greedy_set)
    return result
# EVOLVE-BLOCK-END
