"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 75  # seconds; leave margin before 90s kill
    
    def is_3ap_free(set_list):
        """Check if a set has no 3-term AP."""
        s = set(set_list)
        for a in s:
            for b in s:
                if a < b:
                    c = 2*b - a
                    if c in s and c != a:
                        return False
        return True
    
    def reciprocal_weight(set_list):
        """Compute sum of reciprocals as float."""
        return sum(1.0/x for x in set_list)
    
    def can_add(candidate, current_set):
        """Check if adding candidate to current_set creates a 3-AP."""
        s = set(current_set)
        for a in s:
            # Check if candidate, a, c form AP: candidate + c = 2a => c = 2a - candidate
            c = 2*a - candidate
            if c != candidate and c in s:
                return False
            # Check if a, candidate, c form AP: a + c = 2*candidate => c = 2*candidate - a
            c = 2*candidate - a
            if c != candidate and c in s:
                return False
        return True
    
    # Strategy: Greedy from small to large, but with local search refinement
    # Key insight: small numbers have large reciprocal weight, so we want to include
    # as many small numbers as possible while maintaining 3-AP-free property.
    
    # Phase 1: Greedy construction - try to add each number from 1 to 256
    # This gives a valid 3-AP-free set
    selected = []
    selected_set = set()
    for x in range(1, 257):
        if time.time() - start_time > DEADLINE - 5:
            break
        if can_add(x, selected):
            selected.append(x)
            selected_set.add(x)
    
    if not selected:
        return [1]
    
    best_weight = reciprocal_weight(selected)
    best_set = list(selected)
    
    # Phase 2: Local search - try to swap out larger elements for smaller ones
    # that were skipped, or try different orderings
    # We'll do a hill-climbing search: try removing one element and adding another
    
    # First, let's try a smarter greedy: process in order of decreasing reciprocal weight
    # but with lookahead. Actually, let's try multiple greedy strategies.
    
    # Strategy A: Greedy from 1 to 256 (already done)
    # Strategy B: Greedy from 256 down to 1 (prioritizes large numbers - worse for reciprocal weight)
    # Strategy C: Try to include small numbers first, then fill in
    
    # Let's do a local search: for each element in the set, try removing it and 
    # adding a smaller number that was previously excluded
    
    # Build list of excluded numbers
    all_numbers = set(range(1, 257))
    excluded = all_numbers - selected_set
    
    # Try to improve by swapping
    improved = True
    max_iterations = 1000
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() - start_time > DEADLINE - 10:
            break
        
        current_weight = reciprocal_weight(selected)
        
        # Try removing each element and adding a smaller excluded element
        for i, elem in enumerate(selected):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Try removing elem
            new_selected = selected[:i] + selected[i+1:]
            new_selected_set = set(new_selected)
            
            # Try adding each excluded element smaller than elem
            for candidate in sorted(excluded):
                if candidate >= elem:
                    break
                if can_add(candidate, new_selected):
                    new_weight = current_weight - 1.0/elem + 1.0/candidate
                    if new_weight > best_weight + 1e-12:
                        selected = new_selected + [candidate]
                        selected.sort()
                        selected_set = set(selected)
                        excluded = all_numbers - selected_set
                        best_weight = new_weight
                        best_set = list(selected)
                        improved = True
                        break
    
    # Phase 3: Try a different initial greedy - start with the smallest possible set
    # and try to maximize density of small numbers
    
    # Let's try a randomized restart approach with time limit
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Randomized greedy: shuffle the order slightly
        numbers = list(range(1, 257))
        # Bias towards smaller numbers
        random.shuffle(numbers)
        numbers.sort(key=lambda x: x * random.random() ** 0.5)
        
        selected_r = []
        selected_r_set = set()
        for x in numbers:
            if can_add(x, selected_r):
                selected_r.append(x)
                selected_r_set.add(x)
        
        weight_r = reciprocal_weight(selected_r)
        if weight_r > best_weight + 1e-12:
            best_weight = weight_r
            best_set = list(selected_r)
    
    # Phase 4: Final local search on the best set found
    if time.time() - start_time < DEADLINE - 20:
        selected = list(best_set)
        selected_set = set(selected)
        all_numbers = set(range(1, 257))
        excluded = all_numbers - selected_set
        best_weight = reciprocal_weight(selected)
        
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            current_weight = reciprocal_weight(selected)
            
            for i, elem in enumerate(selected):
                new_selected = selected[:i] + selected[i+1:]
                new_selected_set = set(new_selected)
                
                for candidate in sorted(excluded):
                    if candidate >= elem:
                        break
                    if can_add(candidate, new_selected):
                        new_weight = current_weight - 1.0/elem + 1.0/candidate
                        if new_weight > best_weight + 1e-12:
                            selected = new_selected + [candidate]
                            selected.sort()
                            selected_set = set(selected)
                            excluded = all_numbers - selected_set
                            best_weight = new_weight
                            best_set = list(selected)
                            improved = True
                            break
            
            if not improved:
                break
    
    return best_set
# EVOLVE-BLOCK-END
