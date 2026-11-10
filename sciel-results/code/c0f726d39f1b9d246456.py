"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80  # 80s safety margin (90s kill limit)
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Returns True if element is the mean of some subset of others (size >= 2)."""
        n = len(others)
        if n < 2:
            return False
        target = element
        
        # Sort others for quick bound checks
        sorted_others = sorted(others)
        prefix_sums = np.cumsum(sorted_others)
        
        for k in range(2, n + 1):
            target_sum = k * target
            
            # Quick bounds check
            min_possible = prefix_sums[k - 1]
            max_possible = prefix_sums[-1] - (prefix_sums[-k - 1] if k < n else 0)
            if target_sum < min_possible or target_sum > max_possible:
                continue
            
            # DP: achievable sums with exactly j elements
            # Use a set for each count, but with early termination
            sums_by_count = [set() for _ in range(k + 1)]
            sums_by_count[0].add(0)
            
            for val in sorted_others:
                for j in range(k, 0, -1):
                    if sums_by_count[j - 1]:
                        for s in list(sums_by_count[j - 1]):
                            new_s = s + val
                            if new_s <= target_sum:
                                sums_by_count[j].add(new_s)
                        # Early exit: if we already have the target sum
                        if target_sum in sums_by_count[j]:
                            return True
            
            if target_sum in sums_by_count[k]:
                return True
        
        return False
    
    def can_add_element(element, current_set):
        """Check if element can be added to current_set without violating the constraint."""
        # Check 1: is element the mean of any subset of current_set?
        if current_set and is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        # a is mean of subset of (current_set - {a}) + {element} that includes element
        # => exists subset T of current_set - {a} with |T| >= 1 such that sum(T) = (|T|+1)*a - element
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            n = len(others_without_a)
            if n == 0:
                continue
            
            # Sort for bounds
            sorted_others = sorted(others_without_a)
            prefix_sums = np.cumsum(sorted_others)
            
            for j in range(1, n + 1):
                target_sum = (j + 1) * a - element
                if target_sum <= 0:
                    continue
                
                # Bounds check
                min_possible = prefix_sums[j - 1]
                max_possible = prefix_sums[-1] - (prefix_sums[-j - 1] if j < n else 0)
                if target_sum < min_possible or target_sum > max_possible:
                    continue
                
                # DP check
                sums_by_count = [set() for _ in range(j + 1)]
                sums_by_count[0].add(0)
                
                for val in sorted_others:
                    for jj in range(j, 0, -1):
                        if sums_by_count[jj - 1]:
                            for s in list(sums_by_count[jj - 1]):
                                new_s = s + val
                                if new_s <= target_sum:
                                    sums_by_count[jj].add(new_s)
                            if target_sum in sums_by_count[jj]:
                                return False
                
                if target_sum in sums_by_count[j]:
                    return False
        
        return True
    
    # Strategy: build a large valid set using multiple orderings
    best_set = []
    
    # Define multiple orderings to try
    all_nums = list(range(1, 513))
    
    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        # Powers of 2 first (known to work well)
        [2**i for i in range(9) if 2**i <= 512] + [x for x in range(1, 513) if x not in [2**i for i in range(9) if 2**i <= 512]],
        # Odds first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        # Non-multiples of 3 first
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        # Large numbers first
        list(range(512, 256, -1)) + list(range(256, 0, -1)),
        # Primes first (heuristic)
        [x for x in range(1, 513) if all(x % i != 0 for i in range(2, int(x**0.5) + 1)) and x >= 2] + [x for x in range(1, 513) if not (all(x % i != 0 for i in range(2, int(x**0.5) + 1)) and x >= 2)],
    ]
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if time.time() > deadline - 10:
                break
            if x in selected_set:
                continue
            
            if can_add_element(x, selected):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Local improvement: try to add more elements
    if best_set and time.time() < deadline - 15:
        improved = best_set[:]
        improved_set = set(improved)
        remaining = [x for x in range(1, 513) if x not in improved_set]
        
        # Try adding remaining elements
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if can_add_element(x, improved):
                improved.append(x)
                improved_set.add(x)
        
        if len(improved) > len(best_set):
            best_set = improved[:]
    
    # Try swapping: remove one element, try to add multiple others
    if best_set and time.time() < deadline - 20:
        for i in range(len(best_set)):
            if time.time() > deadline - 10:
                break
            without_i = [x for j, x in enumerate(best_set) if j != i]
            without_i_set = set(without_i)
            remaining2 = [x for x in range(1, 513) if x not in without_i_set]
            
            # Try to add as many as possible from remaining2
            new_set = without_i[:]
            new_set_set = without_i_set.copy()
            for x in remaining2:
                if time.time() > deadline - 10:
                    break
                if can_add_element(x, new_set):
                    new_set.append(x)
                    new_set_set.add(x)
            
            if len(new_set) > len(best_set):
                best_set = new_set[:]
                break
    
    return best_set
# EVOLVE-BLOCK-END
