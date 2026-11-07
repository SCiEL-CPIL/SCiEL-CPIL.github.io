import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    best_set = []
    
    # Candidate pool: Focus on larger numbers where divisibility is less likely
    # Numbers in [128, 256) have fewer multiples within the range
    # Also include some from [96, 127) for diversity
    candidates_base = list(range(96, 257))
    
    max_restarts = 5000
    
    def check_valid_addition(current_sums, new_sums_list):
        """Check if adding new_sums to current_sums maintains the divisibility constraint."""
        if not new_sums_list:
            return True
        
        # Check for duplicates in new_sums
        if len(new_sums_list) != len(set(new_sums_list)):
            return False
        
        # Check new sums against old sums
        for n in new_sums_list:
            for o in current_sums:
                if n % o == 0 or o % n == 0:
                    return False
        
        # Check new sums against each other
        ns_sorted = sorted(new_sums_list)
        for i in range(len(ns_sorted)):
            a = ns_sorted[i]
            for j in range(i + 1, len(ns_sorted)):
                b = ns_sorted[j]
                if b % a == 0:
                    return False
        
        return True
    
    def greedy_build(candidates, order):
        """Build a set using greedy approach with given ordering."""
        current_set = []
        sums = set()
        
        for c in order:
            if time.time() - start_time > DEADLINE:
                break
            
            new_sums_list = [c] + [s + c for s in sums]
            
            if check_valid_addition(sums, new_sums_list):
                current_set.append(c)
                sums.update(new_sums_list)
        
        return current_set, sums
    
    def try_improve_swap(current_set, sums, candidates):
        """Try to improve by swapping one element with another."""
        if not current_set:
            return current_set, sums
        
        best_set_local = current_set.copy()
        best_sums_local = sums.copy()
        
        for i in range(len(current_set)):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try removing element i and adding a different one
            removed = current_set[i]
            remaining_set = current_set[:i] + current_set[i+1:]
            
            # Recompute sums without removed
            remaining_sums = set()
            for j in range(len(remaining_set)):
                if j == 0:
                    remaining_sums.add(remaining_set[j])
                else:
                    new_sums = [remaining_set[j]]
                    for s in remaining_sums:
                        new_sums.append(s + remaining_set[j])
                    remaining_sums.update(new_sums)
            
            # Try adding each candidate not in remaining_set
            for c in candidates:
                if c in remaining_set or c == removed:
                    continue
                
                new_sums_list = [c] + [s + c for s in remaining_sums]
                
                if check_valid_addition(remaining_sums, new_sums_list):
                    new_set = remaining_set + [c]
                    new_sums = remaining_sums.copy()
                    new_sums.update(new_sums_list)
                    
                    # Check if this is an improvement (same size, but different composition)
                    # For now, just accept if valid
                    if len(new_set) == len(best_set_local):
                        best_set_local = new_set
                        best_sums_local = new_sums
                        break
        
        return best_set_local, best_sums_local
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        candidates = candidates_base.copy()
        random.shuffle(candidates)
        
        # Try different orderings
        orderings = [
            candidates,                          # random
            sorted(candidates, reverse=True),    # largest first
            sorted(candidates),                  # smallest first
        ]
        
        for order in orderings:
            if time.time() - start_time > DEADLINE:
                break
            
            current_set, sums = greedy_build(candidates, order)
            
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
                
                # Try to improve with swaps
                if time.time() - start_time < DEADLINE - 5:
                    improved_set, _ = try_improve_swap(current_set, sums, candidates_base)
                    if len(improved_set) > len(best_set):
                        best_set = improved_set
    
    return [int(x) for x in best_set]
