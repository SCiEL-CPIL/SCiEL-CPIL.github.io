# EVOLVE-BLOCK-START
import sys

def construct():
    max_val = 256
    best_set = []
    
    def check_divisibility(sums):
        sum_set = set(sums)
        max_sum = max(sum_set) if sum_set else 0
        
        for s in sum_set:
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += s
        return True

    def max_additional_elements(current_sum):
        """
        Estimate max number of additional elements we can add 
        while keeping the sum <= max_val, assuming superincreasing property.
        """
        count = 0
        s = current_sum
        while s < max_val:
            s = 2 * s + 1 # Next min element is s+1, new sum is s + (s+1) = 2s+1
            if s > max_val:
                break
            count += 1
        return count

    def backtrack(current_set, current_sums, current_sum):
        nonlocal best_set
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Pruning: if we can't beat the best size, return
        if len(current_set) + max_additional_elements(current_sum) <= len(best_set):
            return

        if current_sum >= max_val:
            return

        start = current_sum + 1
        # We can restrict the upper bound of 'a' to max_val
        for a in range(start, max_val + 1):
            # Generate new sums
            new_sums = set(current_sums)
            new_sums.add(a)
            for s in current_sums:
                new_sums.add(s + a)
            
            if check_divisibility(new_sums):
                backtrack(current_set + [a], new_sums, current_sum + a)
                
    backtrack([], set(), 0)
    
    return list(best_set)

# EVOLVE-BLOCK-END
