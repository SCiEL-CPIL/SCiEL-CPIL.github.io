# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85
    
    def greedy_construct(order, sums_by_size, sums_excl):
        # sums_by_size: dict k -> set of sums of size k from current set
        # sums_excl: dict a -> dict k -> set of sums of size k from current set excluding a
        # This is memory heavy. 48 elements * 48 sizes * set of sums.
        # Max sum is 48*512 = 24576.
        # Sets of integers up to 24576.
        # 48 * 48 * (avg set size).
        # Might be too much memory or slow to copy.
        
        # Alternative: Just recompute `sums_excl` on the fly for Check 2?
        # No, that's what was slow.
        
        # Let's stick to the simpler, potentially slower but correct `can_form_sum` but optimize it heavily.
        # Or use bitsets.
        
        pass

    # Let's try a different angle: 
    # The constraint is "No element is the mean of any subset of at least two OTHER elements".
    # This is equivalent to: For all distinct x, y, z... in the set, x != (y+z+...)/k.
    # i.e., kx != y+z+...
    
    # A known construction for "sum-free" or "mean-free" sets is related to Sidon sets or similar.
    # But we just need to find a large one.
    
    # Let's optimize the `can_form_sum` with bitsets.
    # Since max sum is ~25k, we can use a Python integer as a bitset.
    # bit i is 1 if sum i is possible.
    
    def can_form_sum_bitset(lst, target, size):
        # dp[k] is an integer bitset where bit i is 1 if sum i is possible with k elements.
        # This is very fast in Python.
        if size == 0:
            return target == 0
        if size > len(lst):
            return False
            
        # Bounds check
        if size <= len(lst):
            min_sum = sum(sorted(lst)[:size])
            max_sum = sum(sorted(lst)[-size:])
            if target < min_sum or target > max_sum:
                return False

        dp = [0] * (size + 1)
        dp[0] = 1  # bit 0 is set
        
        for val in lst:
            # Shift and OR
            # We need to update dp[k] using dp[k-1]
            # Iterate backwards
            for k in range(size, 0, -1):
                # dp[k] |= dp[k-1] << val
                # But we only care if target bit is set.
                # We can mask to keep numbers small?
                # Max sum is ~25k, so integers are small.
                dp[k] |= (dp[k-1] << val)
                
                # Early exit
                if (dp[k] >> target) & 1:
                    return True
                    
        return (dp[size] >> target) & 1

    def is_valid_candidate(x, current_set, sums_by_size):
        # Check 1: x is not mean of subset of current_set
        n = len(current_set)
        for k in range(2, n + 1):
            if k * x in sums_by_size.get(k, set()):
                return False
        
        # Check 2: No a in current_set is mean of subset containing x
        for a in current_set:
            others = [y for y in current_set if y != a]
            for j in range(1, len(others) + 1):
                target_sum = (j + 1) * a - x
                if can_form_sum_bitset(others, target_sum, j):
                    return False
                    
        return True

    def greedy_construct(order):
        selected = []
        sums_by_size = {0: {0}}
        
        for x in order:
            if time.time() > deadline:
                break
            
            if is_valid_candidate(x, selected, sums_by_size):
                selected.append(x)
                # Update sums_by_size
                new_sums = {k: set() for k in range(len(selected) + 1)}
                new_sums[0] = {0}
                for k in range(1, len(selected) + 1):
                    if k in sums_by_size:
                        new_sums[k].update(sums_by_size[k])
                    if k-1 in sums_by_size:
                        new_sums[k].update(s + x for s in sums_by_size[k-1])
                sums_by_size = new_sums
                
        return selected

    candidates = list(range(1, 513))
    best_set = []
    best_size = 0
    
    # Try a few orders
    orders = [
        candidates,
        sorted(candidates, reverse=True),
    ]
    
    random.seed(42)
    for _ in range(10):
        s = candidates[:]
        random.shuffle(s)
        orders.append(s)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            if best_size >= 48:
                break
                
    while time.time() < deadline - 2:
        s = candidates[:]
        random.shuffle(s)
        res = greedy_construct(s)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
            if best_size >= 48:
                break

    return best_set
# EVOLVE-BLOCK-END
