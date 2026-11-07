import time
import random
import math

# Precompute divisors for numbers up to 3000
# Max sum for 12 elements of 256 is 3072.
MAX_SUM = 3100
divisors = [[] for _ in range(MAX_SUM + 1)]
for i in range(1, MAX_SUM + 1):
    for j in range(i, MAX_SUM + 1, i):
        divisors[j].append(i)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Candidates: 1 to 256. Exclude 1 because sum 1 divides everything.
    # Actually, if the set is {1}, sum is 1. No other sums. Valid?
    # "No two distinct resulting values may divide one another."
    # If only one value, condition is vacuously true.
    # But we want to maximize size.
    # If we include 1, any other sum S must not be divisible by 1. Impossible.
    # So 1 cannot be in any set of size > 1.
    # Since we want max size, we exclude 1.
    candidates = list(range(2, 257))
    
    best_set = []
    best_size = 0

    def get_subset_sums(elements):
        """Returns set of subset sums or None if duplicates exist."""
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    return None
                new_sums.add(ns)
            if x in sums or x in new_sums:
                return None
            new_sums.add(x)
            sums = sums | new_sums
        return sums

    def check_divisibility(sums):
        """Check if no two distinct sums divide each other."""
        s_list = list(sums)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                if a < b:
                    if b % a == 0:
                        return False
                else:
                    if a % b == 0:
                        return False
        return True

    def is_valid(elements):
        sums = get_subset_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    def can_add(current_sums, cand):
        """
        Check if adding cand to a set with sums 'current_sums' is valid.
        Returns (bool, new_sums_set)
        """
        # Generate new sums
        new_sums = set()
        for s in current_sums:
            ns = s + cand
            if ns in current_sums or ns in new_sums:
                return False, None
            new_sums.add(ns)
        
        if cand in current_sums or cand in new_sums:
            return False, None
        
        new_sums.add(cand)
        
        # Check divisibility conflicts
        # 1. New sums vs Old sums
        # For each new sum, check if any old sum divides it, or it divides any old sum.
        
        # Optimization: 
        # a) Check if any old sum divides new sum.
        for ns in new_sums:
            # Check if any d in divisors[ns] is in current_sums
            # Note: divisors[ns] includes ns itself, but ns is not in current_sums (checked above)
            for d in divisors[ns]:
                if d in current_sums:
                    return False, None
            
            # Check if ns divides any old sum
            # Iterate through current_sums? 
            # If current_sums is large, this is slow.
            # But we can break early.
            # Also, if ns > max(current_sums), it can't divide any old sum.
            max_old = max(current_sums) if current_sums else 0
            if ns <= max_old:
                for os in current_sums:
                    if os % ns == 0:
                        return False, None
        
        # 2. New sums vs New sums
        ns_list = list(new_sums)
        n_ns = len(ns_list)
        for i in range(n_ns):
            a = ns_list[i]
            for j in range(i + 1, n_ns):
                b = ns_list[j]
                if a < b:
                    if b % a == 0:
                        return False, None
                else:
                    if a % b == 0:
                        return False, None

        return True, current_sums | new_sums

    # Randomized Greedy Search
    random.seed(42)
    
    # Try to find a large set
    # We will run multiple iterations
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle candidates
        shuffled = list(candidates)
        random.shuffle(shuffled)
        
        current_set = []
        current_sums = set()
        
        for cand in shuffled:
            if time.time() > deadline - 2.0:
                break
            valid, new_sums = can_add(current_sums, cand)
            if valid:
                current_set.append(cand)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            # If we found a large set, we can try to improve it locally
            if best_size >= 6:
                # Optional: break early if we hit a very high size?
                # Or continue to see if we can do better.
                pass

    # Local Search on the best set found
    if best_set and time.time() < deadline - 10.0:
        # Try to add more elements to best_set
        in_set = set(best_set)
        remaining = [c for c in candidates if c not in in_set]
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for cand in remaining:
                if cand in in_set:
                    continue
                valid, new_sums = can_add(get_subset_sums(best_set), cand)
                if valid:
                    best_set.append(cand)
                    in_set.add(cand)
                    remaining.remove(cand)
                    improved = True
                    break # Restart loop with new state
    
    # Final verification
    if best_set:
        if is_valid(best_set):
            return [int(x) for x in best_set]
    
    # Fallback
    return [31, 32, 34]
