import time
import random

def construct():
    start_time = time.time()
    
    # Max possible sum for 15 elements of 256 is 3840.
    # Let's precompute divisors up to 4000.
    MAX_SUM = 4000
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            divisors[j].append(i)
            
    def check_addition(current_sums_set, x):
        """
        Checks if x can be added to the set represented by current_sums_set.
        Returns True if valid, False otherwise.
        Also returns the new sums set if valid? No, just bool for speed in loop.
        """
        # New sums are {s + x for s in current_sums_set} U {x}
        # We need to check:
        # 1. No new sum divides an existing sum.
        # 2. No existing sum divides a new sum.
        # 3. No new sum divides another new sum.
        
        # Optimization: 
        # If x is in current_sums_set, invalid (distinct elements? No, x is element, sums are values. 
        # If x == s for some s, then subset {x} has sum x, and subset for s has sum s. 
        # They are distinct subsets (one is {x}, other is not {x} unless s=x and subset is {x}).
        # Wait, if s=x, then the subset sum x is already present. 
        # The rule says "Equal sum values are counted once... Two different subsets with the same sum still make the set invalid".
        # So if x is already a subset sum, adding x creates a new subset {x} with sum x, which equals the old sum x.
        # So x must NOT be in current_sums_set.
        
        if x in current_sums_set:
            return False
            
        new_sums = [x]
        for s in current_sums_set:
            new_sums.append(s + x)
            
        new_sums_set = set(new_sums)
        
        # Check 1 & 2: Interactions between new_sums and existing sums
        for s_new in new_sums:
            # Check if any divisor of s_new is in current_sums_set
            for d in divisors[s_new]:
                if d in current_sums_set:
                    return False
            
            # Check if any multiple of s_new is in current_sums_set
            # Multiples are k * s_new.
            # We can iterate k such that k*s_new <= MAX_SUM
            # Or just check if current_sums_set contains any multiple.
            # Since current_sums_set is a set, we can just check:
            # Is there any m in current_sums_set such that m % s_new == 0 and m != s_new?
            # This is O(|current_sums_set|).
            # Alternative: Precompute multiples?
            # Let's just iterate current_sums_set. It's small enough for greedy.
            for m in current_sums_set:
                if m != s_new and m % s_new == 0:
                    return False
        
        # Check 3: Interactions within new_sums
        # We already checked new_sums against current_sums.
        # Now check new_sums against each other.
        # Note: new_sums contains x and s+x.
        # If s1 + x divides s2 + x?
        # We can just check all pairs in new_sums_set.
        # But new_sums_set size is |current_sums_set| + 1.
        # For each s_new in new_sums_set:
        #   Check if any divisor of s_new is in new_sums_set (excluding itself)
        
        for s_new in new_sums_set:
            for d in divisors[s_new]:
                if d != s_new and d in new_sums_set:
                    return False
            # Check multiples
            for m in new_sums_set:
                if m != s_new and m % s_new == 0:
                    return False
                    
        return True

    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    # Orders
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        sorted(candidates, key=lambda x: x % 2), # Evens first?
        sorted(candidates, key=lambda x: x % 3),
    ]
    
    for _ in range(20):
        random.shuffle(candidates)
        orders.append(candidates[:])

    for order in orders:
        if time.time() - start_time > 50:
            break
            
        current_set = []
        sums_set = set()
        
        for x in order:
            if time.time() - start_time > 50:
                break
            if check_addition(sums_set, x):
                current_set.append(x)
                # Update sums_set
                new_sums = [x]
                for s in list(sums_set):
                    new_sums.append(s + x)
                sums_set.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Local Search: Try to improve best_set
    # Try to add one element
    # Try to remove one and add two
    
    # We'll do a few iterations of local search on the best set found
    improved = True
    while improved and time.time() - start_time < 55:
        improved = False
        current_set = best_set[:]
        sums_set = set()
        # Rebuild sums_set for current_set
        for x in current_set:
            new_sums = [x]
            for s in list(sums_set):
                new_sums.append(s + x)
            sums_set.update(new_sums)
            
        # Try to add a single element
        for x in range(1, 257):
            if x in current_set:
                continue
            if check_addition(sums_set, x):
                best_set = current_set + [x]
                best_size += 1
                improved = True
                break
                
        if improved:
            continue
            
        # Try to remove one and add two
        # This is O(N^3) roughly. N=256. 256^3 is too big.
        # Only try if best_size is small?
        if best_size < 10:
            for i in range(len(current_set)):
                if time.time() - start_time > 55:
                    break
                removed = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                
                # Rebuild sums for temp_set
                temp_sums = set()
                for x in temp_set:
                    new_sums = [x]
                    for s in list(temp_sums):
                        new_sums.append(s + x)
                    temp_sums.update(new_sums)
                    
                # Try to add two elements a, b
                # To speed up, only try a few pairs
                candidates_to_add = [x for x in range(1, 257) if x not in temp_set and x != removed]
                random.shuffle(candidates_to_add)
                
                found = False
                for a in candidates_to_add[:50]: # Limit candidates
                    if time.time() - start_time > 55:
                        break
                    # Check if a can be added
                    if not check_addition(temp_sums, a):
                        continue
                    
                    # Update temp_sums with a
                    new_sums_a = [a]
                    for s in list(temp_sums):
                        new_sums_a.append(s + a)
                    temp_sums_with_a = set(temp_sums)
                    temp_sums_with_a.update(new_sums_a)
                    
                    # Now try to add b
                    for b in candidates_to_add[:50]:
                        if b == a:
                            continue
                        if check_addition(temp_sums_with_a, b):
                            # Valid
                            new_set = temp_set + [a, b]
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(new_set)
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
                    
    return best_set
