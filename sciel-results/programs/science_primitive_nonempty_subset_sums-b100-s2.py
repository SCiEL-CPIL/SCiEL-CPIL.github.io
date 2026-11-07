import random
import time

def _validate(elements):
    n = len(elements)
    if n == 0:
        return True
    
    sums = set()
    num_subsets = 1 << n
    for mask in range(1, num_subsets):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
        
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 85
    best_set = [31, 32, 34]
    best_size = 3
    
    if _validate(best_set):
        best_size = len(best_set)
        
    random.seed(42)
    
    # Try to find larger sets using greedy approach with random restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Start with a random large number
        start_val = random.randint(100, 256)
        current = [start_val]
        sums = {start_val}
        
        # Try to add more elements
        candidates = [x for x in range(1, 257) if x not in current and x != start_val]
        # Prefer larger numbers to avoid small divisors
        candidates.sort(reverse=True)
        random.shuffle(candidates[:50]) # Shuffle top 50 to mix
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
                
            # Generate new sums
            new_sums = set()
            valid = True
            for s in sums:
                ns = s + x
                if ns in sums:
                    valid = False
                    break
                new_sums.add(ns)
            new_sums.add(x)
            
            if not valid:
                continue
                
            # Check distinctness: new_sums should not intersect sums
            # (Already checked s+x not in sums, but x might be in sums? 
            # x is distinct from current elements, but could x be a sum of current elements?
            # If x is in sums, then x is a sum of a subset of current.
            # Then x divides x (trivial) but we need distinct VALUES.
            # If x is in sums, then x is already a sum.
            # The new sum x is the same value.
            # The problem says "Equal sum values are counted once".
            # "Two different subsets with the same sum still make the set invalid".
            # So if x is a sum of current, we have two subsets {x} and the subset summing to x.
            # So x must NOT be in sums.
            if x in sums:
                continue
                
            # Check divisibility between sums and new_sums
            # For each s in sums, s must not divide any ns in new_sums
            # And ns must not divide s (impossible if ns > s)
            # But x might be smaller than some sums?
            # If x < s, then x could divide s.
            
            for s in sums:
                if x != s: # distinct values
                    if s % x == 0:
                        valid = False
                        break
            if not valid:
                continue
                
            for ns in new_sums:
                if ns != x:
                    if x % ns == 0:
                        valid = False
                        break
            if not valid:
                continue
                
            for s in sums:
                for ns in new_sums:
                    if s != ns:
                        if ns > s:
                            if ns % s == 0:
                                valid = False
                                break
                        else:
                            if s % ns == 0:
                                valid = False
                                break
                if not valid:
                    break
            if not valid:
                continue
                
            # Check divisibility within new_sums
            new_sums_list = sorted(new_sums)
            for i in range(len(new_sums_list)):
                a = new_sums_list[i]
                for j in range(i + 1, len(new_sums_list)):
                    b = new_sums_list[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # If we reach here, x is valid
            current.append(x)
            sums.update(new_sums)
            
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    return list(best_set)
