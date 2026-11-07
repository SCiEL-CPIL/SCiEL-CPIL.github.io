import time
import itertools

def check_validity(elements):
    n = len(elements)
    sums = set()
    # Generate all non-empty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
    
    sum_list = list(sums)
    for i in range(len(sum_list)):
        for j in range(i+1, len(sum_list)):
            a, b = sum_list[i], sum_list[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds to be safe
    
    # We want to maximize size.
    # Try sizes from 10 down to 1? Or build up?
    # Building up is harder to backtrack.
    # Let's try to find a valid set of size k.
    
    # Heuristic: Start with a base and add elements.
    # We know a1 cannot be 1.
    # Let's try to construct a superincreasing sequence.
    
    # Candidate elements must be <= 256.
    
    # Let's use a simple backtracking search.
    # Order elements ascending.
    # a_i > sum(a_1..a_{i-1})
    
    best_set = [2] # Fallback, size 1 is always valid? 
                   # Sums: {2}. No pairs. Valid.
                   # But we can do better. [2,3] -> sums {2,3,5}. 2!|3, 2!|5, 3!|5. Valid. Size 2.
                   
    # Let's try to find size 8, 9, etc.
    
    # To speed up, we can pre-filter candidates.
    
    def backtrack(current_set, current_sums, max_val):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if not current_set:
            return
            
        last = current_set[-1]
        min_next = sum(current_set) + 1
        
        # Try adding next element
        # We can prune if min_next > 256
        
        for x in range(min_next, 257):
            if time.time() > deadline:
                return
            
            # Check if x can be added
            # New sums are old_sums U (old_sums + x) U {x}
            # We need to check divisibility among all new sums.
            
            # Optimization:
            # If x is added, new sums are S' = S U (S+x) U {x}.
            # We need to ensure no s1, s2 in S' divide each other.
            
            # This is expensive.
            
            # Heuristic:
            # If x is a multiple of any s in S, then s | x. Fail.
            # If any s in S divides (s+x)? i.e., s | x. Same as above.
            # If any s in S divides (x + s') for s' in S?
            
            # Let's just do the full check for small sets.
            
            new_sums = set(current_sums)
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(s + x)
                
            # Check distinctness (guaranteed by superincreasing)
            # Check divisibility
            valid = True
            sl = list(new_sums)
            sl.sort()
            for i in range(len(sl)):
                for j in range(i+1, len(sl)):
                    a, b = sl[i], sl[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                backtrack(current_set + [x], new_sums, max_val)
                
    # Start search
    # We can start with different bases to find larger sets.
    # Let's try starting with small numbers.
    
    # Note: The search space is small if we enforce superincreasing.
    # a1 >= 2.
    # a2 > a1.
    # a3 > a1+a2.
    
    # Let's try a few starting points.
    # Actually, just run backtrack from empty set?
    # It might get stuck in local optima or take too long.
    
    # Let's try to guide the search.
    # We want large sets.
    
    # Try specific known good structures?
    # Powers of 2 are superincreasing.
    # {2, 4, 8, 16, 32, 64, 128}
    # Sums are all numbers from 2 to 254 with even parity? No.
    # Sums of {2,4,8...} are all even numbers from 2 to 254.
    # 2 divides 4. Fail.
    
    # So we need to break the divisibility.
    # If we pick primes?
    # {2, 3, 5, 11, 23, 47, 95, 191}
    # Let's check {2, 3, 5, 11, 23, 47, 95, 191}.
    # Sum of first 7: 2+3+5+11+23+47+95 = 186.
    # 191 > 186. Superincreasing.
    # Let's check divisibility for this set in code.
    
    test_set = [2, 3, 5, 11, 23, 47, 95, 191]
    if check_validity(test_set):
        best_set = test_set
        
    # Try to extend or find better via search
    # We'll run a limited search for size > len(best_set)
    
    # To make search effective, we should probably iterate over possible a1, a2...
    
    # Let's run a generic search with time limit
    # We'll try to find a set of size 9.
    
    # Start with a1=2
    # a2=3
    # a3=6 (failed before)
    # Let's let the code find it.
    
    # We'll call backtrack with some initial seeds
    # But backtrack as written above tries all x.
    
    # Let's just run the backtrack from [2]
    # And [3]
    # And [4]
    
    for start in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        initial_sums = {start}
        backtrack([start], initial_sums, 256)
        
    return best_set

construct()
