import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = []
    best_size = 0
    
    # Candidate pool: 11..256
    candidates = list(range(11, 257))
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.shuffle(candidates)
        
        # State: current_set, current_sums (set of non-empty subset sums)
        current_set = []
        current_sums = set()
        
        def backtrack(idx):
            nonlocal best_set, best_size, current_set, current_sums
            
            if time.time() - start_time > time_limit:
                return
            
            # Update best
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
            # Try adding candidates
            for i in range(idx, len(candidates)):
                c = candidates[i]
                
                # Generate new sums
                # new_sums = current_sums U {c} U {s+c for s in current_sums}
                # Check for collisions
                # If any s+c is in current_sums or s+c == c (impossible as s>0) or s+c == s' + c (collision in new part)
                # Actually, just build the set and check size.
                
                # Optimization: 
                # The new elements are c and s+c for all s in current_sums.
                # If len(current_sums) + 1 + len(current_sums) != len(union), collision.
                
                # Let's compute the new sums set.
                # To be efficient, we can check:
                # 1. c not in current_sums
                # 2. (s+c) not in current_sums for any s
                # 3. (s1+c) != (s2+c) => s1 != s2 (guaranteed)
                # 4. c != s+c => s != 0 (guaranteed)
                # 5. c != s1+c1 and s2+c2? No, we just need the union size.
                
                # Fast collision check:
                # If c in current_sums: collision
                # If any (s+c) in current_sums: collision
                
                if c in current_sums:
                    continue
                    
                # Check if any s+c is in current_sums
                # This is O(N)
                collision = False
                for s in current_sums:
                    if (s + c) in current_sums:
                        collision = True
                        break
                        
                if collision:
                    continue
                    
                # If no collision, build new_sums
                # new_sums = current_sums | {c} | {s+c for s in current_sums}
                # We can create a new set.
                new_sums = set(current_sums)
                new_sums.add(c)
                for s in current_sums:
                    new_sums.add(s + c)
                    
                # Check divisibility in new_sums
                # This is the expensive part.
                # But we only do it if no collision.
                
                # Quick Divisibility Check:
                # We need to check if any x in new_sums divides any y in new_sums.
                # We can sort new_sums and check multiples.
                # To speed up, we can check only the new elements against all, and new against new.
                # But new elements are c and s+c.
                # It's safer to just check the whole set if size is small (< 1000).
                # If size is large, we might skip or optimize.
                
                # For now, full check.
                if _check_divisibility(new_sums):
                    # Accept
                    current_set.append(c)
                    current_sums = new_sums
                    
                    backtrack(i + 1)
                    
                    # Backtrack
                    current_set.pop()
                    current_sums = set(current_sums) # Revert to previous state? 
                    # We need to restore current_sums to the state before adding c.
                    # We passed new_sums by reference? No, we assigned.
                    # We need to save the old sums.
                    
                if time.time() - start_time > time_limit:
                    return
                    
        # We need to fix the backtracking state management.
        # Let's pass state as arguments to avoid nonlocal complexity.
        
        def backtrack_v2(idx, curr_set, curr_sums):
            nonlocal best_set, best_size
            
            if time.time() - start_time > time_limit:
                return
            
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set[:]
                
            for i in range(idx, len(candidates)):
                c = candidates[i]
                
                if c in curr_sums:
                    continue
                    
                collision = False
                for s in curr_sums:
                    if (s + c) in curr_sums:
                        collision = True
                        break
                        
                if collision:
                    continue
                    
                new_sums = set(curr_sums)
                new_sums.add(c)
                for s in curr_sums:
                    new_sums.add(s + c)
                    
                if _check_divisibility(new_sums):
                    backtrack_v2(i + 1, curr_set + [c], new_sums)
                    
                if time.time() - start_time > time_limit:
                    return
                    
        backtrack_v2(0, [], set())
        
    if not best_set:
        return [100]
        
    return [int(x) for x in best_set]

def _check_divisibility(sums):
    """
    Check if any sum in the set divides another.
    """
    if not sums:
        return True
        
    sorted_sums = sorted(sums)
    sums_set = sums
    max_sum = sorted_sums[-1]
    
    for s in sorted_sums:
        if s == 1:
            if len(sums) > 1:
                return False
            continue
            
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
            
    return True
