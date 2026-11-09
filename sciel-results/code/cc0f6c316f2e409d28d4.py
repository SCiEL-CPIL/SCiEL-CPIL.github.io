import time
import random
import math

def construct():
    start_time = time.time()
    deadline_total = start_time + 115  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    
    def is_valid_sidon(marks):
        """Check if the set is a valid Sidon set (distinct pairwise sums)."""
        if len(marks) != len(set(marks)):
            return False
        if any(m < 1 or m > N for m in marks):
            return False
        if not prescribed_set.issubset(set(marks)):
            return False
        
        sums = set()
        n = len(marks)
        for i in range(n):
            for j in range(i, n):
                s = marks[i] + marks[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def solve_csp(k, deadline):
        """
        Solve for a Sidon set of size k using backtracking with forward checking.
        """
        if k < len(prescribed):
            return None
            
        # Initialize state with prescribed elements
        current_set = list(prescribed)
        used_sums = set()
        for i in range(len(prescribed)):
            for j in range(i, len(prescribed)):
                used_sums.add(prescribed[i] + prescribed[j])
                
        # Candidates are all other numbers
        candidates = [x for x in range(1, N + 1) if x not in prescribed_set]
        
        # We will try multiple random orders of candidates to escape local traps
        # or different search paths.
        num_restarts = 20
        # Distribute time among restarts
        # If k=20, we have most of the time.
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            random.shuffle(candidates)
            
            # DFS function
            # To avoid recursion depth issues (though 20 is fine), we use recursion.
            # Python recursion limit is usually 1000, so depth 20 is safe.
            
            def dfs(idx, count_remaining, cur_set, cur_sums):
                if time.time() > deadline:
                    return None
                if count_remaining == 0:
                    return list(cur_set)
                if idx >= len(candidates):
                    return None
                if len(candidates) - idx < count_remaining:
                    return None
                
                # Option 1: Skip candidates[idx]
                res = dfs(idx + 1, count_remaining, cur_set, cur_sums)
                if res is not None:
                    return res
                
                # Option 2: Take candidates[idx]
                val = candidates[idx]
                
                # Check validity against current set
                valid = True
                new_sums = []
                
                # Check sums with existing elements
                for y in cur_set:
                    s = val + y
                    if s in cur_sums:
                        valid = False
                        break
                    new_sums.append(s)
                
                # Check sum with itself
                if valid:
                    s_double = val + val
                    if s_double in cur_sums:
                        valid = False
                    else:
                        new_sums.append(s_double)
                
                if valid:
                    # Commit
                    cur_set.append(val)
                    for s in new_sums:
                        cur_sums.add(s)
                        
                    res = dfs(idx + 1, count_remaining - 1, cur_set, cur_sums)
                    if res is not None:
                        return res
                    
                    # Backtrack
                    cur_set.pop()
                    for s in new_sums:
                        cur_sums.remove(s)
                
                return None

            result = dfs(0, k - len(prescribed), current_set, used_sums)
            if result is not None:
                # Verify just in case
                if is_valid_sidon(result):
                    return result
                
        return None

    # Try sizes from 20 down to 4
    # Allocate time:
    # k=20: 100s
    # k=19: 10s
    # k=18: 4s
    # k=17: 1s
    # ...
    
    time_allocations = {
        20: 100,
        19: 10,
        18: 4,
        17: 1,
        16: 0.5
    }
    
    best_result = None
    
    for k in range(20, len(prescribed) - 1, -1):
        elapsed = time.time() - start_time
        remaining_total = deadline_total - time.time()
        
        if remaining_total <= 0:
            break
            
        budget = time_allocations.get(k, 1)
        if budget > remaining_total:
            budget = remaining_total * 0.9 # Keep some margin
            
        if budget < 0.1:
            break
            
        deadline_k = time.time() + budget
        
        result = solve_csp(k, deadline_k)
        
        if result is not None:
            # Since we iterate from high k to low, the first valid result is the max size
            # We can return immediately if we are confident, but let's ensure it's valid.
            if is_valid_sidon(result):
                return result
        
    # If no size >= 5 was found, return prescribed set (size 4)
    # The prescribed set is always valid?
    # 1,2,5,11 sums: 2,3,6,12,4,7,13,10,16,22. All distinct. Yes.
    return prescribed

if __name__ == "__main__":
    res = construct()
    print(len(res))
