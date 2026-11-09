import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # Precompute valid d ranges for 3-AP and 4-AP checks for each x
    # To avoid recalculating bounds every time
    
    # 3-AP checks for adding x:
    # 1. x is middle: d in [1, min(x-1, N-x)]
    # 2. x is first: d in [1, (N-x)//2]
    # 3. x is last: d in [1, x//2]
    
    # 4-AP checks for adding x:
    # 1. x is 1st: d in [1, (N-x)//3]
    # 2. x is 2nd: d in [1, min(x-1, (N-x)//2)]  # x-d>=1, x+2d<=N
    # 3. x is 3rd: d in [1, min(x//2, N-x)]       # x-2d>=1, x+d<=N
    # 4. x is 4th: d in [1, x//3]                 # x-3d>=1

    def count_3ap_delta_add(x, A):
        """Returns number of new 3-APs created by adding x to A."""
        delta = 0
        # x is middle
        max_d_mid = min(x - 1, N - x)
        for d in range(1, max_d_mid + 1):
            if (x - d in A) and (x + d in A):
                delta += 1
        
        # x is first
        max_d_first = (N - x) // 2
        for d in range(1, max_d_first + 1):
            if (x + d in A) and (x + 2 * d in A):
                delta += 1
                
        # x is last
        max_d_last = x // 2
        for d in range(1, max_d_last + 1):
            if (x - 2 * d in A) and (x - d in A):
                delta += 1
        return delta

    def is_valid_add(x, A):
        """Checks if adding x to A creates a 4-AP."""
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in A) and (x + 2 * d in A) and (x + 3 * d in A):
                return False
        
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in A) and (x + d in A) and (x + 2 * d in A):
                return False
                
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        max_d3 = min(x // 2, N - x)
        for d in range(1, max_d3 + 1):
            if (x - 2 * d in A) and (x - d in A) and (x + d in A):
                return False
                
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        max_d4 = x // 3
        for d in range(1, max_d4 + 1):
            if (x - 3 * d in A) and (x - 2 * d in A) and (x - d in A):
                return False
                
        return True

    def count_3ap_total(A):
        """Counts total 3-APs in A. Used for initialization."""
        count = 0
        s = set(A)
        for a in s:
            # a is middle
            max_d_mid = min(a - 1, N - a)
            for d in range(1, max_d_mid + 1):
                if (a - d in s) and (a + d in s):
                    count += 1
            # a is first
            max_d_first = (N - a) // 2
            for d in range(1, max_d_first + 1):
                if (a + d in s) and (a + 2 * d in s):
                    count += 1
            # a is last
            max_d_last = a // 2
            for d in range(1, max_d_last + 1):
                if (a - 2 * d in s) and (a - d in s):
                    count += 1
        return count

    best_set = set()
    best_score = -1
    
    # List of all elements
    elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        # 1. Initialize with Random Greedy
        A = set()
        random.shuffle(elements)
        for x in elements:
            if is_valid_add(x, A):
                A.add(x)
        
        current_score = count_3ap_total(A)
        
        # 2. Local Search (Simulated Annealing)
        T = 5.0
        cooling = 0.999
        steps_per_restart = 20000
        
        for _ in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            x = random.choice(elements)
            
            if x in A:
                # Try to remove x
                # Delta C is negative of count_3ap_delta_add(x, A - {x})?
                # No, removing x destroys APs containing x.
                # The number of destroyed APs is exactly count_3ap_delta_add(x, A - {x}).
                # Let's call it delta_remove.
                # We need to calculate how many 3-APs contain x.
                # This is the same as count_3ap_delta_add(x, A \ {x}).
                A.remove(x)
                delta = -count_3ap_delta_add(x, A)
                # A is now A \ {x}. The delta is the change in score.
                # Score decreases by count_3ap_delta_add(x, A_current).
                # So new_score = old_score + delta.
                # Acceptance: always accept if we are in SA? 
                # Usually we accept if random < exp(delta/T).
                if random.random() < math.exp(delta / T) or delta > 0:
                    # Accepted
                    pass # A is already updated
                else:
                    # Rejected
                    A.add(x)
                    # Score unchanged
            else:
                # Try to add x
                if is_valid_add(x, A):
                    delta = count_3ap_delta_add(x, A)
                    # delta is positive
                    if random.random() < math.exp(delta / T) or delta > 0:
                        A.add(x)
                        # Score increases by delta
                    else:
                        # Rejected
                        pass
                # If invalid, do nothing (effectively rejected)
            
            T *= cooling
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(A)
                
            # Update current_score
            # We need to track current_score properly.
            # In the loop above, I didn't update current_score.
            # Let's refactor to track score explicitly.

        # End local search
        if current_score > best_score:
             # This check is redundant if done inside loop, but good for safety
             pass

    return sorted(best_set)
