import time
import random
import sys

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses randomized greedy initialization and local search (add/swap) to maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 512
    
    def can_add(S, x, n):
        """
        Check if x can be added to S without creating a 4-AP.
        Returns True if x can be added.
        """
        # Case 1: x is the first term (a)
        # x, x+d, x+2d, x+3d
        # x+3d <= n => d <= (n-x)//3
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        
        # Case 2: x is the last term (a+3d)
        # x-3d, x-2d, x-d, x
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return False
        
        # Case 3: x is the second term (a+d)
        # x-d, x, x+d, x+2d
        # x-d >= 1 => d <= x-1
        # x+2d <= n => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        
        # Case 4: x is the third term (a+2d)
        # x-2d, x-d, x, x+d
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return False
                
        return True

    def greedy_randomized(n, seed):
        """
        Generate a 4-AP-free set using randomized greedy insertion.
        """
        random.seed(seed)
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x, n):
                S.add(x)
        return S

    def local_search(S, n, time_limit):
        """
        Perform local search on set S.
        1. Add phase: Add any element that doesn't conflict.
        2. Swap phase: Remove one element, then try to add as many as possible.
           If the resulting set is larger, accept the change.
        """
        S = set(S)
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            
            # Phase 1: Add
            for x in range(1, n + 1):
                if time.time() - start > time_limit:
                    break
                if x not in S:
                    if can_add(S, x, n):
                        S.add(x)
                        improved = True
            if time.time() - start > time_limit:
                break
            
            # Phase 2: Swap (1-exchange with re-optimization)
            # We iterate through elements to remove.
            # To avoid bias, we can shuffle the removal order, but for now, we iterate.
            # We need to be careful not to oscillate.
            # We only accept a swap if the final size is strictly larger.
            
            # Create a list of elements to try removing
            # We can't modify S while iterating over it, so we make a copy or list
            elements_to_remove = list(S)
            random.shuffle(elements_to_remove) # Randomize order to escape local minima
            
            for y in elements_to_remove:
                if time.time() - start > time_limit:
                    break
                if y not in S:
                    continue
                
                # Try removing y
                S_temp = S - {y}
                
                # Now try to add as many elements as possible to S_temp
                # We do a simple pass of adding
                added_any = False
                for x in range(1, n + 1):
                    if time.time() - start > time_limit:
                        break
                    if x not in S_temp:
                        if can_add(S_temp, x, n):
                            S_temp.add(x)
                            added_any = True
                
                # If S_temp is larger than S, accept the change
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    # Break out of removal loop to restart add phase? 
                    # Or continue? Restarting add phase is safer.
                    break
                    
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    num_restarts = 50
    # We allocate time based on how long one iteration takes roughly.
    # Let's just run until deadline.
    
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        seed = 42 + i
        current_set = greedy_randomized(n, seed)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_set = current_set
            best_size = current_size
            
        # Local search on this set
        # Allocate remaining time, minus a buffer for next restarts if possible
        # Or just run local search for a fixed chunk of time
        time_left = deadline - (time.time() - start_time)
        if time_left < 2:
            break
            
        # Use up to 50% of remaining time for local search, or a minimum of 1s
        ls_time = min(time_left * 0.5, 10.0)
        
        searched_set = local_search(current_set, n, ls_time)
        
        if len(searched_set) > best_size:
            best_set = searched_set
            best_size = len(searched_set)
            
    # Final verification
    def is_4ap_free(S):
        S_set = set(S)
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                    return False
        return True
        
    if not is_4ap_free(best_set):
        # Fallback to a known valid set (base 3) if something went wrong
        result = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                result.add(x)
        best_set = result

    return sorted(best_set)

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Size: {len(s)}")
    # Verify
    S_set = set(s)
    ok = True
    for a in s:
        for d in range(1, (512 - a) // 3 + 1):
            if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                ok = False
                print(f"Found 4-AP: {a}, {a+d}, {a+2*d}, {a+3*d}")
                break
        if not ok:
            break
    if ok:
        print("Valid 4-AP-free set.")
