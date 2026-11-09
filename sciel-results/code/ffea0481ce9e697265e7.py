import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin for safety
    
    max_val = 4096
    max_size = 80
    
    best_set = []
    best_size = 0
    
    def can_add(x, current_set, sum_set):
        """
        Check if x can be added to current_set without creating duplicate triple sums.
        Uses early-exit to avoid building the full list of new sums.
        Returns True if x can be added, False otherwise.
        """
        k = len(current_set)
        
        # Collect new sums with early-exit on collision
        seen = set()
        
        # 3x
        s = 3 * x
        if s in seen or s in sum_set:
            return False
        seen.add(s)
        
        # 2x + y for each y in current_set
        for y in current_set:
            s = 2 * x + y
            if s in seen or s in sum_set:
                return False
            seen.add(s)
        
        # x + 2y for each y in current_set
        for y in current_set:
            s = x + 2 * y
            if s in seen or s in sum_set:
                return False
            seen.add(s)
        
        # x + y + z for y < z in current_set
        for i in range(k):
            y = current_set[i]
            for j in range(i + 1, k):
                z = current_set[j]
                s = x + y + z
                if s in seen or s in sum_set:
                    return False
                seen.add(s)
        
        return True
    
    def run_greedy(seed=None):
        """
        Runs a greedy algorithm to build a B_3 set.
        If seed is not None, it shuffles the candidate list with that seed.
        """
        current_set = []
        sum_set = set()
        
        # Create candidate list
        candidates = list(range(1, max_val + 1))
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if can_add(x, current_set, sum_set):
                # Compute and add all new sums
                k = len(current_set)
                new_sums = [3 * x]
                for y in current_set:
                    new_sums.append(2 * x + y)
                    new_sums.append(x + 2 * y)
                for i in range(k):
                    y = current_set[i]
                    for j in range(i + 1, k):
                        new_sums.append(x + y + current_set[j])
                
                current_set.append(x)
                sum_set.update(new_sums)
                
                if len(current_set) >= max_size:
                    break
        
        return current_set
    
    def local_swap_optimize(s, max_swaps=500):
        """
        Try to improve set s by swapping elements with larger candidates.
        For each element in s, try replacing it with a larger number that fits.
        This can help pack more elements or use the range more efficiently.
        """
        if not s:
            return s
        
        s = sorted(s)
        n = len(s)
        
        # Build sum_set from s
        sum_set = set()
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                b = s[j]
                for k in range(j, n):
                    c = s[k]
                    sum_set.add(a + b + c)
        
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            for idx in range(n):
                if time.time() > deadline:
                    return s
                old_val = s[idx]
                # Try replacing old_val with a larger value
                # Start from old_val + 1 up to max_val
                for new_val in range(old_val + 1, max_val + 1):
                    if new_val in s:
                        continue
                    
                    # Check if we can swap: remove old_val, add new_val
                    # First, check if new_val can be added to s without old_val
                    s_without_old = s[:idx] + s[idx+1:]
                    
                    if can_add(new_val, s_without_old, sum_set):
                        # Also need to verify that removing old_val doesn't leave
                        # any sums that are now "orphaned" - but since we're only
                        # checking that new_val doesn't conflict, and the remaining
                        # set is still valid (subset of a valid set), this should be fine.
                        # Actually, we need to be more careful: the sums involving old_val
                        # are removed, so new_val just needs to not conflict with the
                        # remaining sums.
                        
                        # Compute sums of s_without_old
                        remaining_sums = set()
                        m = len(s_without_old)
                        for i in range(m):
                            a = s_without_old[i]
                            for j in range(i, m):
                                b = s_without_old[j]
                                for k in range(j, m):
                                    c = s_without_old[k]
                                    remaining_sums.add(a + b + c)
                        
                        # Check if new_val's sums conflict with remaining_sums
                        if can_add(new_val, s_without_old, remaining_sums):
                            # Perform the swap
                            s[idx] = new_val
                            s.sort()
                            # Rebuild sum_set
                            sum_set = set()
                            n2 = len(s)
                            for i in range(n2):
                                a = s[i]
                                for j in range(i, n2):
                                    b = s[j]
                                    for k in range(j, n2):
                                        c = s[k]
                                        sum_set.add(a + b + c)
                            improved = True
                            swap_count += 1
                            break
                if time.time() > deadline:
                    return s
        
        return s
    
    # 1. Run deterministic increasing greedy
    set1 = run_greedy()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
        
    # Try local swap optimization on the deterministic result
    if best_set:
        optimized = local_swap_optimize(best_set[:])
        if len(optimized) > best_size:
            best_size = len(optimized)
            best_set = optimized
    
    # 2. Try randomized greedy with multiple seeds
    num_random_trials = 0
    base_seed = 12345
    
    while time.time() < deadline - 3.0:  # Leave 3s margin for final processing
        seed = base_seed + num_random_trials
        s = run_greedy(seed=seed)
        
        # Try local swap optimization if the set is promising
        if len(s) >= best_size - 1 and len(s) > 0:
            s_opt = local_swap_optimize(s[:], max_swaps=200)
            if len(s_opt) > best_size:
                best_size = len(s_opt)
                best_set = s_opt
        elif len(s) > best_size:
            best_size = len(s)
            best_set = s
        
        if best_size >= 42:
            break
        num_random_trials += 1
    
    return best_set
