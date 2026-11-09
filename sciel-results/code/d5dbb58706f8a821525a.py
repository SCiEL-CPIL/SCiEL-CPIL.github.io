"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """Find a large Sidon set in [1,284] containing {1,2,5,11}."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Prescribed elements
    prescribed = {1, 2, 5, 11}
    
    def is_valid_sidon(s):
        """Check if set s is a valid Sidon set (all pairwise sums distinct)."""
        s_list = sorted(s)
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sm = s_list[i] + s_list[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def count_sum_conflicts(s, x):
        """Count how many existing sums would be violated by adding x to s.
        Returns number of conflicts (0 means x can be added)."""
        s_list = sorted(s)
        existing_sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                existing_sums.add(s_list[i] + s_list[j])
        
        conflicts = 0
        for y in s_list:
            sm = x + y
            if sm in existing_sums:
                conflicts += 1
        # Also check 2*x
        if 2 * x in existing_sums:
            conflicts += 1
        return conflicts
    
    def greedy_extend(base_set):
        """Greedily extend a base set by adding elements that cause no conflicts."""
        s = set(base_set)
        # Compute current sums
        s_list = sorted(s)
        sums = set()
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
        
        # Try adding elements from largest to smallest (or randomized order)
        candidates = list(range(284, 1, -1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in s:
                continue
            # Check if x can be added
            can_add = True
            for y in s:
                sm = x + y
                if sm in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                # Add x
                s.add(x)
                s_list = sorted(s)
                # Update sums
                idx = s_list.index(x)
                for j in range(idx, len(s_list)):
                    sums.add(s_list[idx] + s_list[j])
        
        return s
    
    def local_search(s, time_limit):
        """Local search: try swaps to improve the set size."""
        s = set(s)
        best_s = set(s)
        best_size = len(s)
        
        while time.time() < time_limit:
            improved = False
            s_list = sorted(s)
            n = len(s_list)
            
            # Compute all sums
            sums = set()
            for i in range(n):
                for j in range(i, n):
                    sums.add(s_list[i] + s_list[j])
            
            # Try removing one element and adding another
            for i in range(n):
                if time.time() >= time_limit:
                    break
                to_remove = s_list[i]
                # Try adding various elements
                for x in range(284, 0, -1):
                    if x in s:
                        continue
                    # Check if removing to_remove and adding x gives a valid set
                    s_new = s - {to_remove}
                    s_new.add(x)
                    
                    # Quick check: compute new sums
                    s_new_list = sorted(s_new)
                    sums_new = set()
                    valid = True
                    for k in range(len(s_new_list)):
                        for l in range(k, len(s_new_list)):
                            sm = s_new_list[k] + s_new_list[l]
                            if sm in sums_new:
                                valid = False
                                break
                            sums_new.add(sm)
                        if not valid:
                            break
                    
                    if valid and len(s_new) > best_size:
                        best_s = s_new
                        best_size = len(s_new)
                        improved = True
                        break
            
            if not improved:
                # Try pairwise swaps
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    for j in range(i+1, n):
                        if time.time() >= time_limit:
                            break
                        a, b = s_list[i], s_list[j]
                        # Try replacing a with x
                        for x in range(284, 0, -1):
                            if x in s:
                                continue
                            s_new = s - {a}
                            s_new.add(x)
                            s_new_list = sorted(s_new)
                            sums_new = set()
                            valid = True
                            for k in range(len(s_new_list)):
                                for l in range(k, len(s_new_list)):
                                    sm = s_new_list[k] + s_new_list[l]
                                    if sm in sums_new:
                                        valid = False
                                        break
                                    sums_new.add(sm)
                                if not valid:
                                    break
                            
                            if valid:
                                best_s = s_new
                                best_size = len(s_new)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
            
            s = best_s
        
        return best_s
    
    # Strategy: multiple randomized greedy + local search
    best_result = set(prescribed)
    
    # First, do a simple greedy to get a baseline
    baseline = greedy_extend(prescribed)
    best_result = baseline
    
    # Now do randomized restarts with local search
    num_restarts = 50
    restart_time = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy: shuffle candidate order
        base = set(prescribed)
        s = greedy_extend(base)
        
        # Local search with time limit
        ls_time = time.time() + restart_time * 0.6
        s = local_search(s, ls_time)
        
        if len(s) > len(best_result):
            best_result = s
    
    # Final: try to extend the best result further
    if time.time() < deadline - 2:
        extended = greedy_extend(best_result)
        if len(extended) > len(best_result):
            best_result = extended
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
