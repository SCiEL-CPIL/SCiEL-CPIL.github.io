import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    best_set = []
    
    # Candidate pool: Mix of large and medium numbers
    # Large numbers (128-256) have fewer multiples
    # Medium numbers (64-127) can sometimes fit and add size
    candidates_base = list(range(64, 257))
    
    max_restarts = 2000
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        candidates = candidates_base.copy()
        random.shuffle(candidates)
        
        current_set = []
        sums = set()
        # divisors_map: maps each sum to its divisors (for fast "does n divide some old sum" check)
        # Actually, let's use a different approach:
        # For direction 2 (old sum is multiple of new sum), we check if any old sum % new_sum == 0
        # We can speed this up by noting that if new_sum > max_old_sum, no old sum can be a multiple.
        
        for c in candidates:
            # New sums: {c} U {s + c for s in sums}
            new_sums_list = [c] + [s + c for s in sums]
            
            # Check for duplicates in new_sums (if two new sums are equal, they divide each other)
            if len(new_sums_list) != len(set(new_sums_list)):
                continue
            
            valid = True
            max_old = max(sums) if sums else 0
            
            # Check all pairs of new sums against each other and against old sums
            # For efficiency, process new sums and check:
            # 1. New sum N vs old sum O: N % O == 0 (N is multiple of O) OR O % N == 0 (O is multiple of N)
            # 2. New sum N1 vs new sum N2: N1 % N2 == 0 or N2 % N1 == 0
            
            # First, check new sums against old sums
            for n in new_sums_list:
                # Check if n is a multiple of any old sum
                if sums:
                    for o in sums:
                        if n % o == 0 or o % n == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check new sums against each other
            # Sort to make it easier: smaller first
            ns_sorted = sorted(new_sums_list)
            for i in range(len(ns_sorted)):
                a = ns_sorted[i]
                for j in range(i+1, len(ns_sorted)):
                    b = ns_sorted[j]
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If valid, add
            current_set.append(c)
            sums.update(new_sums_list)
                    
            if len(current_set) > len(best_set):
                best_set = current_set.copy()
                
    return [int(x) for x in best_set]
