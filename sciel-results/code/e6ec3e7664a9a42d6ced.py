import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    
    best_set = []
    
    # Candidate pool: Mix of large and medium numbers
    # Large numbers (128-256) have fewer multiples
    # Medium numbers (64-127) can sometimes fit and add size
    candidates_base = list(range(64, 257))
    
    max_restarts = 5000
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        candidates = candidates_base.copy()
        random.shuffle(candidates)
        
        current_set = []
        sums = set()
        
        for c in candidates:
            # New sums: {c} U {s + c for s in sums}
            new_sums_list = [c] + [s + c for s in sums]
            
            # Check for duplicates in new_sums
            if len(new_sums_list) != len(set(new_sums_list)):
                continue
            
            valid = True
            
            # Check new sums against old sums
            for n in new_sums_list:
                for o in sums:
                    if n > o:
                        if n % o == 0:
                            valid = False
                            break
                    else:
                        if o % n == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check new sums against each other
            ns_sorted = sorted(new_sums_list)
            for idx_a in range(len(ns_sorted)):
                a = ns_sorted[idx_a]
                for idx_b in range(idx_a + 1, len(ns_sorted)):
                    b = ns_sorted[idx_b]
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
