import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = set()
    best_size = 0
    
    # We will run multiple instances of the shuffled greedy algorithm.
    # Each run generates a random permutation of [1, 512] and greedily adds
    # elements if they do not form a 4-term arithmetic progression.
    
    while time.time() < deadline:
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        current_set = set()
        
        for x in candidates:
            # Check if x can be added
            # A 4-AP is a, a+d, a+2d, a+3d
            # If x is added, it must not complete any 4-AP with 3 existing elements.
            
            valid = True
            
            # Case 1: x is the 4th term (a+3d)
            # a = x-3d. Terms: x-3d, x-2d, x-d, x
            # Constraints: x-3d >= 1 => d <= (x-1)//3
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (x - 3*d in current_set) and (x - 2*d in current_set) and (x - d in current_set):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 2: x is the 3rd term (a+2d)
            # a = x-2d. Terms: x-2d, x-d, x, x+d
            # Constraints: x-2d >= 1 => d <= (x-1)//2
            #             x+d <= 512 => d <= 512-x
            max_d = min((x - 1) // 2, 512 - x)
            for d in range(1, max_d + 1):
                if (x - 2*d in current_set) and (x - d in current_set) and (x + d in current_set):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 3: x is the 2nd term (a+d)
            # a = x-d. Terms: x-d, x, x+d, x+2d
            # Constraints: x-d >= 1 => d <= x-1
            #             x+2d <= 512 => d <= (512-x)//2
            max_d = min(x - 1, (512 - x) // 2)
            for d in range(1, max_d + 1):
                if (x - d in current_set) and (x + d in current_set) and (x + 2*d in current_set):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Case 4: x is the 1st term (a)
            # Terms: x, x+d, x+2d, x+3d
            # Constraints: x+3d <= 512 => d <= (512-x)//3
            max_d = (512 - x) // 3
            for d in range(1, max_d + 1):
                if (x + d in current_set) and (x + 2*d in current_set) and (x + 3*d in current_set):
                    valid = False
                    break
            
            if valid:
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return sorted(best_set)
