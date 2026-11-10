import random
import time

def construct():
    n = 512
    numbers = list(range(1, n + 1))
    
    def can_add(selected, x):
        # Check if x is the 4th term of a 4-AP
        # a, a+d, a+2d, x
        # x-3d, x-2d, x-d must be in selected
        # d >= 1
        # x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                return False
        return True

    best_set = set()
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    # Run multiple randomized greedy attempts
    while time.time() - start_time < time_limit:
        random.shuffle(numbers)
        current_set = set()
        for x in numbers:
            if can_add(current_set, x):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return sorted(best_set)
