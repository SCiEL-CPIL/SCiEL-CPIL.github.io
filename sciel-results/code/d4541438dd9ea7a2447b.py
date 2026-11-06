import random
import time

def construct():
    n = 512
    
    def creates_4ap_with_x(S, x):
        # S is a set
        # Check if adding x to S creates a 4-AP involving x.
        
        # Case 1: x is the first term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # Case 2: x is the second term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # Case 3: x is the third term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # Case 4: x is the fourth term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    all_elements = list(range(1, n + 1))
    
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Randomized Greedy with Restarts (RGR)
    # This approach family is distinct from the Local Search/Swap approach in the seed.
    # It relies on the order of insertion to break ties and find different local optima.
    
    while time.time() - start_time < time_limit:
        current_set = set()
        # Create a randomized order for greedy insertion
        order = list(all_elements)
        random.shuffle(order)
        
        # Greedy Phase
        for x in order:
            if not creates_4ap_with_x(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return sorted(best_set)
