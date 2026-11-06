import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0  # Evaluator kills at 90s, leave 5s buffer
    
    # Helper: Check if adding x to set S creates a 4-AP
    # S is a bytearray of size n+1
    # Returns True if it creates one, False otherwise
    def creates_4ap(S, x):
        # x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return True
        
        # x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x + d] and S[x + 2 * d]:
                return True
        
        # x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return True
        
        # x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if S[x - 3 * d] and S[x - 2 * d] and S[x - d]:
                return True
        
        return False

    # Helper: Greedily fill the set given a starting bytearray
    def greedy_fill(S, order=None):
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not S[x]:
                if not creates_4ap(S, x):
                    S[x] = 1
        return S

    # Initial Greedy Construction
    # We try a few different greedy orders to find a good starting point
    best_set = bytearray(n + 1)
    best_size = 0
    
    random.seed(42)
    
    # Standard order
    current = bytearray(n + 1)
    greedy_fill(current, list(range(1, n + 1)))
    current_size = sum(current)
    if current_size > best_size:
        best_set = current
        best_size = current_size
        
    # Reverse order
    current = bytearray(n + 1)
    greedy_fill(current, list(range(n, 0, -1)))
    current_size = sum(current)
    if current_size > best_size:
        best_set = current
        best_size = current_size
        
    # Random order
    for _ in range(20):
        if time.time() - start_time > time_limit * 0.1: # 10% time for init
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = bytearray(n + 1)
        greedy_fill(current, order)
        current_size = sum(current)
        if current_size > best_size:
            best_set = current
            best_size = current_size

    # Local Search
    # Strategy: Kick and Greedy Fill
    current_set = bytearray(best_set)
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        # Try to add any element that can be added
        added = False
        # Get list of candidates
        candidates = [i for i in range(1, n + 1) if not current_set[i]]
        
        if candidates:
            random.shuffle(candidates)
            # Try a few random candidates
            for x in candidates[:100]: # Check up to 100 random candidates
                if not creates_4ap(current_set, x):
                    current_set[x] = 1
                    current_size += 1
                    added = True
                    if current_size > best_size:
                        best_set = bytearray(current_set)
                        best_size = current_size
                    break
        
        if not added:
            # Stuck. Perform a kick.
            # Remove 3-5 random elements
            k = random.randint(3, 5)
            if current_size > k:
                # Sample elements to remove
                elements_in_set = [i for i in range(1, n + 1) if current_set[i]]
                to_remove = random.sample(elements_in_set, k)
                for x in to_remove:
                    current_set[x] = 0
                    current_size -= 1
                
                # Now try to add as many as possible
                # Use a randomized greedy fill to escape local minima
                candidates = [i for i in range(1, n + 1) if not current_set[i]]
                random.shuffle(candidates)
                for x in candidates:
                    if not creates_4ap(current_set, x):
                        current_set[x] = 1
                        current_size += 1
                
                if current_size > best_size:
                    best_set = bytearray(current_set)
                    best_size = current_size

    # Convert bytearray to sorted list of integers
    result = [i for i in range(1, n + 1) if best_set[i]]
    return result
