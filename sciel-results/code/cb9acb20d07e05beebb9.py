import random
import time

def construct():
    N = 1000
    start_time = time.time()
    limit = 115  # Leave 5s buffer
    
    # Helper to check if x can be added to A given counts
    def is_valid_add(A, x, counts):
        if x in A:
            return False
        if counts.get(2*x, 0) >= 2:
            return False
        for y in A:
            if counts.get(x+y, 0) >= 2:
                return False
        return True

    def compute_valid(A, counts):
        valid = []
        A_set = set(A)
        for x in range(1, N + 1):
            if x not in A_set:
                if is_valid_add(A, x, counts):
                    valid.append(x)
        return valid

    def add_element(A, x, counts):
        A.append(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for y in A[:-1]:
            s = x + y
            counts[s] = counts.get(s, 0) + 1

    def remove_element(A, x, counts):
        A.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in A:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]

    best_A = []
    best_size = 0
    
    # Phase 1: Multiple Random Greedy Initializations
    # Try a few random orders to find a good starting point
    # Time budget for init: 2 seconds
    init_limit = start_time + 2.0
    
    while time.time() < init_limit:
        A = []
        counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(A, x, counts):
                add_element(A, x, counts)
        
        if len(A) > best_size:
            best_size = len(A)
            best_A = list(A)
            
        # If we are already very large, maybe stop?
        # B2[2] sets for N=1000 are around 400-500?
        # Actually, sqrt(N) is ~31. But B2[2] allows 2 representations.
        # The density is higher. 
        # Let's just run the loop.

    # Phase 2: Local Search on the best found so far
    if best_A:
        A = list(best_A)
        # Reconstruct counts for the best A
        counts = {}
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        
        valid = compute_valid(A, counts)
        
        while time.time() - start_time < limit:
            if valid:
                # Pick a random valid element
                # Bias towards smaller elements? Smaller elements create more conflicts with larger ones?
                # Actually, adding small numbers early is good for packing.
                # Let's pick randomly from valid.
                x = random.choice(valid)
                add_element(A, x, counts)
                
                # Update valid
                # Optimization: We don't need to recompute ALL valid.
                # But it's simpler and safe.
                valid = compute_valid(A, counts)
                
                if len(A) > best_size:
                    best_size = len(A)
                    best_A = list(A)
            else:
                # Stuck. Remove a random element.
                if not A:
                    break
                x = random.choice(A)
                remove_element(A, x, counts)
                
                # Try to add the smallest valid element immediately to "repair"
                valid = compute_valid(A, counts)
                if valid:
                    # Add the smallest one to try to fill the gap
                    y = min(valid) 
                    add_element(A, y, counts)
                    valid = compute_valid(A, counts)
                    
                    if len(A) > best_size:
                        best_size = len(A)
                        best_A = list(A)

    return best_A
