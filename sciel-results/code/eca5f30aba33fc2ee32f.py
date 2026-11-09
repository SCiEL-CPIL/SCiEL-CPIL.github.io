import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflicts
    # conflicts[i] is a set of indices j such that numbers[i] and numbers[j] conflict
    conflicts = [set() for _ in range(N)]
    
    for i in range(N):
        a = numbers[i]
        for j in range(i + 1, N):
            b = numbers[j]
            g = 1
            # Calculate gcd
            x, y = a, b
            while y:
                x, y = y, x % y
            g = x
            
            # Check if (a/g + b/g) divides g
            xg = a // g
            yg = b // g
            s = xg + yg
            if s != 0 and g % s == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Heuristic: Greedy Independent Set
    # Order nodes by degree (ascending)
    # Try to pick a node, remove it and its neighbors
    
    # We can try multiple random orderings or different heuristics
    
    best_set = set()
    
    # Initial greedy: pick nodes with lowest degree first
    # But we want to maximize size.
    # A common heuristic is to sort by degree and pick the first available.
    
    # Let's try a simple backtracking with pruning for the first few steps, then greedy?
    # Or just a good greedy.
    
    # Let's try sorting by degree
    order = sorted(range(N), key=lambda i: len(conflicts[i]))
    
    current_set = set()
    for i in order:
        if all(j not in current_set for j in conflicts[i]):
            current_set.add(i)
            
    best_size = len(current_set)
    best_indices = list(current_set)
    
    # Local Search: try to improve by swapping
    # Or try different orderings
    
    # Let's try a more sophisticated approach:
    # Randomized greedy with restarts
    
    best_set_global = set(best_indices)
    
    while time.time() < deadline:
        # Randomly shuffle the order of nodes
        random_order = list(range(N))
        random.shuffle(random_order)
        
        # Or sort by random weight + degree
        # Let's just shuffle
        
        curr_set = set()
        for i in random_order:
            if all(j not in curr_set for j in conflicts[i]):
                curr_set.add(i)
                
        if len(curr_set) > len(best_set_global):
            best_set_global = curr_set
            
    # Convert indices to values
    result = [numbers[i] for i in best_set_global]
    
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a = result[i]
            b = result[j]
            g = 1
            x, y = a, b
            while y:
                x, y = y, x % y
            g = x
            s = a//g + b//g
            if s != 0 and g % s == 0:
                return list(range(1, 257, 2)) # Fallback to valid seed
        
    return result
