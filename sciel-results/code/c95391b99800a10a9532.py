import time
import random

def construct():
    N = 128
    # Start with the known valid set {65, ..., 128}
    base_set = set(range(65, N + 1))
    best_set = list(base_set)
    best_size = len(best_set)
    
    start_time = time.time()
    deadline = start_time + 55  # 55s limit for safety
    
    candidates = list(range(1, 65))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = set(base_set)
        
        # We can also try to start with a slightly different base?
        # E.g., remove one large number to fit two small ones?
        # For now, keep base fixed.
        
        for x in candidates:
            if x in current_set:
                continue
            
            # Check safety
            # Find all a in current_set such that a+x is in current_set
            potential_as = []
            for a in current_set:
                if a + x in current_set:
                    potential_as.append(a)
            
            if len(potential_as) < 2:
                current_set.add(x)
                continue
            
            # Check if any pair (a, b) in potential_as has sum in current_set
            safe = True
            # Optimization: if potential_as is large, we can use a set of sums?
            # But potential_as is derived from current_set.
            # Let's just do nested loop.
            for i in range(len(potential_as)):
                a = potential_as[i]
                for j in range(i + 1, len(potential_as)):
                    b = potential_as[j]
                    if a + b in current_set:
                        safe = False
                        break
                if not safe:
                    break
            
            if safe:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    return best_set
