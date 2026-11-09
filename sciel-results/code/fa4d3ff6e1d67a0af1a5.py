import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Start with all odd numbers
    S = set(range(1, 257, 2))
    
    # List of even numbers to consider
    evens = list(range(2, 257, 2))
    
    # We can sort evens by some heuristic, e.g., number of conflicts with current S
    # But simple order might be fine. Let's try small to large.
    
    for e in evens:
        if time.time() > deadline:
            break
        
        # Check compatibility with all elements in S
        compatible = True
        for b in S:
            if (e * b) % (e + b) == 0:
                compatible = False
                break
        
        if compatible:
            S.add(e)
            
    # If time remains, try to add more by swapping?
    # Or just return the set found.
    # To maximize cardinality, maybe we should try different orders of evens?
    # Or try to remove an odd if it allows adding multiple evens?
    
    # Simple improvement: Try adding evens in reverse order?
    # Or random restarts?
    
    # Let's stick to the greedy addition first.
    # If we have time, we can try a local search.
    
    return list(S)
