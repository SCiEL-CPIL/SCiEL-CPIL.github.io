import time
import random

def _score(S):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d in S."""
    S_set = set(S)
    valid_ds = set()
    # Iterate over all pairs
    # Optimization: list(S) to avoid iteration issues if modified, but S is not modified here
    elems = list(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            # Check if x-d is in S or y+d is in S
            if (x - d) in S_set or (y + d) in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    best_set = None
    best_score = 0
    
    # Try several random restarts
    # Each restart: random init, then hill climb
    # We don't know how many restarts we can do, so we loop until deadline
    
    while time.time() < deadline:
        # Random initialization
        current_set = set(random.sample(range(1, 513), 32))
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Hill climbing phase
        # We can limit the number of steps per restart to avoid getting stuck in one long local search
        # Or just run until a time slice is exhausted
        # Let's do a time-sliced hill climb
        restart_deadline = time.time() + 5.0  # 5 seconds per restart attempt
        
        while time.time() < restart_deadline and time.time() < deadline:
            # Pick a random element to swap
            elem_to_replace = random.choice(list(current_set))
            
            # Pick a random new value
            # To speed up, we can pick from a pool of "promising" values or just random
            new_val = random.randint(1, 512)
            
            if new_val in current_set:
                continue
                
            # Try swap
            new_set = set(current_set)
            new_set.remove(elem_to_replace)
            new_set.add(new_val)
            
            new_score = _score(new_set)
            
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            # Optional: accept worse moves with low probability (Simulated Annealing)
            # But simple hill climbing often works well for this.
            # To escape local minima, the random restarts help.
            
    # Return the best set found
    if best_set is None:
        return list(range(1, 33))
    return list(best_set)
