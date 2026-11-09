import time

def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums
    are distinct, minimizing the largest integer.
    
    Uses a greedy algorithm to find the smallest possible next integer
    that does not create a collision in the subset sums.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Fallback optimal solution
    fallback = [1 << i for i in range(n)]
    
    current_set = []
    sums = {0}
    
    for k in range(n):
        if time.time() > deadline:
            # If we run out of time, return the best we have or fallback
            # Since the greedy approach is fast, this is unlikely.
            # If we have a partial set, we can't return it (needs 12).
            # So return fallback.
            return fallback
            
        # Find the smallest x not in the difference set of sums
        # Equivalent to: sums and sums+x are disjoint
        x = 1
        while True:
            # Check collision
            # If any s in sums has s+x in sums, collision
            # We can break early if found
            collision = False
            for s in sums:
                if s + x in sums:
                    collision = True
                    break
            
            if not collision:
                break
                
            x += 1
            
        # Add x
        current_set.append(x)
        # Update sums
        new_sums = set(s + x for s in sums)
        sums.update(new_sums)
        
    return current_set
