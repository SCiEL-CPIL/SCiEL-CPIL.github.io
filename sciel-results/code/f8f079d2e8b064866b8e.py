def construct():
    """
    Returns a list of 12 distinct positive integers such that all subset sums are distinct.
    Uses a greedy approach to find the minimal largest integer.
    """
    n = 12
    current_set = []
    current_sums = {0}
    
    for i in range(n):
        x = 1
        while True:
            new_sums = set()
            # Check if adding x creates duplicate subset sums
            valid = True
            for s in current_sums:
                new_s = s + x
                if new_s in current_sums or new_s in new_sums:
                    valid = False
                    break
                new_sums.add(new_s)
            
            if valid:
                current_set.append(x)
                current_sums.update(new_sums)
                break
            x += 1
            
    return current_set
