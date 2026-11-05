def construct():
    import time
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit
    
    # Greedy: pick smallest positive integer that keeps subset sums distinct
    subset_sums = {0}
    chosen = []
    
    for i in range(12):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            # Check if adding candidate keeps all subset sums distinct
            new_sums = subset_sums | {s + candidate for s in subset_sums}
            if len(new_sums) == 2 * len(subset_sums):
                # Valid
                subset_sums = new_sums
                chosen.append(candidate)
                break
            candidate += 1
    
    return chosen
