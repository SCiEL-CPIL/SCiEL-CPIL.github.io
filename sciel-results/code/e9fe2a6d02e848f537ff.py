<<<<<<< SEARCH
    # Initial sets
    initial_sets = []
    
    # Geometric
    for base in [2, 3, 5]:
        s = [0]
        v = 1
        while v <= MAX_U_LIMIT:
            s.append(v)
            v *= base
        initial_sets.append(s)
        
    # Random
    for _ in range(5):
        n = random.randint(20, 100)
        rand_set = {0}
        while len(rand_set) < n:
            rand_set.add(random.randint(0, MAX_U_LIMIT))
        initial_sets.append(list(rand_set))
        
    # Dense low
    initial_sets.append(list(range(50)))
    
    best_result = [0]
    best_score = 0.0
    
    for s in initial_sets:
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        # Run SA
        res, score = _sa_search(s, timeout=min(10.0, remaining))
        if score > best_score:
            best_score = score
            best_result = res
            
    # Final check
    if not best_result or 0 not in best_result:
        best_result = [0]
        
    return best_result
=======
    # Initial sets
    initial_sets = []
    
    # Geometric
    for base in [2, 3, 4, 5, 7, 10]:
        for start in [1, 2]:
            s = [0]
            v = start
            while v <= MAX_U_LIMIT:
                s.append(v)
                v *= base
            initial_sets.append(s)
            
    # Fibonacci-like
    s = [0, 1]
    while s[-1] <= MAX_U_LIMIT:
        s.append(s[-1] + s[-2])
    initial_sets.append(s)

    # Random sparse (biased towards smaller values)
    for _ in range(10):
        n = random.randint(10, 60)
        rand_set = {0}
        while len(rand_set) < n:
            # Bias towards smaller numbers to keep max_u small
            val = int(MAX_U_LIMIT * (random.random() ** 2.0))
            rand_set.add(val)
        initial_sets.append(list(rand_set))
        
    # Dense low (known bad, but good for diversity)
    for size in [10, 20, 50]:
        initial_sets.append(list(range(size)))
    
    best_result = [0]
    best_score = 0.0
    
    for s in initial_sets:
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        # Run SA
        res, score = _sa_search(s, timeout=min(10.0, remaining))
        if score > best_score:
            best_score = score
            best_result = res
            
    # Final check
    if not best_result or 0 not in best_result:
        best_result = [0]
        
    return best_result
>>>>>>> REPLACE

