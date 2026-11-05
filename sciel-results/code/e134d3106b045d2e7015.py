def local_search(n, time_limit=5.0):
    best_A = None
    best_cost = float('inf')
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Init random
        A = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                A[i,j] = np.random.rand() < 0.5
                A[j,i] = not A[i,j]
                
        current_cost = count_bad_sets(A)
        
        # Hill climbing
        stuck_counter = 0
        max_iters = 5000 # per restart
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            u = np.random.randint(n)
            v = np.random.randint(n)
            if u == v: continue
            
            # ... delta calculation ...
            
            if new_cost < current_cost:
                # flip
                A[u,v] = not A[u,v]
                A[v,u] = not A[v,u]
                current_cost = new_cost
                stuck_counter = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = A.copy()
                    if best_cost == 0:
                        return best_A, 0
            else:
                stuck_counter += 1
                if stuck_counter > 1000:
                    break # restart
            
    return best_A, best_cost
