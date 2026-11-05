def solve_for_n(n, deadline):
    # Precomputation
    # ...
    
    while time.time() < deadline - 2: # Leave 2s for check
        # Init
        colors = np.random.randint(0, 2, M)
        # ... init counts ...
        
        # SA Loop
        T = 1.0
        min_T = 0.01
        cooling_rate = 0.999
        chunk_start = time.time()
        chunk_deadline = min(chunk_start + 10, deadline - 2)
        
        while time.time() < chunk_deadline:
             # ... step ...
             
        if E == 0:
            return colors.copy()
            
    return None
