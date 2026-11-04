<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    # Different row configurations to explore different topologies
    row_configs = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 7, 7]
    ]
    
    # We will alternate between hex and grid inits, or just try both for a few ratios
    # To save time, let's prioritize hex for square-ish and grid for others?
    # Or just run the loop.
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        # Try hex configurations
        for rows in row_configs:
            if time.time() > deadline - 12.0:
                break
            circles, val = _try_aspect(ar, deadline, rows, use_grid=False)
            if circles is not None and val > best_value:
                best_value = val
                best_result = circles
                
        # Try grid configuration (only for a subset to save time)
        if ar in [1.0, 1.2, 0.8]:
             if time.time() > deadline - 12.0:
                 break
             circles, val = _try_aspect(ar, deadline, use_grid=True)
             if circles is not None and val > best_value:
                 best_value = val
                 best_result = circles

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    # Different row configurations to explore different topologies
    row_configs = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 7, 7]
    ]
    
    # We will alternate between hex and grid inits, or just try both for a few ratios
    # To save time, let's prioritize hex for square-ish and grid for others?
    # Or just run the loop.
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        # Try hex configurations
        for rows in row_configs:
            if time.time() > deadline - 12.0:
                break
            circles, val = _try_aspect(ar, deadline, rows, use_grid=False, use_random=True)
            if circles is not None and val > best_value:
                best_value = val
                best_result = circles
                
        # Try grid configuration (only for a subset to save time)
        if ar in [1.0, 1.2, 0.8]:
             if time.time() > deadline - 12.0:
                 break
             circles, val = _try_aspect(ar, deadline, use_grid=True)
             if circles is not None and val > best_value:
                 best_value = val
                 best_result = circles

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
>>>>>>> REPLACE
