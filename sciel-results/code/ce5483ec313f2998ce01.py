def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    rng = random.Random(42)
    
    # Initial: 4x4 Grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    current_score, has_5 = _score_lines(points)
    if has_5:
        # Fallback
        points = []
        while len(points) < 16:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if [x, y] not in points:
                points.append([x, y])
        current_score, has_5 = _score_lines(points)
        
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # Precompute pairs for speed? No, N=16 is small.
    
    while time.time() < deadline:
        # Pick a point to move
        idx = rng.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Decide new position
        if rng.random() < 0.3:
            # Targeted: Try to join a 3-point line
            # This requires knowing 3-point lines. 
            # Approximation: Just do random moves for now, they are fast.
            # To do targeted moves efficiently, we'd need to cache line info.
            # Given the 120s limit, random moves with high frequency are safer.
            pass
            
        # Random move
        if rng.random() < 0.5:
            step = rng.randint(1, 30)
            new_x = old_x + rng.randint(-step, step)
            new_y = old_y + rng.randint(-step, step)
        else:
            new_x = rng.randint(-1000, 1000)
            new_y = rng.randint(-1000, 1000)
            
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        valid = True
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                valid = False
                break
                
        if not valid:
            continue
            
        # Apply move
        points[idx] = [new_x, new_y]
        
        # Score
        new_score, has_5 = _score_lines(points)
        
        if has_5:
            # Revert
            points[idx] = [old_x, old_y]
            continue
            
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
        elif new_score == current_score:
            # Accept with small probability to explore
            if rng.random() < 0.05:
                current_score = new_score
            else:
                # Revert
                points[idx] = [old_x, old_y]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
    # Final check
    final_score, final_has_5 = _score_lines(best_points)
    if final_has_5:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
