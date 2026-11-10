<<<<<<< SEARCH
def _local_search(points, max_iterations=1000, time_limit=5.0):
    """
    Perform local search by moving one point at a time to minimize ordinary lines.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    current_score = _count_ordinary_lines_fast(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(12):
            # Try several random nearby positions
            for _ in range(20):
                new_x = current_points[i][0] + random.randint(-3, 3)
                new_y = current_points[i][1] + random.randint(-3, 3)
                
                # Check if the new position is valid (distinct from other points)
                valid = True
                for j in range(12):
                    if j != i:
                        if current_points[j][0] == new_x and current_points[j][1] == new_y:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Create a new configuration
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
                # Evaluate
                new_score = _count_ordinary_lines_fast(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    
                    break  # Stop trying this point after improvement
        
        if not improved:
            # No improvement found, try a larger jump
            i = random.randint(0, 11)
            new_x = current_points[i][0] + random.randint(-10, 10)
            new_y = current_points[i][1] + random.randint(-10, 10)
            
            valid = True
            for j in range(12):
                if j != i:
                    if current_points[j][0] == new_x and current_points[j][1] == new_y:
                        valid = False
                        break
            
            if valid:
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                new_score = _count_ordinary_lines_fast(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
    
    return best_points, best_score
=======
def _local_search(points, max_iterations=1000, time_limit=5.0):
    """
    Perform local search by moving one point at a time to minimize ordinary lines.
    Uses a wider search neighborhood and tries structured moves.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    current_score = _count_ordinary_lines_fast(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Precompute set of occupied points for fast validity check
    occupied = set(tuple(p) for p in current_points)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(12):
            old_x, old_y = current_points[i]
            occupied.discard((old_x, old_y))
            
            # Try several random nearby positions with varying step sizes
            step_sizes = [1, 2, 3, 5, 8]
            for step in step_sizes:
                if time.time() - start_time > time_limit:
                    break
                for _ in range(10):
                    new_x = old_x + random.randint(-step, step)
                    new_y = old_y + random.randint(-step, step)
                    
                    # Check if the new position is valid (distinct from other points)
                    if (new_x, new_y) in occupied:
                        continue
                    
                    # Create a new configuration
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [new_x, new_y]
                    
                    # Evaluate
                    new_score = _count_ordinary_lines_fast(new_points)
                    
                    if new_score < current_score:
                        # Accept improvement
                        occupied.add((new_x, new_y))
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        
                        break  # Stop trying this step size after improvement
                
                if improved:
                    break
            
            occupied.add((old_x, old_y))
            if improved:
                break
        
        if not improved:
            # No improvement found, try a larger jump
            i = random.randint(0, 11)
            old_x, old_y = current_points[i]
            occupied.discard((old_x, old_y))
            
            step = random.randint(10, 30)
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)
            
            if (new_x, new_y) not in occupied:
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                new_score = _count_ordinary_lines_fast(new_points)
                
                if new_score < current_score:
                    occupied.add((new_x, new_y))
                    current_points = new_points
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
            
            occupied.add((old_x, old_y))
    
    return best_points, best_score
>>>>>>> REPLACE

Now let me update the `construct` function to prioritize the hexagon+hexagram configuration and use more time effectively:

