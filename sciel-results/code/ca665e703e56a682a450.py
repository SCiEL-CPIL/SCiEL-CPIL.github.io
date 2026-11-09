<<<<<<< SEARCH
def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    size = 32
    candidates = []
    
    # 1. Small Rod (radius 0.2)
    cell_rod = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Medium Rod (radius 0.3)
    cell_rod2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.3**2:
                cell_rod2[i][j] = 1
    candidates.append(cell_rod2)
    
    # 3. Checkerboard
    cell_cb = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 4. Inverted Checkerboard
    cell_icb = 1 - cell_cb
    candidates.append(cell_icb)
    
    # 5. Cross (bars)
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 6. Ring (annulus)
    cell_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 7. Large Cross
    cell_cross2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 4.5 or abs(j - 15.5) < 4.5:
                cell_cross2[i][j] = 1
    candidates.append(cell_cross2)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell_fast(c)
        if s > best_score:
            best_score = s
            best_cell = c.copy()
            
    # Local Search with Simulated Annealing
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Convert to list of lists for flip_pixel? 
    # flip_pixel expects list of lists. 
    # Let's make a helper to flip on numpy array directly for speed.
    
    def flip_pixel_np(cell_np, i, j):
        size = 32
        val = cell_np[i][j]
        new_val = 1 - val
        idxs = [
            (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
            (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
        ]
        seen = set()
        for r, c in idxs:
            if (r, c) not in seen:
                seen.add((r, c))
                cell_np[r][c] = new_val
        return cell_np

    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # Pick random quadrant pixel
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        new_cell = current_cell.copy()
        flip_pixel_np(new_cell, i, j)
        
        new_score = evaluate_cell_fast(new_cell)
        
        # Simulated Annealing acceptance
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if T > T_min:
                if np.random.random() < np.exp(delta / T):
                    current_cell = new_cell
                    current_score = new_score
            T *= cooling_rate
            
    return [list(row) for row in best_cell]
=======
def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # 10s safety margin before 230s limit
    
    size = 32
    candidates = []
    
    # 1. Small Rod (radius 0.2)
    cell_rod = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Medium Rod (radius 0.3)
    cell_rod2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.3**2:
                cell_rod2[i][j] = 1
    candidates.append(cell_rod2)
    
    # 3. Checkerboard
    cell_cb = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 4. Inverted Checkerboard
    cell_icb = 1 - cell_cb
    candidates.append(cell_icb)
    
    # 5. Cross (bars)
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5:
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 6. Ring (annulus) - known to produce larger band gaps
    cell_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 7. Large Cross
    cell_cross2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if abs(i - 15.5) < 4.5 or abs(j - 15.5) < 4.5:
                cell_cross2[i][j] = 1
    candidates.append(cell_cross2)
    
    # 8. Cross-hole (dielectric cross with air hole in center)
    cell_cross_hole = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if r < 0.1:  # Air hole in center
                cell_cross_hole[i][j] = 0
            elif abs(x) > 0.15 and abs(y) > 0.15:  # Air corners
                cell_cross_hole[i][j] = 0
    candidates.append(cell_cross_hole)
    
    # 9. Square ring (dielectric frame)
    cell_sq_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = max(abs(x), abs(y))
            if 0.12 < r < 0.38:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell_fast(c)
        if s > best_score:
            best_score = s
            best_cell = c.copy()
            
    # Local Search with Simulated Annealing
    current_cell = best_cell.copy()
    current_score = best_score
    
    def flip_pixel_np(cell_np, i, j):
        size = 32
        val = cell_np[i][j]
        new_val = 1 - val
        idxs = [
            (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
            (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
        ]
        seen = set()
        for r, c in idxs:
            if (r, c) not in seen:
                seen.add((r, c))
                cell_np[r][c] = new_val
        return cell_np
    
    def multi_flip(cell_np, n_flips):
        """Flip n_flips random quadrant pixels."""
        new_cell = cell_np.copy()
        for _ in range(n_flips):
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            flip_pixel_np(new_cell, i, j)
        return new_cell

    T = 0.05
    T_min = 0.001
    cooling_rate = 0.997
    no_improve_count = 0
    
    while time.time() < deadline:
        # Choose between single flip and multi-flip
        if np.random.random() < 0.7:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            new_cell = current_cell.copy()
            flip_pixel_np(new_cell, i, j)
        else:
            # Multi-flip (2-3 flips) to escape local minima
            n_flips = np.random.randint(2, 4)
            new_cell = multi_flip(current_cell, n_flips)
        
        new_score = evaluate_cell_fast(new_cell)
        
        # Simulated Annealing acceptance
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
        else:
            delta = new_score - current_score
            if T > T_min:
                if np.random.random() < np.exp(delta / T):
                    current_cell = new_cell
                    current_score = new_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
            T *= cooling_rate
        
        # Restart if stuck too long
        if no_improve_count > 200:
            # Random restart from best
            current_cell = best_cell.copy()
            current_score = best_score
            T = 0.05
            no_improve_count = 0
            
    return [list(row) for row in best_cell]
>>>>>>> REPLACE
