<<<<<<< SEARCH
    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # If time remains, try local perturbations around best design
    if time.time() - start_time < deadline - 20:
        base = best_cell
        for flip_i in range(0, size, 2):
            for flip_j in range(0, size, 2):
                if time.time() - start_time > deadline:
                    break
                # Try flipping a small patch
                cell_trial = [row[:] for row in base]
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = (flip_i + di) % size
                        nj = (flip_j + dj) % size
                        cell_trial[ni][nj] = 1 - base[ni][nj]
                cell_sym = enforce_d4_symmetry(cell_trial)
                # Quick check: only evaluate if it's different from base
                if cell_sym != base:
                    score = approximate_gap_ratio(cell_sym)
                    if score > best_score:
                        best_score = score
                        best_cell = cell_sym
                        base = best_cell
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell
=======
    # Generate candidate designs
    candidates = []
    
    # Single rods with various radii
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        candidates.append(generate_single_rod(r))
    
    # Rings
    for inner, outer in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.40), (0.10, 0.25), (0.20, 0.30)]:
        candidates.append(generate_ring(inner, outer))
    
    # Four rods
    for r in [0.10, 0.12, 0.15, 0.18, 0.20]:
        candidates.append(generate_four_rods(r))
    
    # Crosses
    for w, l in [(0.15, 0.40), (0.20, 0.35), (0.25, 0.30), (0.15, 0.35)]:
        candidates.append(generate_cross(w, l))
    
    # Checkerboard blocks
    for bs in [4, 6, 8]:
        candidates.append(generate_checkerboard_block(bs))
    
    # Concentric rings
    for radii in [(0.0, 0.20), (0.0, 0.25, 0.40), (0.15, 0.30, 0.45)]:
        candidates.append(generate_concentric_rings(radii))
    
    # Add more diverse patterns
    # Square ring patterns
    def generate_square_ring(thickness_frac, outer_frac):
        cell = [[0] * size for _ in range(size)]
        t = int(thickness_frac * size)
        o = int(outer_frac * size)
        cx = size // 2
        cy = size // 2
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx + 0.5) - 0.5
                dy = abs(i - cy + 0.5) - 0.5
                max_dist = max(abs(dx), abs(dy))
                if o - t <= max_dist < o:
                    cell[i][j] = 1
        return cell
    
    for t, o in [(2, 10), (3, 12), (4, 14), (2, 12), (3, 14)]:
        candidates.append(generate_square_ring(t / size, o / size))
    
    # Diagonal cross patterns
    def generate_diagonal_cross(width):
        cell = [[0] * size for _ in range(size)]
        half_w = width // 2
        for i in range(size):
            for j in range(size):
                dx = abs(i - j)
                dy = abs(i + j - (size - 1))
                if dx <= half_w or dy <= half_w:
                    cell[i][j] = 1
        return cell
    
    for w in [2, 3, 4, 5]:
        candidates.append(generate_diagonal_cross(w))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for cell in candidates:
        if time.time() - start_time > deadline:
            break
        cell_sym = enforce_d4_symmetry(cell)
        score = approximate_gap_ratio(cell_sym)
        if score > best_score:
            best_score = score
            best_cell = cell_sym
    
    # If time remains, try local perturbations around best design
    if time.time() - start_time < deadline - 15:
        base = best_cell
        improved = True
        while improved and time.time() - start_time < deadline - 10:
            improved = False
            for flip_i in range(0, size, 1):
                for flip_j in range(0, size, 1):
                    if time.time() - start_time > deadline:
                        break
                    # Try flipping a single pixel and its symmetric counterparts
                    cell_trial = [row[:] for row in base]
                    # Apply D4 symmetry to the flip
                    flips = [
                        (flip_i, flip_j),
                        (size - 1 - flip_i, flip_j),
                        (flip_i, size - 1 - flip_j),
                        (size - 1 - flip_i, size - 1 - flip_j),
                        (flip_j, flip_i),
                        (size - 1 - flip_j, size - 1 - flip_i),
                        (flip_j, size - 1 - flip_i),
                        (size - 1 - flip_j, flip_i),
                    ]
                    for fi, fj in flips:
                        cell_trial[fi][fj] = 1 - base[fi][fj]
                    
                    cell_sym = enforce_d4_symmetry(cell_trial)
                    # Quick check: only evaluate if it's different from base
                    if cell_sym != base:
                        score = approximate_gap_ratio(cell_sym)
                        if score > best_score:
                            best_score = score
                            best_cell = cell_sym
                            base = best_cell
                            improved = True
    
    final_cell = enforce_d4_symmetry(best_cell)
    return final_cell
>>>>>>> REPLACE
