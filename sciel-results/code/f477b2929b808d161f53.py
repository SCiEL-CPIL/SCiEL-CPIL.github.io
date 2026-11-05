<<<<<<< SEARCH
    # Generate and evaluate candidates
    best_score = -np.inf
    best_candidate = make_four_quadrant_cell(0.12, 2.0)
    
    # Coarse grid search over radius and exponent
    radii = [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]
    exponents = [2.0, 3.0, 4.0]
    
    candidates = []
    for r in radii:
        for e in exponents:
            candidates.append((r, e))
    
    # Sort by priority: start with likely good values
    candidates.sort(key=lambda x: abs(x[0] - 0.12) + abs(x[1] - 2.0))
    
    for (r, e) in candidates:
        if time.time() > deadline - 30:
            break
        cell = make_four_quadrant_cell(r, e)
        score = compute_tm_bands(cell)
        if score > best_score:
            best_score = score
            best_candidate = cell
            # Fine-tune around the best
            if best_score > 5.0 and time.time() < deadline - 60:
                # Do a local refinement
                for dr in [-0.01, 0.01]:
                    for de in [-0.5, 0.5]:
                        r2 = r + dr
                        e2 = e + de
                        if r2 < 0.03 or e2 < 1.5:
                            continue
                        cell2 = make_four_quadrant_cell(r2, e2)
                        score2 = compute_tm_bands(cell2)
                        if score2 > best_score:
                            best_score = score2
                            best_candidate = cell2
    
    return best_candidate
=======
    # Generate and evaluate candidates
    best_score = -np.inf
    best_candidate = make_four_quadrant_cell(0.12, 2.0)
    
    # Coarse grid search over radius and exponent
    radii = [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18]
    exponents = [2.0, 3.0, 4.0]
    
    candidates = []
    for r in radii:
        for e in exponents:
            candidates.append((r, e))
    
    candidates.sort(key=lambda x: abs(x[0] - 0.12) + abs(x[1] - 2.0))
    
    for (r, e) in candidates:
        if time.time() > deadline - 60:
            break
        cell = make_four_quadrant_cell(r, e)
        score = compute_tm_bands(cell)
        if score > best_score:
            best_score = score
            best_candidate = cell
    
    # Field-intensity-driven refinement
    if time.time() < deadline - 30 and best_score > 0:
        best_cell_arr = np.array(best_candidate, dtype=float)
        
        # Get eigenvectors at critical k-points
        _, crit_k4, crit_k5, eigvec4, eigvec5 = compute_tm_bands(
            best_candidate, n_pw=13, n_kpts=15, return_eigvecs=True
        )
        
        if eigvec4 is not None and eigvec5 is not None:
            # Compute field intensity distributions for bands 4 and 5
            # |E|^2 in real space from plane-wave coefficients
            half = 13 // 2
            g_indices = []
            for m in range(-half, half + 1):
                for n in range(-half, half + 1):
                    if m * m + n * n <= half * half + half:
                        g_indices.append((m, n))
            n_total = len(g_indices)
            
            # Compute intensity on a grid
            coords = (np.arange(size) + 0.5) / size - 0.5
            xx, yy = np.meshgrid(coords, coords, indexing='ij')
            
            def compute_intensity(eigvec, kx, ky):
                """Compute |E(x,y)|^2 from plane-wave eigenvector."""
                intensity = np.zeros((size, size), dtype=float)
                for i, (m, n) in enumerate(g_indices):
                    gx = m + kx
                    gy = n + ky
                    phase = np.exp(2j * np.pi * (gx * xx + gy * yy))
                    intensity += np.abs(eigvec[i]) ** 2
                    # Cross terms
                    for j in range(i+1, n_total):
                        mj, nj = g_indices[j]
                        gxj = mj + kx
                        gyj = nj + ky
                        phase_j = np.exp(2j * np.pi * (gxj * xx + gyj * yy))
                        intensity += 2 * np.real(eigvec[i] * np.conj(eigvec[j]) * phase * np.conj(phase_j))
                return intensity
            
            # Intensity of band 4 at its critical k-point
            if crit_k4 is not None:
                int4 = compute_intensity(eigvec4, crit_k4[0], crit_k4[1])
            else:
                int4 = np.ones((size, size))
            
            # Intensity of band 5 at its critical k-point
            if crit_k5 is not None:
                int5 = compute_intensity(eigvec5, crit_k5[0], crit_k5[1])
            else:
                int5 = np.ones((size, size))
            
            # Normalize
            int4 = int4 / (np.max(int4) + 1e-10)
            int5 = int5 / (np.max(int5) + 1e-10)
            
            # Optimization: increase epsilon where band 5 has high intensity
            # and decrease epsilon where band 4 has high intensity
            # This pushes band 4 down and band 5 up
            num_iter = 0
            max_iter = 20
            learning_rate = 0.05
            
            current_cell = best_cell_arr.copy()
            current_score = best_score
            
            for iteration in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Compute gradient: d(score)/d(epsilon) ~ int5 - int4
                # We want to increase epsilon where int5 > int4
                gradient = int5 - int4
                gradient = gradient - np.mean(gradient)  # Zero mean to preserve average fill fraction
                
                # Update cell
                update = learning_rate * gradient
                new_cell = current_cell + update
                
                # Threshold to binary
                new_cell_bin = (new_cell > 0.5).astype(int)
                
                # Enforce D4 symmetry
                new_cell_sym = enforce_d4_symmetry(new_cell_bin)
                
                # Evaluate
                new_score = compute_tm_bands(new_cell_sym.tolist())
                
                if new_score > current_score:
                    current_cell = new_cell_sym.astype(float)
                    current_score = new_score
                    best_score = max(best_score, current_score)
                    best_candidate = new_cell_sym.tolist()
                    num_iter += 1
                else:
                    # Try smaller step
                    update = 0.5 * learning_rate * gradient
                    new_cell = current_cell + update
                    new_cell_bin = (new_cell > 0.5).astype(int)
                    new_cell_sym = enforce_d4_symmetry(new_cell_bin)
                    new_score = compute_tm_bands(new_cell_sym.tolist())
                    
                    if new_score > current_score:
                        current_cell = new_cell_sym.astype(float)
                        current_score = new_score
                        best_score = max(best_score, current_score)
                        best_candidate = new_cell_sym.tolist()
                        num_iter += 1
                    else:
                        break  # No improvement, stop
    
    return best_candidate
>>>>>>> REPLACE
