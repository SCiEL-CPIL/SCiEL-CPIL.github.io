<<<<<<< SEARCH
    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses
=======
    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        
        # Better grid initialization based on aspect ratio
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.4, 0.7]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3 or w0 < 0.3:
                continue
            
            # Determine grid dimensions to best fit N circles in w0 x h0
            # We want roughly N circles, so side ~ sqrt(N)
            # Adjust rows/cols to match aspect ratio
            base_side = int(np.sqrt(N))
            # Try a few nearby grid shapes
            best_grid = None
            best_score = -1
            
            for rows in range(max(1, base_side - 2), base_side + 3):
                for cols in range(max(1, base_side - 2), base_side + 3):
                    if rows * cols < N:
                        continue
                    # Calculate cell size
                    cw = w0 / cols
                    ch = h0 / rows
                    r_est = min(cw, ch) * 0.45
                    # Score: we want to maximize the minimum radius, and fill space
                    # Simple heuristic: prefer grids where cells are closer to square
                    squareness = min(cw, ch) / max(cw, ch)
                    score = squareness * min(cw, ch)
                    if score > best_score:
                        best_score = score
                        best_grid = (rows, cols)
            
            if best_grid is None:
                best_grid = (base_side, base_side)
                
            rows, cols = best_grid
            z = np.zeros(3 * N + 1)
            count = 0
            for row in range(rows):
                for col in range(cols):
                    if count >= N:
                        break
                    cx = (col + 0.5) / cols * w0
                    cy = (row + 0.5) / rows * h0
                    r0 = min(w0 / cols, h0 / rows) * 0.45
                    z[3 * count] = cx
                    z[3 * count + 1] = cy
                    z[3 * count + 2] = r0
                    count += 1
            
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses
>>>>>>> REPLACE

