import time
import math
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 pixel unit cell to maximize the TM band gap 
    between the 4th and 5th bands.
    
    Strategy:
    1. Use a local plane-wave solver with a sufficiently large basis set to accurately
       rank candidate designs.
    2. Explore a diverse set of D4-symmetric geometries:
       - Square pillars with various corner treatments (superellipse exponent).
       - Concentric rings (dielectric annuli).
       - Cross shapes.
       - Four-rod configurations.
       - "Hollow square" or frame shapes.
    3. Perform a local optimization on the best candidates by perturbing pixels
       on the boundary of the dielectric shape to fine-tune the Fourier coefficients.
    4. Enforce strict D4 symmetry.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill

    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry: row/col reversal and transpose invariance."""
        # We want the final cell to be exactly symmetric.
        # If there's a conflict (e.g. due to pixel discretization), we take the majority or a specific rule.
        # For a binary mask, if we generate it symmetrically, it should be fine.
        # But if we perturb, we must update all 8 symmetric pixels together.
        # This function is used to clean up a potentially asymmetric input or to construct from a quadrant.
        
        final_cell = [[0] * size for _ in range(size)]
        
        # Iterate over the fundamental domain (upper-right triangle including diagonal)
        # Actually, simpler: iterate all, but only set if it's the "canonical" representative?
        # Or just average? No, binary.
        
        # Let's iterate over all pixels, and for each, look at its orbit.
        # If any in the orbit is 1, and we want to preserve "mass", maybe keep it?
        # But for our generators, we generate symmetric directly.
        # For perturbations, we will apply to the whole orbit.
        
        # This function is mainly for verification or if we started with an asymmetric seed.
        # Here we assume input is mostly symmetric or we want to force symmetry.
        
        for i in range(size):
            for j in range(size):
                # Check if this is the "first" in the orbit?
                # Or just check the value. If we want to be robust:
                # Take the value from the "most central" or just the first one found?
                # Let's just copy the value from the pixel itself, but ensure the whole orbit is set to the same value.
                # This implies we need to do it in a way that doesn't overwrite.
                
                # Better approach:
                # 1. Create a set of orbits.
                # 2. For each orbit, decide the value (e.g., max or first non-zero).
                # 3. Assign to all.
                
                pass # Placeholder, see below

        # Implementation:
        # Map each pixel to its canonical representative (min i, then min j in the orbit)
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                
                # Generate orbit
                orbit = set()
                orbit.add((i, j))
                orbit.add((size-1-i, j))
                orbit.add((i, size-1-j))
                orbit.add((size-1-i, size-1-j))
                orbit.add((j, i))
                orbit.add((size-1-j, size-1-i))
                orbit.add((j, size-1-i))
                orbit.add((size-1-j, i))
                
                # Decide value: if any is 1, make all 1? Or if majority?
                # For our purposes, we generate symmetrically, so this shouldn't be needed.
                # But if it is, let's use "any" (OR) to preserve dielectric.
                val = 0
                for (oi, oj) in orbit:
                    if cell[oi][oj] == 1:
                        val = 1
                        break
                
                for (oi, oj) in orbit:
                    final_cell[oi][oj] = val
                    visited.add((oi, oj))
        
        return final_cell

    def compute_gap_ratio(cell, max_order=12, k_points=None):
        """
        Compute the approximate gap ratio using plane-wave expansion.
        Returns -1 if invalid.
        """
        # Convert cell to numpy array
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of 1/eps
        # We need coefficients for G = (m, n) where |m|, |n| <= max_order
        # Using FFT
        fft_2d = np.fft.fft2(inv_eps)
        
        def get_coeff(m, n):
            # fft2 gives F[m, n] = sum_{x,y} f[x,y] exp(-i 2pi (mx/N + ny/N))
            # We want c[m,n] such that f(x,y) = sum c[m,n] exp(i 2pi (mx/N + ny/N))
            # Inverse FFT: f[x,y] = (1/N^2) sum F[m,n] exp(i 2pi (mx/N + ny/N))
            # So c[m,n] = F[m,n] / N^2
            # Indices: m can be negative. FFT index is m % N.
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx] / (size * size)

        if k_points is None:
            # Irreducible BZ for square lattice: 0 <= kx <= ky <= 0.5
            # Key points: Gamma (0,0), X (0.5, 0), M (0.5, 0.5), and some interior.
            k_points = [
                (0.0, 0.0),
                (0.25, 0.0),
                (0.5, 0.0),
                (0.25, 0.25),
                (0.5, 0.5),
                (0.1, 0.1),
                (0.3, 0.3),
                (0.4, 0.1),
                (0.1, 0.4), # Not in irreducible if kx>ky, but let's stick to kx<=ky
                # Correct irreducible: kx >= 0, ky >= kx, kx+ky <= 0.5? No.
                # Square BZ: [-0.5, 0.5]^2. Irreducible wedge: 0 <= kx <= ky <= 0.5.
                # Points:
                # (0,0), (0.5,0), (0.5,0.5)
                # Line kx=ky: (0.25, 0.25)
                # Interior: (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)
            ]
            k_points = [
                (0.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.25, 0.25),
                (0.1, 0.2),
                (0.2, 0.3),
                (0.3, 0.4),
                (0.15, 0.15),
                (0.35, 0.35),
                (0.45, 0.05) # kx>ky, map to (0.05, 0.45)
            ]
            # Ensure kx <= ky
            k_points = [(min(kx, ky), max(kx, ky)) for kx, ky in k_points]
            # Remove duplicates
            k_points = list(set(k_points))

        # Plane wave basis: G vectors
        # We need enough G vectors to resolve the dielectric structure.
        # max_order=12 means G components from -12 to 12.
        # Total G vectors: (2*12+1)^2 = 25^2 = 625.
        # This might be slow for many k-points.
        # Let's try max_order=8 first (17x17=289 vectors).
        
        g_vectors = []
        for m in range(-max_order, max_order + 1):
            for n in range(-max_order, max_order + 1):
                g_vectors.append((m, n))
        
        N = len(g_vectors)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build Hamiltonian matrix H
            # H[i, j] = |k + G_i|^2 * < (k+G_i) | 1/eps | (k+G_j) >
            # For TM polarization, H_ij = (k+G_i) dot (k+G_j) * eps_inv[G_i - G_j]
            # Wait, standard form:
            # (k+G_i)^2 * a_i = sum_j (k+G_i) dot (k+G_j) * eps_inv[G_i - G_j] * a_j
            # This is not Hermitian if we just do that?
            # Actually, the eigenvalue problem is:
            # sum_j H_ij a_j = omega^2 a_i
            # H_ij = (k+G_i) dot (k+G_j) * eps_inv[G_i - G_j]
            # This matrix is Hermitian because eps_inv[-G] = conj(eps_inv[G]) and dot product is symmetric.
            
            H = np.zeros((N, N), dtype=complex)
            
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                k_sq = kx_total**2 + ky_total**2
                
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    
                    # Check if difference is within computed FFT range?
                    # FFT covers all integer differences modulo size.
                    # Since max_order < size/2, the difference is unique and within range.
                    eps_inv_G = get_coeff(m_diff, n_diff)
                    
                    kx_G = kx + G[0]
                    ky_G = ky + G[1]
                    dot = kx_total * kx_G + ky_total * ky_G
                    
                    H[i, j] = dot * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            # We want bands 4 and 5 (index 3 and 4)
            # Note: There might be spurious low-frequency modes?
            # In a unit cell, the modes are ordered by frequency.
            # Band 1 is the lowest.
            
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    def generate_superellipse(n_exp, R):
        """Generate a superellipse (squircle/square) centered in the cell."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5 - cx) / size
                y = (i + 0.5 - cy) / size
                
                # Superellipse: |x|^n + |y|^n <= R^n
                # Normalize R to be in same units as x,y (fraction of cell size)
                # If R=0.25, it covers 25% of the width from center.
                val = abs(x)**n_exp + abs(y)**n_exp
                if val <= R**n_exp:
                    cell[i][j] = 1
                    
        return cell

    def generate_ring(inner_frac, outer_frac):
        """Generate a dielectric ring."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5 - cx) / size
                y = (i + 0.5 - cy) / size
                r = math.sqrt(x*x + y*y)
                
                if inner_frac <= r < outer_frac:
                    cell[i][j] = 1
                    
        return cell

    def generate_cross(arm_width, arm_length):
        """Generate a cross shape."""
        cell = [[0] * size for _ in range(size)]
        cx = size // 2
        cy = size // 2
        
        half_w = arm_width // 2
        
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx)
                dy = abs(i - cy)
                
                # Horizontal arm: |dy| <= half_w and |dx| <= arm_length
                # Vertical arm: |dx| <= half_w and |dy| <= arm_length
                if (dy <= half_w and dx <= arm_length) or (dx <= half_w and dy <= arm_length):
                    cell[i][j] = 1
                    
        return cell

    def generate_four_rods(radius_frac, pos_frac=0.25):
        """Generate four rods at the corners of a central square."""
        cell = [[0] * size for _ in range(size)]
        cx = size / 2.0
        cy = size / 2.0
        
        # Centers at (cx +/- offset, cy +/- offset)
        offset = pos_frac * size
        
        centers = [
            (cx - offset, cy - offset),
            (cx + offset, cy - offset),
            (cx - offset, cy + offset),
            (cx + offset, cy + offset)
        ]
        
        r_px = radius_frac * size
        
        for i in range(size):
            for j in range(size):
                x = j + 0.5
                y = i + 0.5
                for ccx, ccy in centers:
                    dx = x - ccx
                    dy = y - ccy
                    if dx*dx + dy*dy < r_px*r_px:
                        cell[i][j] = 1
                        break
                        
        return cell

    def generate_hollow_square(outer_size, inner_size):
        """Generate a square frame."""
        cell = [[0] * size for _ in range(size)]
        cx = size // 2
        cy = size // 2
        
        half_out = outer_size // 2
        half_in = inner_size // 2
        
        for i in range(size):
            for j in range(size):
                dx = abs(j - cx)
                dy = abs(i - cy)
                
                # Inside outer square
                if dx <= half_out and dy <= half_out:
                    # Outside inner square
                    if not (dx <= half_in and dy <= half_in):
                        cell[i][j] = 1
                        
        return cell

    # --- Search Phase ---
    
    best_score = -1.0
    best_cell = None
    
    # Candidate generators
    candidates = []
    
    # 1. Superellipses (Square-like)
    # R in range 0.2 to 0.4 (fraction of size)
    # n_exp in range 4 to 20
    for R in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        for n_exp in [4, 6, 8, 10, 12, 16, 20]:
            candidates.append(generate_superellipse(n_exp, R))
            
    # 2. Rings
    for inner in [0.10, 0.15, 0.20]:
        for outer in [0.25, 0.30, 0.35, 0.40]:
            if outer > inner:
                candidates.append(generate_ring(inner, outer))
                
    # 3. Crosses
    for w in [4, 6, 8, 10]:
        for l in [10, 12, 14, 16]:
            if w < l:
                candidates.append(generate_cross(w, l))
                
    # 4. Four Rods
    for r in [0.10, 0.12, 0.15]:
        for pos in [0.20, 0.25, 0.30]:
            candidates.append(generate_four_rods(r, pos))
            
    # 5. Hollow Squares
    for out in [12, 14, 16, 18]:
        for inn in [4, 6, 8]:
            if out > inn:
                candidates.append(generate_hollow_square(out, inn))

    # Evaluate candidates
    # Use a smaller basis set for initial screening?
    # Let's use max_order=8 for speed.
    
    scored_candidates = []
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        # Ensure symmetry
        cell = enforce_d4_symmetry(cell)
        
        score = compute_gap_ratio(cell, max_order=8)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
        scored_candidates.append((score, cell))
        
        # Optional: early exit if we found a very good score?
        # The seed score was 0.658. The inspiration found 9.0.
        # Let's keep going.

    # Sort by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local Optimization on the top candidates
    # We will try to flip boundary pixels to improve the score.
    
    top_k = 5
    if len(scored_candidates) > 0:
        for score, cell in scored_candidates[:top_k]:
            if time.time() > deadline - 20:
                break
                
            current_cell = [row[:] for row in cell]
            current_score = score
            
            # Identify boundary pixels
            # A pixel is on the boundary if it is 1 and has a 0 neighbor, or 0 and has a 1 neighbor.
            
            for step in range(100): # Max 100 local steps
                if time.time() > deadline - 10:
                    break
                    
                improved = False
                best_neighbor_cell = None
                best_neighbor_score = current_score
                
                # Get list of boundary pixels
                boundary_pixels = []
                for i in range(size):
                    for j in range(size):
                        val = current_cell[i][j]
                        # Check neighbors
                        is_boundary = False
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni = i + di
                                nj = j + dj
                                if 0 <= ni < size and 0 <= nj < size:
                                    if current_cell[ni][nj] != val:
                                        is_boundary = True
                                        break
                            if is_boundary:
                                break
                        if is_boundary:
                            boundary_pixels.append((i, j))
                            
                # Randomly sample a few boundary pixels to try flipping
                # To ensure D4 symmetry, we must flip the whole orbit.
                
                # Shuffle boundary pixels
                import random
                random.shuffle(boundary_pixels)
                
                # Try flipping up to 10 orbits
                tried = 0
                for (i, j) in boundary_pixels:
                    if tried >= 10:
                        break
                    if time.time() > deadline - 10:
                        break
                        
                    # Generate orbit
                    orbit = set()
                    orbit.add((i, j))
                    orbit.add((size-1-i, j))
                    orbit.add((i, size-1-j))
                    orbit.add((size-1-i, size-1-j))
                    orbit.add((j, i))
                    orbit.add((size-1-j, size-1-i))
                    orbit.add((j, size-1-i))
                    orbit.add((size-1-j, i))
                    
                    # Skip if we already tried a pixel in this orbit?
                    # Simple check: if the first pixel of the orbit is already flipped?
                    # We just track tried count.
                    
                    # Create new cell
                    new_cell = [row[:] for row in current_cell]
                    current_val = current_cell[i][j]
                    new_val = 1 - current_val
                    
                    for (oi, oj) in orbit:
                        new_cell[oi][oj] = new_val
                        
                    # Evaluate
                    new_score = compute_gap_ratio(new_cell, max_order=8)
                    
                    if new_score > best_neighbor_score:
                        best_neighbor_score = new_score
                        best_neighbor_cell = new_cell
                        improved = True
                        
                    tried += 1
                    
                if improved:
                    current_cell = best_neighbor_cell
                    current_score = best_neighbor_score
                    if current_score > best_score:
                        best_score = current_score
                        best_cell = current_cell
                else:
                    # No improvement found in this batch
                    break

    # Final verification of best_cell
    if best_cell is None:
        # Fallback to a simple square
        best_cell = generate_superellipse(10, 0.25)
        
    best_cell = enforce_d4_symmetry(best_cell)
    
    return best_cell

# Quick self-test if run directly (not required but good practice)
if __name__ == "__main__":
    problem = {"pixels": 32, "eps_low": 1, "eps_high": 12, "lower_band": 4}
    cell = design(problem)
    print("Design generated.")
    # Print score for debugging
    # Note: This will take some time
    # score = compute_gap_ratio(cell, max_order=8)
    # print(f"Score: {score}")
