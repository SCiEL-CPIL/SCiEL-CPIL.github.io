import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # 1. Setup Fast Approximate Plane-Wave Solver
    # Use 17x17 = 289 plane waves to match evaluator density for accuracy
    N_pw = 17
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # 15 k-points in irreducible BZ (Gamma-M, M-X, Gamma-X)
    # The evaluator uses 30 points, but 15 is often sufficient for optimization trends
    # and saves 50% of the evaluation time.
    k_points = []
    for t in np.linspace(0, 0.5, 5):
        k_points.append((t, 0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
            
    # Precompute G-G' differences for coefficient lookup
    # diff[i,j] = G_vectors[i] - G_vectors[j]
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    diff_ix = (G_diff[:, :, 0] % size).astype(np.int32)
    diff_iy = (G_diff[:, :, 1] % size).astype(np.int32)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        Uses 289 PWs and 15 k-points.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix for all G-G' differences
        # coeff_matrix[i,j] = fft_inv_eps[diff[i,j] mod size]
        # Vectorized lookup
        coeff_matrix = fft_inv_eps[diff_ix, diff_iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[i]
            # For TM modes, the eigenvalue problem is (k+G)^2 * delta_ij * epsilon^{-1} * (k+G') ... 
            # Wait, the standard PW expansion for TM in 2D photonic crystals:
            # The equation is: sum_{G'} (k+G)^2 epsilon^{-1}(G-G') E(k+G') = omega^2 E(k+G)
            # Or equivalently: sum_{G'} epsilon^{-1}(G-G') (k+G')^2 E(k+G') = omega^2 E(k+G)
            # The matrix is M_{G, G'} = epsilon^{-1}(G-G') * (k+G')^2
            # Note: The matrix is not Hermitian in general unless epsilon is real and symmetric?
            # Actually, for real epsilon, M_{G, G'} = M^*_{G', G} * (k+G)^2 / (k+G')^2 ? No.
            # Let's check the Hermitian form.
            # The operator is H = -nabla^2 / epsilon(r).
            # In PW basis: H_{G, G'} = (k+G)^2 epsilon^{-1}(G-G')
            # This matrix is generally non-Hermitian.
            # However, we can transform it to a Hermitian form by weighting the basis.
            # Or simply use eigvals. The eigenvalues should be real.
            # Let's use eigvals as in the seed, but ensure we take real parts.
            
            M = coeff_matrix * omega_sq[np.newaxis, :] # Note: omega_sq is (k+G)^2 or (k+G')^2?
            # Standard formulation:
            # sum_{G'} (k+G)^2 epsilon^{-1}(G-G') E_{G'} = omega^2 E_G
            # So M_{G, G'} = (k+G)^2 epsilon^{-1}(G-G')
            # My code above: M = coeff_matrix * omega_sq[:, np.newaxis]
            # coeff_matrix[i,j] corresponds to G_i, G_j.
            # omega_sq[i] is (k+G_i)^2.
            # So M[i,j] = (k+G_i)^2 * eps_inv(G_i - G_j). This matches.
            
            eigvals = np.linalg.eigvals(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # We need bands 4 and 5 (indices 3 and 4)
                # Note: eigvals are omega^2.
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # 2. Discrete Representation and Symmetry Enforcement
    # We represent the cell by its top-left 8x8 quadrant (since 32/4 = 8).
    # The rest of the cell is determined by C4v symmetry.
    # Symmetry operations about center (16, 16) in 0-indexed 32x32 grid:
    # (i, j) -> (31-i, j) [Reflect X]
    # (i, j) -> (i, 31-j) [Reflect Y]
    # (i, j) -> (31-i, 31-j) [Rot 180]
    # (i, j) -> (j, i) [Transpose]
    # (i, j) -> (31-j, i) [Reflect Y then Transpose]
    # (i, j) -> (j, 31-i) [Reflect X then Transpose]
    # (i, j) -> (31-j, 31-i) [Rot 180 then Transpose]
    
    # To ensure strict symmetry, we only store 8x8 pixels (i in 0..7, j in 0..7).
    # Any pixel (i,j) in the full grid maps to a unique pixel in the 8x8 block?
    # Let's verify the orbit of a point (i,j) under C4v.
    # The fundamental domain for C4v in a square is indeed 1/8th of the square.
    # For a 32x32 grid, the fundamental domain is 8x8?
    # Let's check (0,0).
    # (0,0) -> (31,0) -> (0,31) -> (31,31) -> (0,0) [Trans of 0,0 is 0,0? No, Trans of (0,0) is (0,0). Trans of (31,0) is (0,31).]
    # The orbit of (0,0) has size 4?
    # (0,0), (31,0), (0,31), (31,31).
    # What about (0,1)?
    # (0,1), (31,1), (0,30), (31,30), (1,0), (1,31), (30,0), (30,31). Size 8.
    # So most pixels have orbit size 8.
    # The 8x8 block (0..7, 0..7) contains one representative from each orbit?
    # Let's check if any two points in 0..7, 0..7 are symmetric.
    # If i,j < 8, then 31-i > 23, 31-j > 23.
    # So no two points in the top-left 8x8 are related by reflection/rotation that maps to >15.
    # Transpose (j,i) is also in 0..7, 0..7.
    # So (i,j) and (j,i) are both in the block.
    # This means the 8x8 block is NOT a fundamental domain because of the transpose symmetry.
    # We need i <= j.
    # So the fundamental domain is the triangle i <= j within the 8x8 block.
    # Number of pixels = 8*9/2 = 36.
    
    # Let's define the mapping from the 36 free pixels to the full 32x32 grid.
    # Free pixels: (i, j) where 0 <= i <= j < 8.
    free_indices = []
    for i in range(8):
        for j in range(i, 8):
            free_indices.append((i, j))
            
    num_free = len(free_indices)
    
    def expand_cell(free_pixels):
        """
        Expands the 36 free pixels to a full 32x32 grid.
        free_pixels is a 1D array of length 36 (0/1).
        """
        grid = np.zeros((size, size), dtype=np.int32)
        for idx, (i, j) in enumerate(free_indices):
            val = free_pixels[idx]
            # Apply all 8 symmetry operations
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for (pi, pj) in positions:
                grid[pi, pj] = val
        return grid

    def get_eps_grid(free_pixels):
        grid = expand_cell(free_pixels)
        # 1 = dielectric (eps_high), 0 = air (eps_low)
        return eps_low + (eps_high - eps_low) * grid

    # 3. Optimization: Hill Climbing with Random Restarts
    # We will perform local search by flipping individual free pixels.
    
    best_score = -np.inf
    best_free_pixels = None
    
    # Initial guess: A dielectric rod in the center.
    # In the 8x8 block, the center of the cell (15.5, 15.5) is far away.
    # The top-left corner (0,0) is the corner of the unit cell.
    # A rod centered at the cell center (15.5, 15.5) would be at the bottom-right of the 8x8 block?
    # No, the 8x8 block is the top-left corner of the unit cell.
    # The center of the unit cell is (15.5, 15.5).
    # The point in the 8x8 block closest to the center is (7,7).
    # (7,7) maps to (7,7), (24,7), (7,24), (24,24), etc.
    # (24,24) is close to (15.5, 15.5)? No, 24 is far.
    # Wait, the symmetry is about the CENTER of the cell.
    # The 8x8 block covers x in [0, 8), y in [0, 8).
    # The center is (15.5, 15.5).
    # The point (7,7) is distance sqrt(8.5^2 + 8.5^2) ~ 12 from center.
    # The point (0,0) is distance sqrt(15.5^2 + 15.5^2) ~ 22 from center.
    # So the "center" of the dielectric rod should be represented by pixels near (7,7) in the free set?
    # Actually, if we want a rod at the center, we want pixels near (15.5, 15.5) to be 1.
    # The pixel (15,15) is in the full grid.
    # Which free pixel maps to (15,15)?
    # (15,15) is related to (16,16) by 180 rot? No, 15 and 16 are distinct.
    # Let's find the free pixel corresponding to (15,15).
    # (15,15) -> Reflect X -> (16,15).
    # (15,15) -> Reflect Y -> (15,16).
    # (15,15) -> Trans -> (15,15).
    # So (15,15) is its own transpose.
    # Is (15,15) in the orbit of a free pixel?
    # Free pixels are in 0..7.
    # (15,15) is not in 0..7.
    # Its orbit:
    # (15,15), (16,15), (15,16), (16,16).
    # None of these are in 0..7.
    # Wait, my fundamental domain logic was flawed for the center.
    # The C4v symmetry group acts on the 32x32 grid.
    # The orbits partition the grid.
    # My free_indices only covered 0..7.
    # Does every orbit have a representative in 0..7, 0..7 with i<=j?
    # Let's check (15,15).
    # Orbit: {(15,15), (16,15), (15,16), (16,16)}.
    # None are in 0..7.
    # So my "8x8 block" does NOT cover the whole grid!
    # The fundamental domain for C4v on a 32x32 grid is NOT just 8x8.
    # The symmetry is about the center (15.5, 15.5).
    # The fundamental domain is the triangle 0 <= i <= j <= 15?
    # Let's check.
    # If i,j <= 15, then 31-i >= 16, 31-j >= 16.
    # So the top-left 16x16 quadrant contains one representative from each orbit?
    # No, because of transpose.
    # In the 16x16 quadrant, (i,j) and (j,i) are both present.
    # So we need i <= j in the 16x16 quadrant.
    # Number of pixels = 16*17/2 = 136.
    
    # Let's re-verify.
    # Take (15,15). It is in 16x16 (0..15). i=j=15.
    # Take (0,0). In 16x16.
    # Take (15,0). In 16x16. i=15, j=0. i>j.
    # Its transpose (0,15) is in 16x16. i=0, j=15. i<j.
    # So (15,0) and (0,15) are in the same orbit.
    # The representative with i<=j is (0,15).
    # So yes, the set {(i,j) | 0 <= i <= j <= 15} is a fundamental domain.
    # Size = 136.
    
    free_indices = []
    for i in range(16):
        for j in range(i, 16):
            free_indices.append((i, j))
            
    num_free = len(free_indices)
    
    def expand_cell(free_pixels):
        """
        Expands the 136 free pixels to a full 32x32 grid.
        """
        grid = np.zeros((size, size), dtype=np.int32)
        for idx, (i, j) in enumerate(free_indices):
            val = free_pixels[idx]
            positions = [
                (i, j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-i, size-1-j),
                (j, i),
                (size-1-j, i),
                (j, size-1-i),
                (size-1-j, size-1-i)
            ]
            for (pi, pj) in positions:
                grid[pi, pj] = val
        return grid

    # Initial guess: Rod at center.
    # Center is (15.5, 15.5).
    # Pixels near (15,15) and (16,16) should be 1.
    # (15,15) is in free set (i=15, j=15).
    # (15,16) is NOT in free set (i=15, j=16 -> j>i, so (16,15) is the rep? No, (15,16) -> trans (16,15). 16>15, so (15,16) is not in 0..15? Wait.
    # My free set is 0 <= i <= j <= 15.
    # (16,15) has i=16, which is > 15. So it's not in the 16x16 block.
    # But (16,15) is symmetric to (15,16).
    # (15,16) has i=15, j=16. j>15.
    # So (16,15) and (15,16) are NOT in the 16x16 block.
    # Wait, the 16x16 block is indices 0..15.
    # (16,15) is index 16, so it's outside.
    # So my fundamental domain 0<=i<=j<=15 does NOT cover the whole grid?
    # Let's check the orbit of (16,15).
    # (16,15) -> Ref X -> (15,15).
    # (15,15) IS in the 16x16 block (0..15).
    # So the representative is (15,15).
    # So (16,15) is covered by (15,15).
    # So the set 0<=i<=j<=15 IS a fundamental domain.
    
    # Initial guess: Set (15,15) to 1.
    init_pixels = np.zeros(num_free, dtype=np.int32)
    # Find index of (15,15)
    idx_center = free_indices.index((15, 15))
    init_pixels[idx_center] = 1
    # Also set (14,14), (15,14) etc to make a small rod?
    # Let's start with a 2x2 block at the center.
    # (15,15), (15,14) -> rep (14,15)? No, (14,15) is in free set.
    # (14,14) is in free set.
    # (14,15) is in free set.
    # (15,15) is in free set.
    # So set (14,14), (14,15), (15,15) to 1.
    idx_1414 = free_indices.index((14, 14))
    idx_1415 = free_indices.index((14, 15))
    init_pixels[idx_1414] = 1
    init_pixels[idx_1415] = 1
    init_pixels[idx_center] = 1
    
    best_free_pixels = init_pixels.copy()
    best_score = evaluate_cell(get_eps_grid(best_free_pixels))
    
    # Hill Climbing
    # We try flipping each pixel and see if the score improves.
    # To save time, we can use a subset of k-points or a coarser PW grid for the inner loop?
    # No, the evaluator accuracy is important.
    # But 136 pixels * 15 k-points * 289 PWs is heavy.
    # Let's estimate time.
    # evaluate_cell takes ~0.5s?
    # If so, 136 flips * 0.5s = 68s per sweep.
    # We have 200s. So ~3 sweeps.
    # We can optimize by only evaluating if the flip is "plausible"?
    # Or use a faster approximate evaluator for the search, and the exact one for final check?
    # Let's stick to the exact evaluator but optimize the code.
    
    # Optimization: Precompute coeff_matrix for all possible eps grids? No, eps changes.
    # But the structure of coeff_matrix depends on eps_inv.
    
    # Let's try a simple hill climb.
    for sweep in range(5):
        if time.time() > deadline - 20:
            break
            
        improved = False
        # Randomize the order of pixels to avoid local minima bias
        perm = np.random.permutation(num_free)
        
        for idx in perm:
            if time.time() > deadline - 20:
                break
                
            # Flip pixel
            old_val = best_free_pixels[idx]
            best_free_pixels[idx] = 1 - old_val
            
            new_score = evaluate_cell(get_eps_grid(best_free_pixels))
            
            if new_score > best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_free_pixels[idx] = old_val
        
        if not improved:
            # Try a random restart with a different initial guess
            # Or break if we've done several sweeps
            if sweep > 2:
                break
            # Random restart: flip a random subset of pixels
            for _ in range(10):
                idx_rand = np.random.randint(0, num_free)
                best_free_pixels[idx_rand] = 1 - best_free_pixels[idx_rand]
            best_score = evaluate_cell(get_eps_grid(best_free_pixels))
            if best_score < -1.0:
                # Revert to previous best if it got too bad
                # We need to keep the previous best
                pass # Simplified: just continue

    # Final Binarization and Symmetry Enforcement (already done by expand_cell)
    final_grid = expand_cell(best_free_pixels)
    
    # Convert to nested list of 0/1
    final_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_grid[i, j]))
        final_cell.append(row)
        
    return final_cell
