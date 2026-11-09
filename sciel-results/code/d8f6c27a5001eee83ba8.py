import numpy as np
import time

def design(problem):
    """
    Direct pixel-level search using exact 289-PW non-Hermitian solver.
    Uses C4-symmetric local mutations (hill climbing) to maximize the
    gap-midgap ratio between TM bands 4 and 5.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 230s. We stop searching at 215s.
    deadline = start_time + 215.0
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Precomputation for the Evaluator's Method ---
    # The evaluator uses 289 plane waves (17x17 grid) and exact Fourier coefficients.
    # G vectors are integers from -8 to 8.
    G_range = np.arange(-8, 9)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors) # 289
    
    # Precompute G_diff matrix (num_pw, num_pw, 2)
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    # Flatten G_diff indices for vectorized lookup
    # We need coeff[i, j] = fft_inv_eps[G_diff[i,j,0] % size, G_diff[i,j,1] % size]
    # To make this fast, we can precompute the indices into the FFT array
    diff_x_flat = (G_diff[:, :, 0] % size).flatten()
    diff_y_flat = (G_diff[:, :, 1] % size).flatten()
    
    # K-points: 30 points covering the irreducible BZ
    # The evaluator uses 30 wave vectors. We approximate this set.
    # Typical set: Gamma-M, M-X, Gamma-X, and interior points.
    # Let's use a dense set of 30 points in the irreducible wedge.
    # Wedge: 0 <= kx <= ky, kx+ky <= 0.5 (in units of 2pi/a? No, usually k is in units of 1/a or similar)
    # The problem statement says "30 wave vectors covering the boundary and interior".
    # Standard irreducible BZ for square lattice: triangle with vertices (0,0), (0.5,0), (0.25,0.25).
    # Let's generate 30 points uniformly in this triangle.
    k_points = []
    # Add high-symmetry points
    k_points.append((0, 0))       # Gamma
    k_points.append((0.5, 0))     # X
    k_points.append((0.25, 0.25)) # M
    # Fill interior
    # We need 27 more points.
    # Use a grid in the triangle
    n_grid = 6
    for i in range(n_grid + 1):
        for j in range(n_grid + 1):
            kx = 0.5 * i / n_grid
            # ky range: 0 to 0.5 - kx
            max_ky = 0.5 - kx
            if max_ky < 0: continue
            for l in range(int(n_grid * max_ky / 0.5) + 1):
                ky = max_ky * l / (int(n_grid * max_ky / 0.5) + 1) if int(n_grid * max_ky / 0.5) > 0 else 0
                if kx <= ky + 1e-9: # Irreducible: kx <= ky
                    k_points.append((kx, ky))
    
    # Deduplicate and limit to 30
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 4), round(k[1], 4))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    # Ensure we have exactly 30 (or close). If we have more, subsample.
    # If we have less, add random points in the wedge.
    if len(unique_k) > 30:
        unique_k = unique_k[:30]
    while len(unique_k) < 30:
        kx = np.random.uniform(0, 0.5)
        ky = np.random.uniform(kx, 0.5 - kx)
        unique_k.append((kx, ky))
    
    # Precompute (k+G)^2 for all k points and G vectors
    # k_sq[k_idx, g_idx] = |k_k + G_g|^2
    k_sq = np.zeros((len(unique_k), num_pw))
    for idx, (kx, ky) in enumerate(unique_k):
        k_vec = np.array([kx, ky])
        k_plus_G = k_vec + G_vectors.T # (num_pw, 2)
        k_sq[idx, :] = np.sum(k_plus_G ** 2, axis=1)
    
    def evaluate_cell(binary_cell):
        """
        Computes the score using the exact non-Hermitian eigenvalue method.
        binary_cell: 32x32 array of 0s and 1s.
        """
        # Construct epsilon grid
        eps_grid = np.where(binary_cell == 1, eps_high, eps_low)
        
        # Compute 1/epsilon
        inv_eps = 1.0 / eps_grid
        
        # FFT of 1/epsilon
        fft_inv_eps = np.fft.fft2(inv_eps)
        # Note: The plane wave method uses Fourier coefficients of 1/epsilon.
        # The coefficient for G is (1/N^2) * sum_{r} (1/eps(r)) exp(-i G . r)
        # np.fft.fft2 computes sum_{r} f(r) exp(-i 2 pi (k r / N))
        # So fft_inv_eps / N^2 gives the coefficients.
        fft_inv_eps /= (size * size)
        
        # Get coefficients for all G_diff pairs
        # coeff_matrix[i, j] = fft_inv_eps[diff_x_flat[i,j], diff_y_flat[i,j]]
        coeff_matrix = fft_inv_eps[diff_x_flat, diff_y_flat].reshape(num_pw, num_pw)
        
        max_omega_4 = -np.inf
        min_omega_5 = np.inf
        
        for idx in range(len(unique_k)):
            # M = coeff_matrix * k_sq[idx, :]
            # M is non-Hermitian generally.
            # We need the 4th and 5th smallest positive real eigenvalues.
            M = coeff_matrix * k_sq[idx, :][:, np.newaxis]
            
            # Solve eigenvalue problem
            # Since M is non-Hermitian, use np.linalg.eigvals
            try:
                eigvals = np.linalg.eigvals(M)
            except np.linalg.LinAlgError:
                continue
            
            # Filter for real, positive eigenvalues
            real_eigvals = np.real(eigvals)
            # Sort
            real_eigvals = np.sort(real_eigvals)
            
            # We need the 4th and 5th bands.
            # Band 1 is the lowest frequency (smallest omega^2).
            # So we need indices 3 and 4 (0-indexed) of the sorted positive eigenvalues.
            # Note: Sometimes there are spurious negative eigenvalues or numerical noise.
            # We assume the first 5 positive eigenvalues correspond to bands 1-5.
            # Let's take the first 5 positive ones.
            pos_eigs = real_eigvals[real_eigvals > 1e-10]
            if len(pos_eigs) < 5:
                continue
                
            omega_4_sq = pos_eigs[3]
            omega_5_sq = pos_eigs[4]
            
            omega_4 = np.sqrt(omega_4_sq)
            omega_5 = np.sqrt(omega_5_sq)
            
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

    # --- Initialization ---
    # Start with a dielectric rod (high epsilon in center)
    # This is a common starting point for photonic crystal gaps.
    cell = np.zeros((size, size), dtype=int)
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < radius*radius:
                cell[i, j] = 1
                
    # Enforce C4 symmetry on the initial cell
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        counts = np.zeros_like(c, dtype=int)
        for i in range(size):
            for j in range(size):
                val = c[i, j]
                # 8 symmetric positions
                pos = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-j, size-1-i)
                ]
                for pi, pj in pos:
                    sym[pi, pj] += val
                    counts[pi, pj] += 1
        # Majority vote
        final = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                if counts[i, j] > 0:
                    final[i, j] = 1 if sym[i, j] > counts[i, j] / 2.0 else 0
        return final

    cell = enforce_symmetry(cell)
    best_cell = cell.copy()
    best_score = evaluate_cell(best_cell)
    
    # --- Optimization: Local Search (Hill Climbing) ---
    # We will flip pixels in C4-symmetric groups.
    # Precompute the orbits of pixels under C4.
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orbit = [(i, j)]
                # Generate orbit
                pi, pj = i, j
                while True:
                    visited[pi, pj] = True
                    # Rotate 90 deg
                    ni, nj = size - 1 - pj, pi
                    if not visited[ni, nj]:
                        orbit.append((ni, nj))
                    pi, pj = ni, nj
                    if (pi, pj) == (i, j):
                        break
                orbits.append(orbit)
    
    # Shuffle orbits for randomization
    np.random.shuffle(orbits)
    
    # Hill climbing loop
    no_improve_count = 0
    max_no_improve = 50
    step_size = 1 # Number of orbits to flip per step
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a slightly perturbed version of best
            # Or just break if we've been stuck too long
            # Let's try a random restart from best with a larger perturbation
            perturbed = best_cell.copy()
            # Flip a random set of orbits
            num_flips = np.random.randint(5, 15)
            idxs = np.random.choice(len(orbits), num_flips, replace=False)
            for idx in idxs:
                for pi, pj in orbits[idx]:
                    perturbed[pi, pj] = 1 - perturbed[pi, pj]
            perturbed = enforce_symmetry(perturbed)
            score = evaluate_cell(perturbed)
            if score > best_score:
                best_cell = perturbed
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            continue
            
        # Select a random orbit to flip
        orbit_idx = np.random.randint(len(orbits))
        orbit = orbits[orbit_idx]
        
        # Try flipping this orbit
        new_cell = best_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        # Evaluate
        score = evaluate_cell(new_cell)
        
        if score > best_score:
            best_cell = new_cell
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Simulated annealing style: accept with small probability
            # But since we don't have temperature, let's just stick to hill climbing
            # with occasional restarts as above.
            
    # Return the best cell found
    return best_cell.tolist()

