import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    lower_band = problem["lower_band"]
    
    # Time budget management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for safety
    
    # Initial design: centered rod (seed)
    # We represent the cell as a 1D array of 0/1 for efficiency
    cell = np.zeros(size * size, dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i * size + j] = 1
                
    # Ensure symmetry about the cell center (as required by the problem statement: invariant under square symmetry group about cell centre)
    # The problem states: cell == cell[::-1] row-wise and column-wise and cell == its transpose.
    # This implies 4-fold rotational symmetry and reflection symmetry across horizontal, vertical, and diagonal axes?
    # "invariant under the square symmetry group about the cell centre" usually means C4v symmetry.
    # Let's enforce C4v symmetry to reduce the search space and satisfy the constraint.
    # The center of a 32x32 grid is between pixels 15,16.
    # Indices: 0..31. Center is 15.5.
    # Symmetry operations:
    # 1. Reflection x -> 31-x
    # 2. Reflection y -> 31-y
    # 3. Rotation 90 deg: (x,y) -> (y, 31-x)
    
    def enforce_symmetry(c):
        c = c.copy()
        c = c.reshape(size, size)
        # Reflect horizontally (x -> 31-x)
        c = np.maximum(c, c[:, ::-1])
        # Reflect vertically (y -> 31-y)
        c = np.maximum(c, c[::-1, :])
        # Rotate 90 (x,y) -> (y, 31-x)
        # c_rot[i, j] = c[j, 31-i]
        c_rot = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                c_rot[i, j] = c[j, 31-i]
        c = np.maximum(c, c_rot)
        return c.flatten()
    
    cell = enforce_symmetry(cell)
    
    # Setup Plane Wave Method
    # Number of plane waves: 289 (17x17)
    N_pw = 17
    G_vals = np.arange(-N_pw//2, N_pw//2 + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G2 = Gx**2 + Gy**2
    G_vecs = np.stack([Gx.flatten(), Gy.flatten()], axis=1)
    N_G = G_vecs.shape[0]
    
    # Precompute inverse dielectric function for the initial cell
    # epsilon(x) = eps_low + (eps_high - eps_low) * cell(x)
    # 1/epsilon(x) is periodic. We need its Fourier coefficients.
    # For a pixelated dielectric, we can approximate the Fourier coefficients by summing over pixels.
    # E_n = sum_k cell_k * exp(i G_n . r_k) / (N_pixels * epsilon_k) ... wait.
    # The operator is (1/epsilon) * (-grad^2).
    # In plane wave basis, the matrix element for 1/epsilon is:
    # V_{mn} = (1/N) sum_{pixels} (1/epsilon_p) exp(-i (G_m - G_n) . r_p)
    # where r_p is the center of pixel p.
    
    # To speed up, we can precompute the basis functions exp(-i G . r_p) for all G and all pixels.
    # However, 289 G-vectors * 1024 pixels * 289 G-differences is too large to precompute fully in memory for fast updates?
    # 289 * 289 * 1024 is ~84 million complex numbers. 84M * 16 bytes = 1.3 GB. Too much.
    
    # Alternative: Compute V matrix on the fly for each candidate?
    # 289x289 matrix. 
    # V_{mn} = sum_p (1/eps_p) * exp(-i (G_m - G_n) . r_p) / N_pix
    # We can compute this using FFT if we resample the dielectric function onto a finer grid?
    # Or just do the direct sum?
    # 289*289 = 83521 elements.
    # Each element is a sum of 1024 terms.
    # 83521 * 1024 = 85 million operations.
    # In Python/NumPy, this might take ~0.1-0.5 seconds per evaluation.
    # If we do 100 evaluations, that's 10-50 seconds. Feasible.
    
    # Let's optimize the V matrix calculation.
    # We can group pixels by their dielectric value (0 or 1).
    # V = (1/N) * ( sum_{p in air} (1/eps_low) exp(...) + sum_{p in dielectric} (1/eps_high) exp(...) )
    # V = (1/N) * ( (1/eps_low) S_air + (1/eps_high) S_diel )
    # S_air + S_diel = S_total (constant vector of exps for all pixels)
    # S_diel = S_total - S_air
    # V = (1/N) * ( (1/eps_low) S_air + (1/eps_high) (S_total - S_air) )
    # V = (1/N) * ( (1/eps_high) S_total + (1/eps_low - 1/eps_high) S_air )
    
    # Precompute S_total: 289x289 matrix where V_total[m,n] = sum_p exp(-i (G_m - G_n) . r_p)
    # This depends only on the geometry (pixel positions), not the material.
    # r_p = (x_p, y_p).
    # G_m - G_n = (Gx_m - Gx_n, Gy_m - Gy_n)
    
    # Let's precompute S_total.
    # r_p for p in 0..1023.
    # x_p = (p % size + 0.5)/size - 0.5
    # y_p = (p // size + 0.5)/size - 0.5
    
    px = (np.arange(size) + 0.5) / size - 0.5
    py = (np.arange(size) + 0.5) / size - 0.5
    RX, RY = np.meshgrid(px, py, indexing='ij')
    RX = RX.flatten()
    RY = RY.flatten()
    N_pix = size * size
    
    # S_total[m, n] = sum_p exp(-i (G_m - G_n) . r_p)
    # = sum_p exp(-i (Gx_m - Gx_n) x_p) * exp(-i (Gy_m - Gy_n) y_p)
    
    # We can compute this using matrix multiplication if we separate x and y?
    # exp(-i (Gx_m - Gx_n) x_p) = exp(-i Gx_m x_p) * exp(i Gx_n x_p)
    # Let Mx[m, p] = exp(-i Gx_m x_p)
    # My[m, p] = exp(-i Gy_m y_p)
    # Then term_p = Mx[m, p] * My[m, p] * conj(Mx[n, p]) * conj(My[n, p])
    # This is not a simple matmul.
    
    # Direct calculation of S_total:
    # G_diff_x[m, n] = Gx[m] - Gx[n]
    # G_diff_y[m, n] = Gy[m] - Gy[n]
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    
    Gx_diff = Gx_flat[:, None] - Gx_flat[None, :]
    Gy_diff = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Phase_x[m, n, p] = -i * Gx_diff[m, n] * RX[p]
    # This is 289 * 289 * 1024. Memory heavy but manageable if we do it in chunks or use broadcasting carefully.
    # 289*289 = 83521. 83521 * 1024 = 85.5M floats. 85.5M * 8 bytes = 684 MB. 
    # Plus complex = 1.3 GB. Might be tight but let's try to avoid storing the full 3D array.
    
    # We can compute S_total by summing over pixels in a loop? No, too slow.
    # We can use the fact that S_total is constant.
    # Let's compute it once.
    
    # S_total = sum_p exp(-i Gx_diff * RX_p) * exp(-i Gy_diff * RY_p)
    # = sum_p exp(-i (Gx_diff * RX_p + Gy_diff * RY_p))
    
    # Let's compute S_total using a loop over pixels? 1024 iterations, each doing 289x289 vectorized ops.
    # 1024 * (289*289) = 85M ops. This should be fast in NumPy.
    
    S_total = np.zeros((N_G, N_G), dtype=np.complex128)
    for p in range(N_pix):
        phase = -1j * (Gx_diff * RX[p] + Gy_diff * RY[p])
        S_total += np.exp(phase)
        
    S_total /= N_pix
    
    # Now, for a given cell configuration, we need S_air.
    # S_air = sum_{p in air} exp(...)
    # We can maintain S_air incrementally or recompute it.
    # Recomputing S_air:
    # S_air = sum_p (1 - cell_p) * exp(...)
    # = S_total - sum_p cell_p * exp(...)
    # Let S_diel = sum_p cell_p * exp(...)
    
    # To update S_diel efficiently when flipping a pixel:
    # If we flip pixel p from 0 to 1: S_diel += exp_p
    # If we flip pixel p from 1 to 0: S_diel -= exp_p
    
    # We need the vector E_p of length N_G*N_G for each pixel p?
    # E_p[m, n] = exp(-i (G_m - G_n) . r_p)
    # This is 1024 * 289 * 289 complex numbers. 1024 * 83521 * 16 bytes = 1.3 GB.
    # Too much memory.
    
    # Alternative: Recompute S_diel from scratch for each candidate.
    # S_diel = sum_{p: cell_p=1} exp_p
    # We can do this by iterating over the 1s in the cell.
    # Number of 1s is ~20% of 1024 = 200.
    # 200 iterations of 289x289 vectorized ops.
    # 200 * 83521 = 16.7M ops. Very fast.
    
    def compute_S_diel(c):
        c = c.reshape(size, size)
        S_d = np.zeros((N_G, N_G), dtype=np.complex128)
        # Find indices of 1s
        ys, xs = np.where(c == 1)
        for k in range(len(xs)):
            x = xs[k]
            y = ys[k]
            rx = (x + 0.5) / size - 0.5
            ry = (y + 0.5) / size - 0.5
            phase = -1j * (Gx_diff * rx + Gy_diff * ry)
            S_d += np.exp(phase)
        return S_d / N_pix
    
    # Initial S_diel
    S_diel = compute_S_diel(cell)
    
    def get_V_matrix(S_d):
        # V = (1/eps_high) S_total + (1/eps_low - 1/eps_high) S_air
        # S_air = S_total - S_d
        # V = (1/eps_high) S_total + (1/eps_low - 1/eps_high) (S_total - S_d)
        # V = (1/eps_high + 1/eps_low - 1/eps_high) S_total - (1/eps_low - 1/eps_high) S_d
        # V = (1/eps_low) S_total - (1/eps_low - 1/eps_high) S_d
        # Wait, check signs.
        # V = (1/eps_low) S_air + (1/eps_high) S_d
        # S_air = S_total - S_d
        # V = (1/eps_low) (S_total - S_d) + (1/eps_high) S_d
        # V = (1/eps_low) S_total + (1/eps_high - 1/eps_low) S_d
        return (1.0/eps_low) * S_total + (1.0/eps_high - 1.0/eps_low) * S_d
    
    def compute_bands(V):
        # H = V @ G2 (diagonal)
        # In plane wave basis, the Hamiltonian is:
        # H_{mn} = V_{mn} * G_m^2 ? No.
        # The operator is -1/eps * grad^2.
        # In Fourier space, grad^2 -> -G^2.
        # So H_{mn} = (1/eps)_{mn} * G_m^2 ? 
        # Actually, the matrix element is <m| (1/eps) (-grad^2) |n>
        # = sum_k (1/eps)_{m,k} <k| (-grad^2) |n>
        # <k| (-grad^2) |n> = G_k^2 delta_{kn}
        # So H_{mn} = (1/eps)_{mn} * G_n^2
        # This is a matrix multiplication: V @ diag(G^2)
        # Or (V * G^2[None, :]) if G^2 is broadcast along columns.
        
        G2_flat = G2.flatten()
        H = V * G2_flat[None, :]
        
        # Solve eigenvalue problem
        # H is Hermitian?
        # V is Hermitian (since 1/eps is real and periodic).
        # G2 is real.
        # So H is Hermitian.
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals
    
    def get_gap_ratio(eigvals):
        # TM bands 4 and 5.
        # Indices 3 and 4 (0-indexed).
        # We need min omega_5 and max omega_4 over the BZ.
        # Since we only compute at one k-point here, we need to sample multiple k-points.
        # The problem says "30 wave vectors covering the boundary and interior".
        # We must replicate the evaluator's k-points.
        
        # Irreducible BZ for square lattice: 0 <= kx <= ky <= kx+ky? 
        # Standard irreducible zone: triangle with vertices (0,0), (pi,0), (pi,pi).
        # Or (0,0), (pi/2, pi/2), (pi,0)?
        # Let's use the standard 30 points.
        # We need to define the 30 k-points.
        # Common set: 
        # Gamma (0,0)
        # M (pi, 0)
        # X (pi/2, pi/2) -- wait, M is (pi,0) or (pi, pi)? 
        # In 2D square:
        # Gamma: (0,0)
        # X: (pi, 0) (midpoint of side)
        # M: (pi, pi) (corner)
        # High symmetry lines: Gamma-X, X-M, M-Gamma.
        
        # Let's generate 30 points.
        # 10 on Gamma-X: k = (t*pi, 0), t in [0,1]
        # 10 on X-M: k = (pi, t*pi), t in [0,1]
        # 10 on M-Gamma: k = (pi*(1-t), pi*(1-t)), t in [0,1]
        # This covers the boundary. The problem says "boundary and interior".
        # Maybe a grid?
        # "30 wave vectors covering the boundary and interior"
        # Let's assume a specific set. If I don't match it exactly, my local search might optimize for the wrong points.
        # However, the gap is a global property. Optimizing on a dense set of points is better.
        # Let's use a 5x5 grid in the irreducible zone? 25 points. Plus center?
        
        # Let's define a robust set of k-points.
        # kx in [0, pi], ky in [0, kx].
        # We will use 30 points.
        
        k_points = []
        # Boundary: Gamma-X (10 pts), X-M (10 pts), M-Gamma (10 pts)
        for i in range(10):
            t = i / 9.0
            k_points.append((t * np.pi, 0.0))
            k_points.append((np.pi, t * np.pi))
            k_points.append(((1.0 - t) * np.pi, (1.0 - t) * np.pi))
            
        # This is 30 points. It covers the boundary.
        # Does it cover the interior? No.
        # The problem says "boundary and interior".
        # Let's add some interior points? But I must use exactly 30?
        # "on 30 wave vectors"
        # I will assume the evaluator uses a specific 30 points.
        # A common choice for 30 points in a triangle is a grid.
        # Let's try to mimic a grid.
        # 5x5 grid in the full BZ is 25. 
        # Maybe the 30 points are:
        # 1. Gamma
        # 2. X
        # 3. M
        # 4-30: 27 points on the lines?
        
        # To be safe, I will use a set that is dense enough.
        # If my k-points differ from the evaluator's, I might overfit to my set.
        # But the gap is usually dominated by the boundary.
        
        min_w5 = np.inf
        max_w4 = -np.inf
        
        for kx, ky in k_points:
            # Shift G vectors by k
            # G_m + k
            Gx_k = Gx_flat + kx
            Gy_k = Gy_flat + ky
            G2_k = Gx_k**2 + Gy_k**2
            
            H = V * G2_k[None, :]
            eigvals = np.linalg.eigvalsh(H)
            
            # TM bands 4 and 5 (indices 3, 4)
            w4 = eigvals[3]
            w5 = eigvals[4]
            
            if w5 < min_w5:
                min_w5 = w5
            if w4 > max_w4:
                max_w4 = w4
                
        if min_w5 == np.inf:
            return -1.0
            
        ratio = 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)
        return ratio

    # Initial evaluation
    V_init = get_V_matrix(S_diel)
    current_score = get_gap_ratio(V_init)
    
    best_cell = cell.copy()
    best_score = current_score
    
    # Hill Climbing
    # We will try flipping pixels.
    # To respect symmetry, we must flip orbits of pixels.
    # An orbit is a set of pixels related by C4v symmetry.
    # We can precompute the orbits.
    
    orbits = []
    visited = np.zeros(size * size, dtype=bool)
    for p in range(size * size):
        if not visited[p]:
            orbit = set()
            # Generate orbit
            i, j = divmod(p, size)
            # Reflections and Rotations
            # (i, j)
            # (i, 31-j)
            # (31-i, j)
            # (31-i, 31-j)
            # (j, i) -- Rot 180? No.
            # C4v generators: R90, Rx (reflect x)
            # (x, y) -> (y, 31-x) [R90]
            # (x, y) -> (31-x, y) [Rx]
            
            def get_orbit(i, j):
                pts = set()
                x, y = i, j
                for _ in range(4):
                    pts.add((x, y))
                    pts.add((31-x, y))
                    # Rotate 90: (x, y) -> (y, 31-x)
                    x, y = y, 31-x
                return pts
            
            orbit = get_orbit(i, j)
            for (ox, oy) in orbit:
                visited[ox * size + oy] = True
            orbits.append(list(orbit))
            
    # Shuffle orbits for diversity?
    # np.random.shuffle(orbits)
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate through a subset of orbits? Or all?
        # 1024 pixels / 4-8 per orbit = ~128-256 orbits.
        # Trying all orbits each step might be slow if each eval is 0.1s.
        # 256 * 0.1s = 25s per step. Too slow.
        
        # We need faster evaluation or fewer orbits.
        # Let's pick a random subset of 20 orbits per step.
        
        # Actually, let's just try flipping one orbit at a time, but do it efficiently.
        # If we can't do 256 evals in 110s, we can't do hill climbing on all orbits.
        
        # Let's estimate eval time more precisely.
        # compute_S_diel: ~200 pixels * 83521 ops.
        # In NumPy, 83521 ops is ~0.001s? 
        # 200 * 0.001 = 0.2s? 
        # Plus eigvalsh on 289x289. 
        # np.linalg.eigvalsh(289x289) is ~0.05s.
        # 30 k-points * 0.05s = 1.5s per candidate.
        # 110s / 1.5s = ~70 candidates.
        
        # So we can only evaluate ~70 candidates.
        # We should pick the most promising flips.
        # Or just do a random search?
        
        # Let's do a random search with local refinement.
        # 1. Start with seed.
        # 2. Pick a random orbit.
        # 3. Flip it.
        # 4. Evaluate.
        # 5. If better, keep. Else, revert.
        
        # To make it more effective, we can try both adding and removing material?
        # Flipping an orbit means changing 0->1 or 1->0.
        
        while time.time() - start_time < time_limit:
            # Pick a random orbit
            orbit_idx = np.random.randint(len(orbits))
            orbit = orbits[orbit_idx]
            
            # Determine current state of the orbit (should be uniform due to symmetry)
            # Check first pixel
            p0 = orbit[0]
            current_val = cell[p0]
            
            # Flip
            for p in orbit:
                cell[p] = 1 - current_val
            
            # Evaluate
            S_diel_new = compute_S_diel(cell)
            V_new = get_V_matrix(S_diel_new)
            new_score = get_gap_ratio(V_new)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = cell.copy()
                improved = True
            else:
                # Revert
                for p in orbit:
                    cell[p] = current_val
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0
                
    return best_cell.reshape(size, size).tolist()
