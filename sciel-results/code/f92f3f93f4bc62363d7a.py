import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between the 4th and 5th bands.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # Precompute symmetry orbits for 32x32 grid
    # Symmetry group of the square: D4
    # We need to identify a set of representative pixels such that 
    # changing one implies changing all in its orbit.
    orbits = []
    visited = np.zeros((size, size), dtype=bool)
    
    def get_orbit(i, j):
        ops = [
            lambda x, y: (x, y),
            lambda x, y: (y, size - 1 - x),
            lambda x, y: (size - 1 - x, size - 1 - y),
            lambda x, y: (size - 1 - y, x),
            lambda x, y: (size - 1 - x, y),
            lambda x, y: (x, size - 1 - y),
            lambda x, y: (y, x),
            lambda x, y: (size - 1 - y, size - 1 - x)
        ]
        orbit = set()
        for op in ops:
            ni, nj = op(i, j)
            if 0 <= ni < size and 0 <= nj < size:
                orbit.add((ni, nj))
        return orbit

    for i in range(size):
        for j in range(size):
            if not visited[i, j]:
                orb = get_orbit(i, j)
                orbits.append(orb)
                for pi, pj in orb:
                    visited[pi, pj] = True

    # Fast Plane Wave Solver
    # Parameters:
    # M: number of G vectors in each dimension (from -M to M)
    # We use a square cut for G vectors.
    M = 5 
    # G vectors: G = (n_x, n_y) * 2*pi
    # K vectors: We evaluate at high symmetry points Gamma, X, M
    # Gamma: (0,0)
    # X: (0.5, 0)  (in units of 2*pi) -> (pi, 0)
    # M: (0.5, 0.5) (in units of 2*pi) -> (pi, pi)
    # The gap is usually determined by these points.
    # We will compute the bands at these points and take the max of band 4 and min of band 5.
    
    # Generate G vector indices
    g_indices = []
    for nx in range(-M, M + 1):
        for ny in range(-M, M + 1):
            g_indices.append((nx, ny))
    N_pw = len(g_indices)
    
    # Precompute G vector components (in units of 2*pi, so G = 2*pi * g_idx)
    # (K+G) dot (K'+G') = 4*pi^2 * (k_x + g_x) * (k_x' + g_x') ...
    # We can factor out 4*pi^2.
    
    def compute_fourier_coeffs(cell):
        """
        Computes Fourier coefficients of 1/epsilon for the given cell.
        cell is a 2D list of 0s and 1s.
        1 -> eps_high, 0 -> eps_low.
        """
        # Convert to array
        c = np.array(cell, dtype=float)
        eps = np.where(c == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients for 1/epsilon
        # f_G = (1/N^2) * sum_{x,y} f(x,y) * exp(-i G . r)
        # r = (x/N, y/N) * L? No, coordinate system:
        # Pixel (i,j) is at x = (i+0.5)/size, y = (j+0.5)/size.
        # G = (n_x, n_y) * 2*pi.
        
        coeffs = np.zeros(N_pw, dtype=complex)
        # For speed, we can use FFT if the grid matches, but size 32 is small.
        # Direct sum is O(N^2 * N_pw) = 1024 * 121 ~ 124k ops. Very fast.
        
        # Precompute exponentials? 
        # exp(-i 2*pi (n_x * x + n_y * y))
        # x = (i+0.5)/size, y = (j+0.5)/size
        
        # We can vectorize this.
        # Create meshgrid of x, y
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Flatten inv_eps
        ie_flat = inv_eps.flatten()
        
        for idx, (nx, ny) in enumerate(g_indices):
            # G = (nx, ny) * 2*pi
            # Phase = -2*pi * (nx * X + ny * Y)
            phase = -2.0 * np.pi * (nx * X + ny * Y)
            exp_term = np.exp(1j * phase)
            coeffs[idx] = np.sum(ie_flat * exp_term.flatten()) / (size * size)
            
        return coeffs

    def solve_bands(cell, k_point):
        """
        Solves for TM bands at a given K point.
        k_point is (kx, ky) in units of 2*pi (i.e., Gamma is (0,0), X is (0.5,0)).
        """
        inv_eps_coeffs = compute_fourier_coeffs(cell)
        
        # Build the matrix A
        # A_{G, G'} = (K+G) dot (K+G') * inv_eps_{G-G'}
        # Note: inv_eps_{G-G'} is the Fourier coefficient of 1/epsilon at vector G-G'.
        # Our coeffs array is indexed by g_indices.
        # We need a lookup table for inv_eps_{G-G'}.
        
        # Map (nx, ny) to index
        g_map = {}
        for idx, (nx, ny) in enumerate(g_indices):
            g_map[(nx, ny)] = idx
            
        # Precompute K+G vectors
        # K = (kx, ky) * 2*pi. We work in units of 2*pi for simplicity, 
        # so K_eff = (kx, ky).
        # (K+G) = (kx + nx, ky + ny)
        
        kg_vectors = []
        for nx, ny in g_indices:
            kg_vectors.append((kx + nx, ky + ny))
            
        A = np.zeros((N_pw, N_pw), dtype=complex)
        
        for i, (nx_i, ny_i) in enumerate(g_indices):
            kx_i, ky_i = kg_vectors[i]
            for j, (nx_j, ny_j) in enumerate(g_indices):
                kx_j, ky_j = kg_vectors[j]
                
                # G - G' = (nx_i - nx_j, ny_i - ny_j)
                diff_g = (nx_i - nx_j, ny_i - ny_j)
                
                # Check if diff_g is in our G set
                if diff_g in g_map:
                    coeff_idx = g_map[diff_g]
                    v_coeff = inv_eps_coeffs[coeff_idx]
                else:
                    # If the difference is outside our truncated set, we ignore it.
                    # This is an approximation.
                    v_coeff = 0.0
                
                # Dot product
                dot = (kx_i * kx_j + ky_i * ky_j)
                
                A[i, j] = dot * v_coeff
        
        # Solve eigenvalue problem
        # A is Hermitian?
        # inv_eps is real and symmetric, so inv_eps_{-G} = conj(inv_eps_G).
        # Since inv_eps is real, inv_eps_{-G} = inv_eps_G.
        # (K+G) dot (K+G') is real.
        # So A is real and symmetric? 
        # A_{i,j} = (K+G_i) dot (K+G_j) * V_{G_i - G_j}
        # A_{j,i} = (K+G_j) dot (K+G_i) * V_{G_j - G_i}
        # V_{G_j - G_i} = V_{-(G_i - G_j)} = V_{G_i - G_j} (since real and even)
        # So A is symmetric real.
        
        # Use np.linalg.eigh for Hermitian/Symmetric
        try:
            eigenvalues, _ = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            return np.zeros(10)
            
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)
        return eigenvalues

    def evaluate_gap(cell):
        """
        Evaluates the gap between band 4 and 5 at high symmetry points.
        Returns the gap ratio.
        """
        # Points: Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
        # Note: Due to symmetry, X (0.5, 0) and (0, 0.5) are equivalent.
        # M (0.5, 0.5) is unique.
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
        
        for kx, ky in k_points:
            eigs = solve_bands(cell, kx, ky)
            if len(eigs) >= 5:
                w4 = eigs[3]  # 4th band (index 3)
                w5 = eigs[4]  # 5th band (index 4)
                
                # We need to be careful: the "4th band" globally is the 4th lowest eigenvalue 
                # across the whole BZ. But usually, we track the bands continuously.
                # For a quick surrogate, taking the max of the 4th eigenvalue at high-symmetry points 
                # and min of the 5th is a reasonable approximation for the gap location.
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        mid = (min_omega_5 + max_omega_4) / 2.0
        
        if mid <= 0:
            return -1.0
            
        ratio = 2.0 * gap / mid
        return ratio

    # --- Initialization ---
    
    def create_circle(radius):
        cell = [[0]*size for _ in range(size)]
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = (i - cx) / size
                dy = (j - cy) / size
                if dx*dx + dy*dy < radius*radius:
                    cell[i][j] = 1
        return cell

    def create_square(half_width):
        cell = [[0]*size for _ in range(size)]
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i - cx) / size
                dy = abs(j - cy) / size
                if dx < half_width and dy < half_width:
                    cell[i][j] = 1
        return cell

    def create_superellipse(n, scale):
        # |x|^(2/n) + |y|^(2/n) < 1
        cell = [[0]*size for _ in range(size)]
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i - cx) / size
                dy = abs(j - cy) / size
                if dx == 0 and dy == 0:
                    val = 0
                else:
                    val = (dx/scale)**n + (dy/scale)**n
                if val < 1.0:
                    cell[i][j] = 1
        return cell

    # Generate candidates
    candidates = []
    
    # Circles
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(create_circle(r))
        
    # Squares
    for hw in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(create_square(hw))
        
    # Superellipses
    for n in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for s in [0.15, 0.2, 0.25, 0.3]:
             candidates.append(create_superellipse(n, s))
             
    # Crosses (Dielectric along axes)
    def create_cross(width):
        cell = [[0]*size for _ in range(size)]
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dx = abs(i - cx) / size
                dy = abs(j - cy) / size
                if (dx < width and dy < 0.4) or (dx < 0.4 and dy < width):
                    cell[i][j] = 1
        return cell
    for w in [0.05, 0.1, 0.15]:
        candidates.append(create_cross(w))
        
    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
            
    if best_cell is None:
        # Fallback to seed
        best_cell = create_circle(0.2)
        
    # --- Local Search ---
    
    # Sort orbits by potential impact? 
    # Just iterate.
    # To speed up, we can pre-compute the Fourier coeffs update.
    # But recomputing is fast enough.
    
    # We will perform multiple passes.
    max_passes = 5
    improved = True
    current_cell = best_cell
    current_score = best_score
    
    pass_idx = 0
    while improved and pass_idx < max_passes:
        improved = False
        pass_idx += 1
        
        # Shuffle orbits to avoid bias?
        # For now, fixed order.
        
        for orb in orbits:
            if time.time() > deadline - 5.0:
                break
                
            # Try flipping this orbit
            # Save original state
            original_vals = []
            for (i, j) in orb:
                original_vals.append((i, j, current_cell[i][j]))
                
            # Flip
            for (i, j) in orb:
                current_cell[i][j] = 1 - current_cell[i][j]
                
            new_score = evaluate_gap(current_cell)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                for (i, j, v) in original_vals:
                    current_cell[i][j] = v
        
        if time.time() > deadline - 5.0:
            break

    return current_cell

