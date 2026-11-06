"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a symmetric 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Implement a fast plane-wave method (PWE) to evaluate the band structure.
    2. Use a local search (hill-climbing with simulated annealing) to optimize the dielectric distribution.
    3. Enforce square symmetry (D4) to reduce the search space.
    4. Time-budget management: stop if approaching the deadline.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for evaluation overhead
    
    # Symmetry: D4 (square symmetry)
    # The cell is invariant under:
    # - Reflection about vertical axis (x -> -x)
    # - Reflection about horizontal axis (y -> -y)
    # - Rotation by 90 degrees (x,y) -> (-y,x)
    # This means the cell is determined by its fundamental domain: x >= 0, y >= 0, x >= y
    # For a 32x32 grid, indices 0..31. Center is at (15.5, 15.5).
    # Fundamental domain: i <= 15, j <= 15, and i >= j (upper triangle of top-left quadrant)
    
    # Precompute symmetry mappings
    def apply_symmetry(cell):
        """Apply D4 symmetry to a 32x32 cell."""
        # cell is 32x32
        # Reflect horizontally: cell[i][j] == cell[i][31-j]
        # Reflect vertically: cell[i][j] == cell[31-i][j]
        # Rotate 90: cell[i][j] == cell[31-j][i]
        # So we only need to define the top-left triangle where i <= 15, j <= 15, i >= j
        # Then fill the rest.
        sym_cell = np.zeros((size, size), dtype=int)
        for i in range(size // 2):
            for j in range(size // 2):
                if i >= j:
                    val = cell[i][j]
                    # Fundamental domain point (i, j)
                    sym_cell[i][j] = val
                    sym_cell[i][size - 1 - j] = val  # Horizontal reflection
                    sym_cell[size - 1 - i][j] = val  # Vertical reflection
                    sym_cell[size - 1 - i][size - 1 - j] = val  # Both reflections
                    # Rotation 90: (i, j) -> (size-1-j, i)
                    # Wait, rotation symmetry means cell[i][j] == cell[size-1-j][i]
                    # But if we have horizontal and vertical reflection, rotation is implied?
                    # Let's check: D4 = {e, r90, r180, r270, s_h, s_v, s_d1, s_d2}
                    # With s_h and s_v, we get r180. To get full D4, we need one diagonal reflection or rotation.
                    # The problem states: "invariant under the square symmetry group about the cell centre"
                    # and "cell == cell[::-1] row-wise and column-wise and cell == its transpose"
                    # cell[::-1] row-wise means vertical reflection (flip top-bottom).
                    # cell[::-1] column-wise means horizontal reflection (flip left-right).
                    # cell == its transpose means diagonal reflection (main diagonal).
                    # These three generate D4.
                    # So we need:
                    # 1. cell[i][j] == cell[size-1-i][j] (vertical flip)
                    # 2. cell[i][j] == cell[i][size-1-j] (horizontal flip)
                    # 3. cell[i][j] == cell[j][i] (transpose)
                    # Let's verify if these three imply the rest of D4.
                    # From 1 and 2, we get cell[i][j] == cell[size-1-i][size-1-j] (180 rotation).
                    # From 3, cell[i][j] == cell[j][i].
                    # Combining 2 and 3: cell[i][j] == cell[j][i] == cell[j][size-1-i].
                    # This is rotation by 90? Let's check (i,j) -> (size-1-j, i).
                    # cell[i][j] == cell[size-1-j][i]?
                    # We know cell[i][j] == cell[j][i] (transpose).
                    # And cell[j][i] == cell[j][size-1-i] (horizontal flip of row j).
                    # So cell[i][j] == cell[j][size-1-i].
                    # This is not exactly (size-1-j, i).
                    # Let's re-read the requirement carefully.
                    # "cell == cell[::-1] row-wise" -> reverse each row. This is horizontal reflection (left-right).
                    # "and column-wise" -> reverse each column. This is vertical reflection (top-bottom).
                    # "and cell == its transpose" -> diagonal reflection.
                    # So the constraints are:
                    # A[i][j] = A[i][size-1-j]
                    # A[i][j] = A[size-1-i][j]
                    # A[i][j] = A[j][i]
                    # Let's determine the fundamental domain.
                    # From A[i][j] = A[j][i], we only need i <= j or j <= i.
                    # From A[i][j] = A[i][size-1-j], we only need j <= size//2.
                    # From A[i][j] = A[size-1-i][j], we only need i <= size//2.
                    # So fundamental domain: i in [0, size//2-1], j in [i, size//2-1] (assuming size even).
                    # For size=32, i in [0,15], j in [i,15].
                    
                    # Let's fill based on the three constraints.
                    # We will define values for i in [0, 15], j in [i, 15].
                    # Then:
                    # A[i][j] = val
                    # A[j][i] = val (transpose)
                    # A[i][size-1-j] = val (horizontal)
                    # A[size-1-i][j] = val (vertical)
                    # A[j][size-1-i] = val (transpose + horizontal, or vertical + transpose)
                    # A[size-1-j][i] = val (transpose + vertical)
                    # A[size-1-i][size-1-j] = val (all three)
                    # A[size-1-j][size-1-i] = val (transpose of above)
                    
                    sym_cell[j][i] = val
                    sym_cell[i][size - 1 - j] = val
                    sym_cell[size - 1 - i][j] = val
                    sym_cell[j][size - 1 - i] = val
                    sym_cell[size - 1 - j][i] = val
                    sym_cell[size - 1 - i][size - 1 - j] = val
                    sym_cell[size - 1 - j][size - 1 - i] = val
        return sym_cell

    # Initialize with a random symmetric pattern
    # Fundamental domain size: sum_{i=0}^{15} (16-i) = 16+15+...+1 = 136 pixels
    num_fundamental = sum(size // 2 - i for i in range(size // 2))
    
    # Start with a known good structure: a dielectric rod or hole
    # Let's try a large dielectric disk in the center.
    # Or a checkerboard-like pattern?
    # Actually, for TM gaps, large dielectric regions with small air holes often work well.
    # Let's start with a central dielectric square/diamond.
    
    def init_cell():
        cell = np.zeros((size, size), dtype=int)
        # Create a central dielectric region
        # For TM, high dielectric contrast is good.
        # Let's try a large circle of dielectric.
        center = (size - 1) / 2.0
        radius = size * 0.35  # Large radius
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                if dist < radius:
                    cell[i][j] = 1
        # Apply symmetry
        # But wait, the circle is already symmetric.
        # Let's just use the fundamental domain to ensure exact symmetry.
        sym_cell = np.zeros((size, size), dtype=int)
        for i in range(size // 2):
            for j in range(i, size // 2):
                dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                val = 1 if dist < radius else 0
                # Fill symmetric positions
                sym_cell[i][j] = val
                sym_cell[j][i] = val
                sym_cell[i][size - 1 - j] = val
                sym_cell[size - 1 - i][j] = val
                sym_cell[j][size - 1 - i] = val
                sym_cell[size - 1 - j][i] = val
                sym_cell[size - 1 - i][size - 1 - j] = val
                sym_cell[size - 1 - j][size - 1 - i] = val
        return sym_cell

    cell = init_cell()
    
    # PWE Solver
    # We need to solve the eigenvalue problem for TM modes.
    # H = (k + G) x (k + G) / eps(G)
    # Using 289 plane waves (17x17 grid of G vectors).
    # G = (n, m) where n, m in [-8, 8]
    
    G_vectors = []
    for n in range(-8, 9):
        for m in range(-8, 9):
            G_vectors.append((n, m))
    num_G = len(G_vectors)
    
    # Precompute G vectors as numpy array
    G = np.array(G_vectors, dtype=float)
    
    # Dielectric function in reciprocal space
    # eps(G) = 1 / <1/eps>(G)
    # <1/eps>(G) = (1/A) sum_i (1/eps(r_i)) exp(-i G.r_i)
    # For a pixel grid, this is a 2D FFT.
    
    def compute_inv_eps_fourier(cell):
        """Compute Fourier coefficients of 1/eps."""
        # Create eps array
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        # 2D FFT
        # The grid is size x size.
        # We need the Fourier coefficients for G = (n, m) / size?
        # The lattice constant is 1. The unit cell is 1x1.
        # Pixel (i, j) is at (i/size, j/size).
        # exp(-i 2pi (n i/size + m j/size))
        # This is exactly the 2D DFT.
        fft_inv_eps = np.fft.fft2(inv_eps)
        # The DFT gives sum_{i,j} f(i,j) exp(-i 2pi (n i/size + m j/size))
        # We need (1/A) sum ... which is (1/size^2) * fft
        # Wait, the standard DFT definition is sum f exp(-i 2pi k n / N).
        # The Fourier series coefficient c_k = (1/N^2) sum f(n) exp(-i 2pi k n / N).
        # So c_k = fft2(f) / N^2.
        c_inv_eps = fft_inv_eps / (size * size)
        
        # We need c_inv_eps for G = (n, m) where n, m in [-8, 8].
        # The FFT output is ordered [0..N/2, -N/2+1..-1].
        # We need to extract the coefficients for indices -8..8.
        # Let's create a dictionary or array.
        inv_eps_fourier = {}
        for n in range(-8, 9):
            for m in range(-8, 9):
                # Map n, m to FFT indices
                # FFT index for n: if n >= 0, n; else N + n
                idx_n = n if n >= 0 else size + n
                idx_m = m if m >= 0 else size + m
                inv_eps_fourier[(n, m)] = c_inv_eps[idx_n, idx_m]
        return inv_eps_fourier

    def solve_bands(cell, num_bands=5):
        """Solve for the first num_bands TM bands at specific k-points."""
        # k-points in irreducible BZ
        # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
        # And points along the path.
        # The problem says "30 wave vectors covering the boundary and interior".
        # We'll use a set of representative points.
        k_points = []
        # Gamma to M
        for t in np.linspace(0, 0.5, 10):
            k_points.append((t, 0))
        # M to X
        for t in np.linspace(0.5, 0, 10):
            k_points.append((0.5, t))
        # X to Gamma? No, that's not in the irreducible wedge.
        # The irreducible BZ is 1/8 of the BZ.
        # 0 <= kx <= ky <= 0.5
        # So Gamma(0,0) -> M(0.5, 0) is not correct. M is (0.5, 0).
        # The path is usually Gamma(0,0) -> M(0.5, 0) -> X(0.5, 0.5) -> Gamma(0,0).
        # But the irreducible zone is the triangle (0,0)-(0.5,0)-(0.5,0.5).
        # So we sample kx, ky such that 0 <= kx <= ky <= 0.5.
        
        k_points = []
        # Grid of k points in the irreducible BZ
        # Let's use a dense grid to be safe, but the evaluator uses 30 specific points.
        # We don't know the exact 30 points, so we'll maximize the minimum gap over a dense set.
        # Or just use the high-symmetry points and some interior points.
        
        # Let's use 30 points roughly matching a typical evaluation set.
        # 10 points on Gamma-M, 10 on M-X, 10 on X-Gamma? No, X-Gamma is not in the wedge.
        # Let's use a grid.
        # kx from 0 to 0.5, ky from kx to 0.5.
        # Let's just use a few key points for speed during optimization.
        # The final score uses 30 points. We should optimize for the worst-case gap.
        
        # For speed, we'll use a smaller set during optimization, and maybe expand at the end.
        # But let's use a reasonable set.
        k_points = []
        # Gamma
        k_points.append((0.0, 0.0))
        # M
        k_points.append((0.5, 0.0))
        # X
        k_points.append((0.5, 0.5))
        # Along Gamma-M
        for t in np.linspace(0.1, 0.4, 4):
            k_points.append((t, 0.0))
        # Along M-X
        for t in np.linspace(0.1, 0.4, 4):
            k_points.append((0.5, t))
        # Interior
        k_points.append((0.2, 0.2))
        k_points.append((0.3, 0.3))
        k_points.append((0.1, 0.3))
        k_points.append((0.2, 0.4))
        k_points.append((0.4, 0.4))
        k_points.append((0.1, 0.1))
        k_points.append((0.3, 0.4))
        k_points.append((0.4, 0.45))
        k_points.append((0.05, 0.2))
        k_points.append((0.25, 0.35))
        
        # Compute inv_eps_fourier
        inv_eps_fourier = compute_inv_eps_fourier(cell)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            # Build the matrix H
            # H_ab = (k + G_a) x (k + G_b) / eps(G_b - G_a)  ?
            # The PWE equation is:
            # sum_b [ (k + G_a) x (k + G_b) / eps(G_b - G_a) ] E_b = omega^2 E_a
            # Wait, the standard form is:
            # sum_b H_ab E_b = omega^2 E_a
            # H_ab = (k + G_a) dot (k + G_b) * (1/eps)(G_b - G_a)  (for TM, scalar)
            # For TM, the field is z-polarized. The equation is:
            # (1/eps(r)) ∇^2 E_z = -omega^2 E_z
            # In PWE:
            # sum_b [ (k + G_a) dot (k + G_b) ] (1/eps)(G_b - G_a) E_b = omega^2 E_a
            # Wait, the operator is (1/eps) * (-∇^2).
            # -∇^2 in reciprocal space is (k+G)^2.
            # So the matrix element is (k+G_a) dot (k+G_b) * <1/eps>(G_b - G_a).
            
            H = np.zeros((num_G, num_G), dtype=complex)
            for a in range(num_G):
                Ga = G[a]
                ka_Ga = k + Ga
                for b in range(num_G):
                    Gb = G[b]
                    ka_Gb = k + Gb
                    # G_b - G_a
                    dG = Gb - Ga
                    inv_eps_val = inv_eps_fourier.get(tuple(dG), 0.0)
                    H[a, b] = (ka_Ga[0] * ka_Gb[0] + ka_Ga[1] * ka_Gb[1]) * inv_eps_val
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigenvalues = np.linalg.eigvalsh(H)
            # omega^2 = eigenvalues
            # omega = sqrt(eigenvalues)
            # We want bands 4 and 5 (0-indexed: 3 and 4)
            # The bands are ordered by frequency.
            # At Gamma, some bands are degenerate. We need to be careful.
            # Usually, we take the sorted eigenvalues.
            
            # Find the 4th and 5th bands
            # We need to ensure we are counting distinct bands.
            # For now, just take indices 3 and 4.
            if len(eigenvalues) >= 5:
                omega_4 = math.sqrt(eigenvalues[3])
                omega_5 = math.sqrt(eigenvalues[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # Optimization
    best_cell = cell.copy()
    best_score = solve_bands(cell)
    
    # Local search
    # We will flip pixels in the fundamental domain.
    # Use simulated annealing or simple hill climbing.
    
    current_cell = cell.copy()
    current_score = best_score
    
    # Temperature for simulated annealing
    T = 0.01
    T_min = 0.001
    
    # Precompute fundamental domain indices
    fund_indices = []
    for i in range(size // 2):
        for j in range(i, size // 2):
            fund_indices.append((i, j))
    
    # Optimize
    while time.time() < deadline:
        # Pick a random pixel in the fundamental domain
        idx = np.random.randint(len(fund_indices))
        i, j = fund_indices[idx]
        
        # Flip the pixel
        current_cell[i][j] = 1 - current_cell[i][j]
        
        # Apply symmetry
        sym_cell = np.zeros((size, size), dtype=int)
        for ii in range(size // 2):
            for jj in range(ii, size // 2):
                val = current_cell[ii][jj]
                sym_cell[ii][jj] = val
                sym_cell[jj][ii] = val
                sym_cell[ii][size - 1 - jj] = val
                sym_cell[size - 1 - ii][jj] = val
                sym_cell[jj][size - 1 - ii] = val
                sym_cell[size - 1 - jj][ii] = val
                sym_cell[size - 1 - ii][size - 1 - jj] = val
                sym_cell[size - 1 - jj][size - 1 - ii] = val
        
        # Evaluate
        new_score = solve_bands(sym_cell)
        
        # Accept or reject
        if new_score > current_score:
            current_score = new_score
            current_cell = sym_cell.copy()
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current_score = new_score
                    current_cell = sym_cell.copy()
                else:
                    # Reject: revert
                    current_cell[i][j] = 1 - current_cell[i][j]
            else:
                # This shouldn't happen if we checked new_score > current_score
                current_score = new_score
                current_cell = sym_cell.copy()
        
        # Cool down
        T = max(T_min, T * 0.99)
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Return the best cell found
    # Convert numpy array to list of lists
    return best_cell.tolist()
# EVOLVE-BLOCK-END
