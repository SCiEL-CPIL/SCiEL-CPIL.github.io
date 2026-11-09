import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell to maximize the TM band gap 
    between bands 4 and 5 using a fast plane-wave surrogate and simulated annealing.
    """
    start_time = time.time()
    deadline = 100.0  # Safety margin before 120s limit
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Symmetry: D4 group. Fundamental domain is upper triangle of top-left 16x16 quadrant.
    # Indices i, j in [0, 15] with i <= j.
    half = size // 2  # 16
    # List of indices in the fundamental domain
    fund_indices = []
    for i in range(half):
        for j in range(i, half):
            fund_indices.append((i, j))
    n_vars = len(fund_indices)
    
    # Precompute pixel centers for the full 32x32 grid
    # r_p = (i + 0.5)/size - 0.5, (j + 0.5)/size - 0.5
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    XX, YY = np.meshgrid(xs, ys)
    
    # Plane wave setup for surrogate
    # Use 11x11 grid of G vectors (total 121)
    N_pw_side = 11
    G_max = N_pw_side // 2
    G_vals = np.arange(-G_max, G_max + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # Precompute exp(-i G . r) for all pixels and all G vectors
    # Shape: (N_pixels, N_pw)
    # r_p is (size*size, 2)
    r_pixels = np.column_stack((XX.ravel(), YY.ravel()))
    # G_vectors is (N_pw, 2)
    # dot product: (size*size, 1) @ (1, N_pw) -> (size*size, N_pw)
    dot_prods = r_pixels @ G_vectors.T
    exp_factors = np.exp(-1j * dot_prods)
    
    # epsilon_inv values
    eps_inv_high = 1.0 / eps_high
    eps_inv_low = 1.0 / eps_low
    
    # Map fundamental indices to full pixel indices
    # For a fundamental index (i, j) with i <= j < half:
    # It maps to 8 symmetric pixels in the 32x32 grid.
    # (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
    # and their transposes?
    # The symmetry is D4.
    # Let's list the 8 equivalent pixels for a generic (i,j) in the fundamental domain.
    # The fundamental domain is the sector 0 <= theta <= pi/4.
    # Coordinates:
    # 1. (i, j)
    # 2. (j, i)  [Transpose]
    # 3. (i, size-1-j) [Vertical Refl]
    # 4. (size-1-j, i) [Transpose of 3]
    # 5. (size-1-i, j) [Horizontal Refl]
    # 6. (j, size-1-i) [Transpose of 5]
    # 7. (size-1-i, size-1-j) [Rotation 180]
    # 8. (size-1-j, size-1-i) [Transpose of 7]
    
    # Check for overlaps when i=j or i=size-1-i (center).
    # Since size=32, half=16.
    # i, j in 0..15.
    # size-1-i is 31-i.
    # If i=15, size-1-i=16.
    # The pixels are distinct because 0..15 and 16..31 are disjoint.
    # However, if i=j, some symmetries might overlap?
    # (i,i) and (i,i) -> same.
    # (i, size-1-i) and (size-1-i, i).
    # Since i < 16 and size-1-i >= 16, (i, size-1-i) is in top-right, (size-1-i, i) in bottom-left. Distinct.
    # So each fundamental variable controls 8 distinct pixels, EXCEPT if we consider the geometry.
    # Wait, the fundamental domain size is 136. 136 * 8 = 1088 > 1024.
    # This implies overlaps.
    # Overlaps occur on the symmetry axes.
    # Axis 1: i=j. Pixels (i,i). Transpose is itself.
    # Axis 2: i=size-1-i? No, i is 0..15.
    # The "center" of the cell is between 15 and 16.
    # So no pixel lies exactly on the center axis in a way that maps to itself under 180 rot?
    # (i,j) -> (31-i, 31-j).
    # If i=15, j=15 -> (16,16). Distinct.
    # So where are the overlaps?
    # The D4 group has 8 elements. The fundamental domain of a square under D4 is 1/8th of the area.
    # Area = 32*32 = 1024. 1024/8 = 128.
    # My fundamental domain count was 136.
    # Why?
    # The fundamental domain for D4 on a grid is the region $0 \le i \le j$ AND $i+j < size/2$?
    # No, the standard fundamental domain for the square symmetry about the center is the 45-degree wedge.
    # For a $32 \times 32$ grid, the center is at $(15.5, 15.5)$.
    # The wedge is $0 \le i \le 15$ and $0 \le j \le 15$ and $i \le j$?
    # Let's check the mapping again.
    # If I define $Q[i][j]$ for $0 \le i \le j < 16$.
    # This covers the upper triangle of the top-left quadrant.
    # Size: $16 \times 17 / 2 = 136$.
    # Does this tile the square?
    # The square is divided into 4 quadrants.
    # Top-Left: $i<16, j<16$.
    # Top-Right: $i<16, j \ge 16$.
    # Bottom-Left: $i \ge 16, j<16$.
    # Bottom-Right: $i \ge 16, j \ge 16$.
    
    # Symmetry relations:
    # $C[i][j] = Q[i][j]$ for $i,j < 16$ AND $i \le j$?
    # If $i > j$ in TL quadrant, $C[i][j] = C[j][i] = Q[j][i]$.
    # So TL quadrant is defined by Q.
    # TR quadrant: $C[i][j] = C[i][31-j]$. $31-j < 16$.
    # So $C[i][j] = Q[i][31-j]$ if $i \le 31-j$?
    # This is getting complicated.
    
    # Simpler approach:
    # Just define the top-left 16x16 quadrant $Q$ (256 pixels).
    # Enforce symmetry $Q[i][j] = Q[j][i]$ (Transpose symmetry within quadrant? No, global transpose).
    # Global Transpose: $C[i][j] = C[j][i]$.
    # Global Vertical Refl: $C[i][j] = C[i][31-j]$.
    # Global Horizontal Refl: $C[i][j] = C[31-i][j]$.
    
    # If we define $C$ for $i,j < 16$ (TL Quadrant), we have 256 variables.
    # Constraints:
    # 1. $C[i][j] = C[j][i]$ for $i,j < 16$. (Symmetric matrix) -> 136 variables.
    # 2. $C[i][j] = C[i][31-j]$. Since $j<16$, $31-j \ge 16$. This links TL to TR.
    # 3. $C[i][j] = C[31-i][j]$. Links TL to BL.
    # 4. $C[i][j] = C[31-i][31-j]$. Links TL to BR.
    
    # So, if we specify the TL quadrant $C_{TL}$ (16x16), the rest is determined.
    # And $C_{TL}$ must be symmetric: $C_{TL}[i][j] = C_{TL}[j][i]$.
    # So we have 136 binary variables.
    # Each variable corresponds to a pair $(i,j)$ with $0 \le i \le j < 16$.
    # When we set this variable, we set:
    # $C[i][j]$, $C[j][i]$ in TL.
    # $C[i][31-j]$, $C[j][31-i]$ in TR.
    # $C[31-i][j]$, $C[31-j][i]$ in BL.
    # $C[31-i][31-j]$, $C[31-j][31-i]$ in BR.
    
    # Are these 8 indices always distinct?
    # $i,j \in [0, 15]$.
    # $31-i, 31-j \in [16, 31]$.
    # TL indices are in $[0,15] \times [0,15]$.
    # TR indices are in $[0,15] \times [16,31]$.
    # BL indices are in $[16,31] \times [0,15]$.
    # BR indices are in $[16,31] \times [16,31]$.
    # These 4 quadrants are disjoint.
    # Within TL: $(i,j)$ and $(j,i)$. If $i \neq j$, distinct. If $i=j$, same pixel.
    # Within TR: $(i, 31-j)$ and $(j, 31-i)$.
    # If $i \neq j$, are they distinct?
    # $i \neq j \implies 31-i \neq 31-j$.
    # So $(i, 31-j) \neq (j, 31-i)$ unless $i=j$.
    # So if $i \neq j$, we have 8 distinct pixels.
    # If $i = j$, we have 4 distinct pixels (2 in TL, 2 in TR, 2 in BL, 2 in BR... wait).
    # If $i=j$:
    # TL: $(i,i)$ (1 pixel).
    # TR: $(i, 31-i)$ and $(i, 31-i)$ (same pixel).
    # BL: $(31-i, i)$ and $(31-i, i)$ (same pixel).
    # BR: $(31-i, 31-i)$ (1 pixel).
    # Total 4 pixels.
    
    # So, for each of the 136 variables, we know exactly which pixels it controls.
    
    # Precompute the list of pixels for each variable
    var_pixels = []
    for k, (i, j) in enumerate(fund_indices):
        pixels = set()
        # TL
        pixels.add(i * size + j)
        pixels.add(j * size + i)
        # TR
        pixels.add(i * size + (31 - j))
        pixels.add(j * size + (31 - i))
        # BL
        pixels.add((31 - i) * size + j)
        pixels.add((31 - j) * size + i)
        # BR
        pixels.add((31 - i) * size + (31 - j))
        pixels.add((31 - j) * size + (31 - i))
        var_pixels.append(list(pixels))

    # Fast Surrogate Function
    def evaluate_config(config):
        """
        config: list of 136 booleans (0/1) for the fundamental variables.
        Returns the gap ratio.
        """
        # Construct epsilon_inv Fourier coefficients
        # eps_inv(G) = delta(G,0) + (1/12 - 1) * Sum_{p in Dielectric} exp(-i G . r_p)
        # We can compute Sum_{p in Dielectric} exp(-i G . r_p) efficiently.
        
        # Start with air (all 0)
        # Sum = 0
        # For each variable k that is 1 (dielectric), add the contributions of its pixels.
        
        # Precomputed exp_factors[pixel, G]
        # We need Sum_{p in D} exp_factors[p] for each G.
        
        # Let S_G = Sum_{p in D} exp(-i G . r_p)
        # Initialize S_G = 0 for all G.
        S_G = np.zeros(N_pw, dtype=complex)
        
        # Optimization: Precompute the contribution of each variable to each G.
        # var_contrib[k, G] = Sum_{p in var_pixels[k]} exp_factors[p, G]
        # This is 136 * 121 complex numbers. Small.
        # We can precompute this once.
        
        # But we are inside the loop. Let's precompute var_contrib outside.
        # var_contrib shape: (136, 121)
        
        # If config[k] is 1, add var_contrib[k] to S_G.
        
        # To make it faster, we can maintain S_G incrementally.
        # But for simplicity in the code structure, we might just sum.
        # 136 * 121 additions is ~16k ops. Very fast.
        
        # However, we can do better.
        # S_G = Sum_{k: config[k]==1} var_contrib[k]
        
        # Let's assume var_contrib is precomputed globally.
        
        for k in range(n_vars):
            if config[k]:
                S_G += var_contrib[k]
        
        # Now S_G is the sum of exp(-i G . r_p) for all dielectric pixels.
        # eps_inv(G) = 1 (for G=0, sum is N_pixels? No.)
        # Formula: eps_inv(G) = (1/A) * Integral eps_inv(r) exp(-i G r)
        # = (1/N) * Sum_{p} eps_inv_p exp(-i G r_p)
        # = (1/N) * [ Sum_{p in Air} 1 * exp(...) + Sum_{p in Diel} (1/12) * exp(...) ]
        # = (1/N) * [ Sum_{all} exp(...) + Sum_{p in Diel} (1/12 - 1) exp(...) ]
        # Sum_{all} exp(...) is N for G=0, 0 otherwise.
        # So eps_inv(G) = delta_{G,0} + (1/12 - 1) / N * S_G
        
        delta_G0 = np.zeros(N_pw)
        delta_G0[0] = 1.0 # G=0 is the first vector
        
        eps_inv_G = delta_G0 + (eps_inv_high - eps_inv_low) / (size * size) * S_G
        
        # Solve Eigenvalue Problem for a few k-points
        # We need min omega_5 and max omega_4.
        # We will use a small set of k-points.
        
        k_points = [
            (0, 0),      # Gamma
            (0.5, 0),    # X
            (0.5, 0.5),  # M
            (0.25, 0.25),
            (0.4, 0.1)
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # k vector in reciprocal units (1/a)
            k_vec = np.array([kx, ky])
            
            # Build Hamiltonian H
            # H_{alpha beta} = (k - G_alpha)^2 * delta_{alpha beta} - Sum_{gamma} eps_inv(G_alpha - G_beta) (k - G_gamma)^2 ?
            # Standard TM PWE:
            # Sum_{G'} eps_inv(G - G') (k - G')^2 c_{G'} = k^2 c_G
            # Matrix: M_{G, G'} = eps_inv(G - G') (k - G')^2
            # We need G - G'.
            
            # G_vectors is (N_pw, 2).
            # G_diff[i, j] = G_vectors[i] - G_vectors[j]
            # We need eps_inv at G_diff.
            # Our eps_inv_G is indexed by G vector index.
            # We need to map G_diff to an index.
            # Since G_vectors are on a grid, we can find the index.
            
            # Precompute G_diff indices?
            # G_vectors are sorted.
            # G_diff will be in range [-2*G_max, 2*G_max].
            # Our eps_inv_G only has G in [-G_max, G_max].
            # This is a problem. The convolution requires G up to 2*G_max.
            # So we need eps_inv for larger G.
            
            # Let's re-calculate eps_inv for a larger grid of G.
            # G_range for eps_inv: [-2*G_max, 2*G_max].
            # N_pw_side_eff = 2*N_pw_side + 1?
            # Let's just compute eps_inv for all needed G differences.
            
            # Actually, we can compute eps_inv_G for the larger grid directly from S_G.
            # S_G is defined for G in [-G_max, G_max].
            # But the sum S_G = Sum_{p in D} exp(-i G r_p) is valid for any G.
            # We just need to evaluate exp(-i G r_p) for the larger G set.
            
            # This is getting complex for the "fast" surrogate.
            # Alternative: Use a smaller N_pw_side for the surrogate, e.g., 5x5 (25 waves).
            # Then G_diff is in [-4, 4].
            # We can precompute exp_factors for G in [-10, 10] to be safe.
            
            pass # Placeholder, will implement below
            
        return 0.0 # Placeholder

    # Precompute var_contrib for N_pw=121
    # var_contrib[k, g_idx] = Sum_{p in var_pixels[k]} exp_factors[p, g_idx]
    var_contrib = np.zeros((n_vars, N_pw), dtype=complex)
    for k in range(n_vars):
        pixels = var_pixels[k]
        var_contrib[k] = np.sum(exp_factors[pixels], axis=0)

    # Redefine evaluate_config with correct logic
    def evaluate_config(config):
        S_G = np.zeros(N_pw, dtype=complex)
        for k in range(n_vars):
            if config[k]:
                S_G += var_contrib[k]
        
        # eps_inv for G in G_vectors
        delta_G0 = np.zeros(N_pw)
        delta_G0[0] = 1.0
        eps_inv_G = delta_G0 + (eps_inv_high - eps_inv_low) / (size * size) * S_G
        
        # We need eps_inv for G_diff.
        # G_diff values are differences of G_vectors.
        # We need to map G_diff to an index in a larger table or compute on the fly.
        # For speed, let's just use a smaller N_pw_side = 7 (49 waves) for the surrogate.
        # Then G_diff is in [-6, 6].
        # We can precompute a map from G vector to index for range [-10, 10].
        
        # Let's stick to N_pw_side = 7 for surrogate to be safe and fast.
        # Re-initialize with N_pw_side = 7
        N_pw_side_surr = 7
        G_max_surr = N_pw_side_surr // 2
        G_vals_surr = np.arange(-G_max_surr, G_max_surr + 1)
        Gx_surr, Gy_surr = np.meshgrid(G_vals_surr, G_vals_surr, indexing='ij')
        G_vectors_surr = np.column_stack((Gx_surr.ravel(), Gy_surr.ravel()))
        N_pw_surr = len(G_vectors_surr)
        
        # Precompute exp_factors for surrogate G's
        dot_prods_surr = r_pixels @ G_vectors_surr.T
        exp_factors_surr = np.exp(-1j * dot_prods_surr)
        
        # Precompute var_contrib for surrogate
        var_contrib_surr = np.zeros((n_vars, N_pw_surr), dtype=complex)
        for k in range(n_vars):
            pixels = var_pixels[k]
            var_contrib_surr[k] = np.sum(exp_factors_surr[pixels], axis=0)
            
        # Map G vector to index for eps_inv lookup
        # G vectors are integer coordinates.
        # Create a dictionary or use index arithmetic.
        # G_vals_surr is [-3, -2, -1, 0, 1, 2, 3]
        # Index for (gx, gy) is (gx + 3) * 7 + (gy + 3)
        
        k_points = [
            (0, 0),
            (0.5, 0),
            (0.5, 0.5),
            (0.25, 0.25),
            (0.4, 0.1)
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # k in units of 1/a
            # G in units of 1/a (since lattice constant is 1)
            
            # Build Matrix M (N_pw_surr x N_pw_surr)
            M = np.zeros((N_pw_surr, N_pw_surr), dtype=complex)
            
            # Precompute (k - G')^2 for all G'
            k_minus_G_sq = np.zeros(N_pw_surr)
            for idx, (gx, gy) in enumerate(G_vectors_surr):
                k_minus_G_sq[idx] = (kx - gx)**2 + (ky - gy)**2
            
            for i in range(N_pw_surr):
                for j in range(N_pw_surr):
                    G_diff = G_vectors_surr[i] - G_vectors_surr[j]
                    gx_diff = int(G_diff[0])
                    gy_diff = int(G_diff[1])
                    
                    # Check if G_diff is within our eps_inv range
                    if -G_max_surr <= gx_diff <= G_max_surr and -G_max_surr <= gy_diff <= G_max_surr:
                        idx_diff = (gx_diff + G_max_surr) * N_pw_side_surr + (gy_diff + G_max_surr)
                        eps_inv_val = eps_inv_G_surr[idx_diff] # Need to compute this
                    else:
                        # If out of range, eps_inv is likely small or zero for high G?
                        # For pixelated, it decays. We can approximate as 0.
                        eps_inv_val = 0.0
                        
                    M[i, j] = eps_inv_val * k_minus_G_sq[j]
            
            # Solve Eigenvalue problem
            # M is Hermitian?
            # eps_inv(G) = eps_inv(-G)^*
            # (k-G)^2 is real.
            # So M should be Hermitian.
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            
            # We need bands 4 and 5.
            # Indices 3 and 4 (0-based).
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(eigvals[3])
                omega_5 = np.sqrt(eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    # Main Search Loop
    # Initialize with seed design
    # Seed: Rod of radius 0.2
    config = [0] * n_vars
    for k, (i, j) in enumerate(fund_indices):
        # Check if pixel (i,j) is in rod
        # Center of cell is (15.5, 15.5)
        # Pixel center (i+0.5, j+0.5)
        # Distance squared
        dx = (i + 0.5) - 15.5
        dy = (j + 0.5) - 15.5
        if dx*dx + dy*dy < 0.2*32*0.2*32: # 0.2 lattice constants = 0.2 * 32 pixels?
            # Wait, radius 0.2 lattice constants.
            # Lattice constant = 32 pixels.
            # Radius = 0.2 * 32 = 6.4 pixels.
            if dx*dx + dy*dy < 6.4*6.4:
                config[k] = 1
                
    best_config = config[:]
    best_score = evaluate_config(config)
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < deadline:
        # Generate neighbor
        new_config = config[:]
        # Flip 1-3 random pixels
        n_flips = random.randint(1, 3)
        for _ in range(n_flips):
            idx = random.randint(0, n_vars - 1)
            new_config[idx] = 1 - new_config[idx]
            
        new_score = evaluate_config(new_config)
        
        # Acceptance
        if new_score > best_score:
            config = new_config
            best_config = config[:]
            best_score = new_score
        else:
            delta = new_score - best_score
            if delta > 0 or random.random() < np.exp(delta / T):
                config = new_config
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Construct final cell
    cell = [[0] * size for _ in range(size)]
    for k, (i, j) in enumerate(fund_indices):
        val = best_config[k]
        pixels = var_pixels[k]
        for p in pixels:
            pi = p // size
            pj = p % size
            cell[pi][pj] = val
            
    return cell

# Pre-compute var_contrib_surr inside evaluate_config is inefficient.
# We should move it out.

def design(problem):
    start_time = time.time()
    deadline = 100.0
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    half = size // 2
    fund_indices = []
    for i in range(half):
        for j in range(i, half):
            fund_indices.append((i, j))
    n_vars = len(fund_indices)
    
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    XX, YY = np.meshgrid(xs, ys)
    r_pixels = np.column_stack((XX.ravel(), YY.ravel()))
    
    # Surrogate Setup
    N_pw_side_surr = 7
    G_max_surr = N_pw_side_surr // 2
    G_vals_surr = np.arange(-G_max_surr, G_max_surr + 1)
    Gx_surr, Gy_surr = np.meshgrid(G_vals_surr, G_vals_surr, indexing='ij')
    G_vectors_surr = np.column_stack((Gx_surr.ravel(), Gy_surr.ravel()))
    N_pw_surr = len(G_vectors_surr)
    
    dot_prods_surr = r_pixels @ G_vectors_surr.T
    exp_factors_surr = np.exp(-1j * dot_prods_surr)
    
    var_pixels = []
    for k, (i, j) in enumerate(fund_indices):
        pixels = set()
        pixels.add(i * size + j)
        pixels.add(j * size + i)
        pixels.add(i * size + (31 - j))
        pixels.add(j * size + (31 - i))
        pixels.add((31 - i) * size + j)
        pixels.add((31 - j) * size + i)
        pixels.add((31 - i) * size + (31 - j))
        pixels.add((31 - j) * size + (31 - i))
        var_pixels.append(list(pixels))
        
    var_contrib_surr = np.zeros((n_vars, N_pw_surr), dtype=complex)
    for k in range(n_vars):
        pixels = var_pixels[k]
        var_contrib_surr[k] = np.sum(exp_factors_surr[pixels], axis=0)
        
    eps_inv_high = 1.0 / eps_high
    eps_inv_low = 1.0 / eps_low
    
    def evaluate_config(config):
        S_G = np.zeros(N_pw_surr, dtype=complex)
        for k in range(n_vars):
            if config[k]:
                S_G += var_contrib_surr[k]
                
        delta_G0 = np.zeros(N_pw_surr)
        delta_G0[0] = 1.0
        eps_inv_G = delta_G0 + (eps_inv_high - eps_inv_low) / (size * size) * S_G
        
        k_points = [
            (0, 0),
            (0.5, 0),
            (0.5, 0.5),
            (0.25, 0.25),
            (0.4, 0.1)
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            k_minus_G_sq = np.zeros(N_pw_surr)
            for idx, (gx, gy) in enumerate(G_vectors_surr):
                k_minus_G_sq[idx] = (kx - gx)**2 + (ky - gy)**2
            
            M = np.zeros((N_pw_surr, N_pw_surr), dtype=complex)
            for i in range(N_pw_surr):
                for j in range(N_pw_surr):
                    G_diff = G_vectors_surr[i] - G_vectors_surr[j]
                    gx_diff = int(G_diff[0])
                    gy_diff = int(G_diff[1])
                    if -G_max_surr <= gx_diff <= G_max_surr and -G_max_surr <= gy_diff <= G_max_surr:
                        idx_diff = (gx_diff + G_max_surr) * N_pw_side_surr + (gy_diff + G_max_surr)
                        eps_inv_val = eps_inv_G[idx_diff]
                    else:
                        eps_inv_val = 0.0
                    M[i, j] = eps_inv_val * k_minus_G_sq[j]
            
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(eigvals[3])
                omega_5 = np.sqrt(eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
                
        if min_omega_5 == float('inf'):
            return -1.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    # Initialize
    config = [0] * n_vars
    for k, (i, j) in enumerate(fund_indices):
        dx = (i + 0.5) - 15.5
        dy = (j + 0.5) - 15.5
        if dx*dx + dy*dy < 6.4*6.4:
            config[k] = 1
            
    best_config = config[:]
    best_score = evaluate_config(config)
    
    T = 0.1
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < deadline:
        new_config = config[:]
        n_flips = random.randint(1, 3)
        for _ in range(n_flips):
            idx = random.randint(0, n_vars - 1)
            new_config[idx] = 1 - new_config[idx]
            
        new_score = evaluate_config(new_config)
        
        if new_score > best_score:
            config = new_config
            best_config = config[:]
            best_score = new_score
        else:
            delta = new_score - best_score
            if delta > 0 or random.random() < np.exp(delta / T):
                config = new_config
                
        T *= cooling
        if T < T_min:
            T = T_min

    cell = [[0] * size for _ in range(size)]
    for k, (i, j) in enumerate(fund_indices):
        val = best_config[k]
        for p in var_pixels[k]:
            pi = p // size
            pj = p % size
            cell[pi][pj] = val
            
    return cell
