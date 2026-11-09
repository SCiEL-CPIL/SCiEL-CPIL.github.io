"""Photonic crystal design for maximum TM band gap (bands 4-5)."""
import numpy as np
import time
from collections import defaultdict

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM band 4-5 gap."""
    start_time = time.time()
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # 289 plane waves => (2*N+1)^2 = 289 => N=8
    N_pw = 8
    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # shape (289, 2)
    num_pw = len(G_vectors)
    
    # Precompute |k+G|² for k in BZ (dimensionless, a=1)
    # k is in units of 2π/a, so k+G where G is integer
    # |k+G|² = (kx + Gx)² + (ky + Gy)² (in units of (2π/a)²)
    # The 2π² factor cancels in ratios
    
    # Symmetry orbits for the 32x32 cell under square group
    # Square group: identity, 90° rot, 180° rot, 270° rot, 
    #              horizontal flip, vertical flip, two diagonal flips
    # For a cell indexed [0..31, 0..31], center at (15.5, 15.5)
    
    def get_orbits(size):
        """Get symmetry orbits for square symmetry about cell center."""
        visited = set()
        orbits = []
        for i in range(size):
            for j in range(size):
                if (i, j) not in visited:
                    # Generate orbit under square group
                    # Transformations about center (15.5, 15.5):
                    # Identity: (i, j)
                    # 180° rot: (size-1-i, size-1-j)
                    # H flip: (i, size-1-j)
                    # V flip: (size-1-i, j)
                    # Diagonal: (j, i)
                    # Anti-diagonal: (size-1-j, size-1-i)
                    orbit = set()
                    orbit.add((i, j))
                    orbit.add((size-1-i, size-1-j))
                    orbit.add((i, size-1-j))
                    orbit.add((size-1-i, j))
                    orbit.add((j, i))
                    orbit.add((size-1-j, size-1-i))
                    # 90° and 270° are implied by the above for square symmetry
                    
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits
    
    orbits = get_orbits(size)
    num_orbits = len(orbits)
    
    # Map from pixel to orbit index
    pixel_to_orbit = np.zeros((size, size), dtype=int)
    for o_idx, orbit in enumerate(orbits):
        for (i, j) in orbit:
            pixel_to_orbit[i, j] = o_idx
    
    def orbit_to_cell(orbit_vals):
        """Convert orbit values to full cell."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for o_idx in range(num_orbits):
            val = orbit_vals[o_idx]
            for (i, j) in orbits[o_idx]:
                cell[i, j] = val
        return cell
    
    def compute_fourier_eps_inv(cell):
        """Compute Fourier coefficients of 1/eps from pixel cell."""
        # 1/eps: air (0) -> 1/1 = 1, dielectric (1) -> 1/12
        inv_eps = np.where(cell == 0, 1.0, 1.0/eps_high)
        
        # DFT for 289 plane waves
        # eps_inv_G = (1/N²) * Σ_{i,j} inv_eps(i,j) * exp(-i*G·(i,j))
        # where G = (Gx, Gy) in integer units, (i,j) in pixel units
        # Phase: exp(-i*2π*(Gx*i + Gy*j)/size)
        
        # Build the DFT matrix or compute directly
        # For efficiency, use the fact that this is a 2D DFT
        # but with non-uniform G values (only -8..8)
        
        # Direct computation:
        # inv_eps_G[G_idx] = (1/size²) * sum over i,j of inv_eps[i,j] * exp(-2πi(Gx*i + Gy*j)/size)
        
        inv_eps_G = np.zeros(num_pw, dtype=complex)
        for idx in range(num_pw):
            gx = G_vectors[idx, 0]
            gy = G_vectors[idx, 1]
            # Compute phase
            phase_x = np.exp(-2j * np.pi * gx * np.arange(size) / size)
            phase_y = np.exp(-2j * np.pi * gy * np.arange(size) / size)
            # 2D DFT
            # inv_eps_G[idx] = (1/size²) * sum_i sum_j inv_eps[i,j] * phase_x[i] * phase_y[j]
            temp = inv_eps * phase_x[:, np.newaxis]
            temp = temp * phase_y[np.newaxis, :]
            inv_eps_G[idx] = np.sum(temp) / (size * size)
        
        return inv_eps_G
    
    def compute_bands(inv_eps_G, kx, ky):
        """Compute TM band eigenvalues at given k-point.
        
        Returns sorted array of omega² (dimensionless).
        """
        # k in units of 2π/a, G in integer units
        # |k+G|² = (kx + Gx)² + (ky + Gy)²
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)  # shape (num_pw,)
        
        # Build Hamiltonian: H[a,b] = inv_eps_G[a-b] * kG2[b]
        # where a,b index plane waves
        # G_diff[a,b] = G_vectors[a] - G_vectors[b]
        # inv_eps_G is indexed by G_diff
        
        # Precompute G_diff indices
        # For each pair (a,b), we need inv_eps_G at G = G_vectors[a] - G_vectors[b]
        
        # Build lookup: map (gx, gy) -> index in inv_eps_G
        G_lookup = {}
        for idx in range(num_pw):
            key = (int(G_vectors[idx, 0]), int(G_vectors[idx, 1]))
            G_lookup[key] = idx
        
        H = np.zeros((num_pw, num_pw), dtype=complex)
        for a in range(num_pw):
            ga = G_vectors[a]
            for b in range(num_pw):
                gb = G_vectors[b]
                dgx = int(ga[0] - gb[0])
                dgy = int(ga[1] - gb[1])
                key = (dgx, dgy)
                if key in G_lookup:
                    inv_eps_val = inv_eps_G[G_lookup[key]]
                else:
                    # G_diff outside the DFT grid -> zero (bandlimited assumption)
                    inv_eps_val = 0.0
                H[a, b] = inv_eps_val * kG2[b]
        
        # Solve eigenvalue problem
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.real(eigenvalues)
        eigenvalues.sort()
        return eigenvalues  # omega² values
    
    # K-points for evaluation (irreducible BZ)
    # BZ: [-π, π] × [-π, π] in units of 1/a, but we use dimensionless where k in [0, 0.5]
    # Actually, let me use k in units where the BZ is [0, 0.5] × [0, 0.5] for the first BZ
    # No, standard: BZ for square lattice with a=1 is [-π, π]²
    # In dimensionless units with k* = k*a/(2π), BZ is [-0.5, 0.5]²
    # Irreducible wedge: 0 ≤ ky ≤ kx ≤ 0.5
    # Key points: Γ=(0,0), X=(0.5,0), M=(0.5,0.5)
    
    k_points = [
        (0.0, 0.0),       # Γ
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),      # Midpoint Γ-X
        (0.25, 0.25),     # Midpoint Γ-M
        (0.5, 0.25),      # Midpoint X-M
        (0.1, 0.0),
        (0.4, 0.1),
        (0.3, 0.3),
        (0.15, 0.05),
    ]
    
    def evaluate_cell(orbit_vals):
        """Evaluate the gap-midgap ratio for given orbit values."""
        cell = orbit_to_cell(orbit_vals)
        inv_eps_G = compute_fourier_eps_inv(cell)
        
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        
        for kx, ky in k_points:
            bands_sq = compute_bands(inv_eps_G, kx, ky)
            # bands_sq[3] = ω4², bands_sq[4] = ω5²
            w4_sq = bands_sq[3]
            w5_sq = bands_sq[4]
            
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        
        if max_w4_sq <= 0:
            return 0.0
        
        # Gap-midgap ratio: 2*(min_ω5 - max_ω4)/(min_ω5 + max_ω4)
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        
        gap = min_w5 - max_w4
        ratio = 2.0 * gap / midgap
        return ratio
    
    # Initialize with a reasonable starting point
    # Try a few initial configurations
    np.random.seed(42)
    
    best_score = -float('inf')
    best_vals = None
    
    # Try several initial designs
    initial_configs = []
    
    # 1. Centered circle (baseline)
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx, orbit in enumerate(orbits):
        # Check if orbit center is within radius 0.2
        # Use the representative pixel (first in orbit)
        i, j = orbit[0]
        x = (i + 0.5) / size - 0.5
        y = (j + 0.5) / size - 0.5
        if x*x + y*y < 0.2**2:
            vals[o_idx] = 1
    initial_configs.append(vals.copy())
    
    # 2. Checkerboard pattern
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx, orbit in enumerate(orbits):
        i, j = orbit[0]
        if (i + j) % 2 == 0:
            vals[o_idx] = 1
    initial_configs.append(vals.copy())
    
    # 3. Random
    for _ in range(5):
        vals = np.random.randint(0, 2, num_orbits)
        initial_configs.append(vals.copy())
    
    # Evaluate initial configs
    for config in initial_configs:
        score = evaluate_cell(config)
        if score > best_score:
            best_score = score
            best_vals = config.copy()
    
    # Local search with random restarts
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    current_vals = best_vals.copy()
    current_score = best_score
    
    while time.time() < deadline:
        # Random flips
        n_flips = np.random.randint(1, 5)
        indices = np.random.choice(num_orbits, n_flips, replace=False)
        
        new_vals = current_vals.copy()
        new_vals[indices] = 1 - new_vals[indices]
        
        new_score = evaluate_cell(new_vals)
        
        if new_score > current_score:
            current_vals = new_vals
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
        
        # Occasional random restart
        if np.random.random() < 0.03:
            current_vals = np.random.randint(0, 2, num_orbits)
            current_score = evaluate_cell(current_vals)
            
            if current_score > best_score:
                best_score = current_score
                best_vals = current_vals.copy()
        
        # Simulated annealing style: accept worse moves with some probability
        # (kept simple for now)
    
    # Convert best orbit values to cell
    best_cell = orbit_to_cell(best_vals)
    return best_cell.tolist()
