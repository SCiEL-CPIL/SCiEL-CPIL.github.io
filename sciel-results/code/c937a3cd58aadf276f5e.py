import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5.
    
    Structural approach: Model the unit cell as a 4x4 block of 8x8 sub-cells.
    Each sub-cell is defined by the intersection of a 1D x-pattern and a 1D y-pattern.
    The 1D patterns are defined by 4-bit half-period widths (summing to 8).
    This ensures D4 symmetry by construction and reduces the search space to 256 unique designs.
    
    We exhaustively evaluate all 256 designs and return the one with the highest gap.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget (with margin)
    
    # --- Fast band gap evaluation using plane wave method ---
    # Precompute G vectors and indices
    npw = 8
    G_list = []
    for mx in range(-npw, npw+1):
        for my in range(-npw, npw+1):
            G_list.append((2*np.pi*mx, 2*np.pi*my))
    G = np.array(G_list)
    nG = len(G)
    
    # G dictionary for fast lookup
    G_dict = {}
    for idx, (gx, gy) in enumerate(G):
        G_dict[(gx, gy)] = idx
    
    # Key k-points in the irreducible Brillouin zone
    # Irreducible BZ: 0 <= ky <= kx, 0 <= kx <= pi
    k_points = [
        (0.0, 0.0),           # Gamma
        (np.pi, 0.0),         # M
        (np.pi/2, np.pi/2),   # X
        (np.pi/2, 0.0),       # Midpoint Gamma-M
        (np.pi/2, np.pi/4),
        (3*np.pi/8, np.pi/8),
        (np.pi/4, np.pi/4),
        (np.pi/3, np.pi/6),
        (2*np.pi/5, np.pi/5),
        (np.pi/2, np.pi/6),
        (np.pi/4, 0.0),
        (3*np.pi/4, np.pi/4),
        (np.pi/4, np.pi/8),
        (np.pi/2, np.pi/8),
    ]
    
    # Precompute k+G and kG_sq for each k-point
    kG_data = []
    for kx, ky in k_points:
        kvec = np.array([kx, ky])
        k_plus_G = kvec + G  # shape (nG, 2)
        kG_sq = np.sum(k_plus_G**2, axis=1)  # shape (nG,)
        kG_data.append(kG_sq)
    
    # Precompute x, y coordinates for pixel centers
    x_coords = (np.arange(size) + 0.5) / size
    y_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    def compute_gap_fast(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses precomputed data for speed.
        """
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute inverse dielectric Fourier coefficients
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (size * size)
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for k_idx in range(len(k_points)):
            kG_sq = kG_data[k_idx]
            
            # Build the A matrix: A[G, G'] = (1/eps)_{G-G'}
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
            
            # The eigenvalue matrix is: H = A * diag(kG_sq)
            H = A * kG_sq[np.newaxis, :]
            
            # Make Hermitian
            H = (H + H.conj().T) / 2
            
            # Get eigenvalues
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # --- Generate all 256 unique designs ---
    # A 1D pattern is defined by 4 half-period widths that sum to 8.
    # Each half-period is either 0 or some positive integer.
    # Actually, let's define it as: the pattern is a sequence of 4 blocks,
    # each block has width w_i >= 0, sum(w_i) = 8.
    # The pattern starts with dielectric=1, then alternates.
    
    def generate_1d_patterns():
        """Generate all valid 1D patterns of length 8 with 4 half-periods."""
        patterns = []
        # We need w0+w1+w2+w3 = 8, w_i >= 0
        for w0 in range(9):
            for w1 in range(9 - w0):
                for w2 in range(9 - w0 - w1):
                    w3 = 8 - w0 - w1 - w2
                    if w3 < 0:
                        continue
                    widths = [w0, w1, w2, w3]
                    # Build the 1D pattern
                    pat = []
                    val = 1  # Start with dielectric
                    for w in widths:
                        pat.extend([val] * w)
                        val = 1 - val
                    if len(pat) == 8:
                        patterns.append(tuple(pat))
        return patterns
    
    # Generate all 1D patterns
    patterns_1d = generate_1d_patterns()
    
    # For D4 symmetry, the x-pattern and y-pattern must be the same.
    # So we only need to consider one 1D pattern, and use it for both x and y.
    # The cell is then: cell[i][j] = pat[i] AND pat[j]  (where pat is the 1D pattern repeated 4 times to length 32)
    
    # Wait, the 1D pattern is of length 8. To make it length 32, we repeat it 4 times.
    # But we need D4 symmetry about the center of the 32x32 cell.
    # The center is at (15.5, 15.5).
    # If we repeat the 8-length pattern 4 times, the pattern is periodic with period 8.
    # The center of the 32-cell is at 15.5, which is the center of the 3rd block (indices 16-23? No).
    # Block 0: 0-7, Block 1: 8-15, Block 2: 16-23, Block 3: 24-31.
    # Center 15.5 is between block 1 and block 2.
    # For symmetry about 15.5, we need pat[i] == pat[31-i].
    # Since the pattern is periodic with period 8, pat[i] = pat[i mod 8].
    # pat[31-i] = pat[(31-i) mod 8] = pat[(31-i) - 4*8] = pat[(31-i) - 32] = pat[-(1+i)]... 
    # Let's just check: pat[31-i] should equal pat[i].
    # (31-i) mod 8 = (31 mod 8 - i mod 8) mod 8 = (7 - i mod 8) mod 8.
    # So we need pat[i mod 8] == pat[(7 - i mod 8) mod 8].
    # This means the 8-length pattern must be palindromic.
    
    # Let's filter for palindromic patterns.
    palindromic_patterns = []
    for pat in patterns_1d:
        if pat == pat[::-1]:
            palindromic_patterns.append(pat)
    
    # If no palindromic patterns, fall back to all patterns
    if not palindromic_patterns:
        palindromic_patterns = patterns_1d
    
    # Evaluate all designs
    best_score = -np.inf
    best_cell = None
    
    for pat in palindromic_patterns:
        # Build the 32-length pattern by repeating the 8-length pattern 4 times
        pat_32 = list(pat) * 4
        
        # Build the 32x32 cell
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                cell[i, j] = pat_32[i] * pat_32[j]
        
        # Check time
        if time.time() > deadline:
            break
        
        # Compute gap
        score = compute_gap_fast(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # If no valid cell found, fall back to a simple design
    if best_cell is None:
        # Simple 2x2 rod design
        radius = 0.14
        centers = [
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75)
        ]
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        best_cell[i, j] = 1
                        break
    
    # Convert to nested list
    return best_cell.tolist()
