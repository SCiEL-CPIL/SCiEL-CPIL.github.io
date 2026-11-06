"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_symmetric(cell):
    """Enforce square symmetry: cell == cell[::-1] row-wise and col-wise and cell == cell.T"""
    size = len(cell)
    cell = np.array(cell, dtype=int)
    # Enforce 180-degree rotation symmetry: cell[i][j] == cell[size-1-i][size-1-j]
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size-1-i][size-1-j]:
                cell[i][j] = 1 if cell[i][j] + cell[size-1-i][size-1-j] > 0 else 0
    # Enforce transpose symmetry: cell[i][j] == cell[j][i]
    for i in range(size):
        for j in range(i+1, size):
            if cell[i][j] != cell[j][i]:
                val = 1 if cell[i][j] + cell[j][i] > 0 else 0
                cell[i][j] = val
                cell[j][i] = val
    # Re-enforce 180-degree symmetry after transpose fix
    for i in range(size):
        for j in range(size):
            if cell[i][j] != cell[size-1-i][size-1-j]:
                cell[i][j] = 1 if cell[i][j] + cell[size-1-i][size-1-j] > 0 else 0
    # Final consistency check
    for i in range(size):
        for j in range(size):
            cell[i][j] = 1 if (cell[i][j] + cell[size-1-i][size-1-j] + cell[j][i] + cell[size-1-j][size-1-i]) >= 2 else 0
    return cell.tolist()

def _compute_fourier_coeffs(cell, n_g=7):
    """Compute Fourier coefficients of 1/epsilon for a 32x32 pixel cell.
    
    Returns dict mapping (mx, my) -> coefficient for mx,my in [-n_g, n_g]^2
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    
    # Build epsilon array
    eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            eps[i, j] = eps_high if cell[i][j] == 1 else eps_low
    
    # 1/epsilon
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients
    coeffs = {}
    for mx in range(-n_g, n_g+1):
        for my in range(-n_g, n_g+1):
            # Sum over all pixels
            total = 0.0
            for i in range(size):
                for j in range(size):
                    total += inv_eps[i, j] * np.exp(-2j * np.pi * (mx * i / size + my * j / size))
            coeffs[(mx, my)] = total / (size * size)
    
    return coeffs

def _estimate_gap(cell, n_g=5, n_kpts=5):
    """Estimate the 4-5 TM band gap using a simplified plane-wave method.
    
    This is a fast approximation for screening purposes.
    Returns the gap-midgap ratio.
    """
    size = len(cell)
    
    # Compute Fourier coefficients
    coeffs = _compute_fourier_coeffs(cell, n_g)
    
    # Define k-points in irreducible BZ: (0,0), (0, 0.5), (0.5, 0.5), (0.25, 0.5), (0.25, 0.25)
    k_points = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (0.25, 0.5),
        (0.25, 0.25),
    ]
    
    # For each k-point, solve the eigenvalue problem
    # H = |k+G|^2 * (1/eps) convolved with G
    # For TM: eigenvalues are omega^2
    
    all_band4_max = 0.0
    all_band5_min = float('inf')
    
    for kx, ky in k_points:
        # Build matrix
        g_indices = [(mx, my) for mx in range(-n_g, n_g+1) for my in range(-n_g, n_g+1)]
        n_gv = len(g_indices)
        
        # Build H matrix: H[G1, G2] = (k+G1)^2 * sum_G3 delta(G1-G2-G3) / eps(G3)
        # For TM polarization: H[G1, G2] = (k+G1)·(k+G1) * sum_{G3: G1-G2=G3} (1/eps)(G3)
        # Actually for TM: the operator is -∇·(1/eps ∇), which in Fourier space is:
        # H[G1, G2] = (k+G1)^2 * (1/eps)(G1-G2)
        
        H = np.zeros((n_gv, n_gv), dtype=complex)
        
        for idx1, G1 in enumerate(g_indices):
            kx1 = kx + G1[0]
            ky1 = ky + G1[1]
            ksq1 = kx1*kx1 + ky1*ky1
            for idx2, G2 in enumerate(g_indices):
                Gdiff = (G1[0]-G2[0], G1[1]-G2[1])
                if Gdiff in coeffs:
                    H[idx1, idx2] = ksq1 * coeffs[Gdiff]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            if len(eigvals) >= 5:
                band4 = eigvals[3]  # 0-indexed, so 4th band
                band5 = eigvals[4]  # 5th band
                all_band4_max = max(all_band4_max, band4)
                all_band5_min = min(all_band5_min, band5)
        except:
            continue
    
    if all_band5_min == float('inf') or all_band5_min <= 0:
        return 0.0
    
    gap = all_band5_min - all_band4_max
    midgap = (all_band5_min + all_band4_max) / 2.0
    if midgap <= 0:
        return 0.0
    return gap / midgap

def _flip_symmetric_pixel(cell, i, j, size):
    """Flip a pixel and all its symmetry-equivalent pixels."""
    cell = [row[:] for row in cell]  # Copy
    val = cell[i][j]
    
    # Symmetry group of square: identity, 90°, 180°, 270° rotations, and reflections
    # For a pixel at (i, j), the symmetry-equivalent positions are:
    # Identity: (i, j)
    # 90° rot: (j, size-1-i)
    # 180° rot: (size-1-i, size-1-j)
    # 270° rot: (size-1-j, i)
    # Reflection x: (size-1-i, j)
    # Reflection y: (i, size-1-j)
    # Reflection main diag: (j, i)
    # Reflection anti-diag: (size-1-j, size-1-i)
    
    positions = set()
    positions.add((i, j))
    positions.add((j, size-1-i))
    positions.add((size-1-i, size-1-j))
    positions.add((size-1-j, i))
    positions.add((size-1-i, j))
    positions.add((i, size-1-j))
    positions.add((j, i))
    positions.add((size-1-j, size-1-i))
    
    for pi, pj in positions:
        if 0 <= pi < size and 0 <= pj < size:
            cell[pi][pj] = 1 - val
    
    return cell

def design(problem):
    """Sieve optimization for photonic crystal unit cell design."""
    start_time = time.time()
    deadline = start_time + 180  # Leave margin for safety
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Start with the seed design: centered dielectric rod
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    
    # Ensure symmetry
    cell = _make_symmetric(cell)
    
    # Evaluate initial design
    best_score = _estimate_gap(cell)
    best_cell = [row[:] for row in cell]
    
    # Sieve optimization: try flipping pixels
    # Focus on pixels near the boundary of the dielectric region
    rng = np.random.default_rng(42)
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random pixel to flip
        i = rng.integers(0, size)
        j = rng.integers(0, size)
        
        # Flip
        new_cell = _flip_symmetric_pixel(cell, i, j, size)
        
        # Check if it's the same as before
        if new_cell == cell:
            continue
        
        # Evaluate
        new_score = _estimate_gap(new_cell)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = [row[:] for row in new_cell]
            cell = new_cell
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Occasionally accept worse solutions to escape local optima
            if no_improve_count % 50 == 0 and rng.random() < 0.1:
                cell = new_cell
        
        # Reset if too many non-improving steps
        if no_improve_count > 200:
            # Reset to best and try a different strategy
            cell = [row[:] for row in best_cell]
            no_improve_count = 0
    
    # Return the best cell found
    return _make_symmetric(best_cell)
# EVOLVE-BLOCK-END
