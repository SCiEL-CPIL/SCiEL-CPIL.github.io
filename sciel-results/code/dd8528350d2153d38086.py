"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, quarter):
    """Build a 32x32 cell from a 16x16 top-left quarter, enforcing full square symmetry."""
    n = size // 2
    cell = []
    for i in range(size):
        ri = i if i < n else (2 * n - 1 - i)
        row = []
        for j in range(size):
            rj = j if j < n else (2 * n - 1 - j)
            row.append(quarter[ri][rj])
        cell.append(row)
    return cell

def _compute_1over_eps_fourier(cell, size, n_pw):
    """Compute Fourier coefficients of 1/epsilon for the pixel map."""
    half = n_pw // 2
    inv_eps_coeff = {}
    
    # Create 1/eps array
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            eps = 12.0 if cell[i][j] == 1 else 1.0
            inv_eps[i][j] = 1.0 / eps
    
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            # Use numpy FFT-like computation
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y, indexing='ij')
            angle = 2.0 * np.pi * (mx * (X + 0.5) / size + my * (Y + 0.5) / size)
            s = np.sum(inv_eps * (np.cos(angle) + 1j * np.sin(angle)))
            inv_eps_coeff[(mx, my)] = s / (size * size)
    
    return inv_eps_coeff

def _band_structure_tm(inv_eps_coeff, n_pw, k_points=None):
    """Compute TM band structure using plane-wave Galerkin method."""
    half = n_pw // 2
    G_vectors = []
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            G_vectors.append((2.0 * math.pi * mx, 2.0 * math.pi * my))
    
    n_pw_total = len(G_vectors)
    
    if k_points is None:
        k_points = []
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_points.append((math.pi * t, 0.0))
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_points.append((math.pi, math.pi * t))
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
    
    bands = []
    
    for kx, ky in k_points:
        M = np.zeros((n_pw_total, n_pw_total), dtype=float)
        for i in range(n_pw_total):
            kx_i = kx + G_vectors[i][0]
            ky_i = ky + G_vectors[i][1]
            for j in range(n_pw_total):
                kx_j = kx + G_vectors[j][0]
                ky_j = ky + G_vectors[j][1]
                G_diff_x = G_vectors[i][0] - G_vectors[j][0]
                G_diff_y = G_vectors[i][1] - G_vectors[j][1]
                mx = int(round(G_diff_x / (2.0 * math.pi)))
                my = int(round(G_diff_y / (2.0 * math.pi)))
                f_val = inv_eps_coeff.get((mx, my), 0.0)
                M[i][j] = (kx_i * kx_j + ky_i * ky_j) * f_val.real
        
        eigenvalues = np.linalg.eigvalsh(M)
        omegas = [math.sqrt(max(0.0, ev)) for ev in eigenvalues]
        bands.append((kx, ky, omegas))
    
    return bands

def _compute_gap_ratio(bands, lower_band=4):
    """Compute the gap-midgap ratio between bands 4 and 5."""
    max_omega_lower = 0.0
    min_omega_upper = float('inf')
    
    for kx, ky, omegas in bands:
        if len(omegas) > lower_band:
            max_omega_lower = max(max_omega_lower, omegas[lower_band - 1])
            min_omega_upper = min(min_omega_upper, omegas[lower_band])
    
    if min_omega_upper == float('inf'):
        return -1.0
    
    gap = min_omega_upper - max_omega_lower
    midgap = (min_omega_upper + max_omega_lower) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def _four_rod_pattern(size, r1, r2, r3, r4, offset):
    """Create a four-rod pattern at (±offset, ±offset) positions."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    
    centers = [
        (offset, offset),
        (-offset, offset),
        (offset, -offset),
        (-offset, -offset),
    ]
    radii = [r1, r2, r3, r4]
    
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            for c, r in zip(centers, radii):
                dx = x - c[0]
                dy = y - c[1]
                if dx * dx + dy * dy < r * r:
                    quarter[i][j] = 1
                    break
    return quarter

def _single_rod_pattern(size, radius):
    """Create a single centered rod pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                quarter[i][j] = 1
    return quarter

def _square_pattern(size, half_size):
    """Create a square pattern centered at origin."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    hs = half_size / 2.0
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if abs(x) < hs and abs(y) < hs:
                quarter[i][j] = 1
    return quarter

def _cross_pattern(size, arm_width, arm_length):
    """Create a cross (plus) pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    hw = arm_width / 2.0
    hl = arm_length / 2.0
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if (abs(x) < hw and abs(y) < hl) or (abs(y) < hw and abs(x) < hl):
                quarter[i][j] = 1
    return quarter

def _ring_pattern(size, inner_radius, outer_radius):
    """Create a ring (annulus) pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            if inner_radius * inner_radius < r2 < outer_radius * outer_radius:
                quarter[i][j] = 1
    return quarter

def _hexagon_pattern(size, radius):
    """Create a hexagon pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            # Hexagon: |x| < r/2 and |y| < r/2 * sqrt(3) and |x| + |y|/sqrt(3) < r
            if abs(x) < radius / 2 and abs(y) < radius * math.sqrt(3) / 2:
                if abs(x) + abs(y) / math.sqrt(3) < radius:
                    quarter[i][j] = 1
    return quarter

def _diamond_pattern(size, radius):
    """Create a diamond (rotated square) pattern."""
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    for i in range(n):
        x = (i + 0.5) / size - 0.5
        for j in range(n):
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < radius:
                quarter[i][j] = 1
    return quarter

def design(problem):
    """Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210.0  # 20s safety margin
    
    best_cell = None
    best_score = -float('inf')
    
    def evaluate_cell(cell):
        """Evaluate the gap ratio for a given cell."""
        try:
            if time.time() > deadline - 15.0:
                return -float('inf')
            n_pw = 17
            inv_eps_coeff = _compute_1over_eps_fourier(cell, size, n_pw)
            bands = _band_structure_tm(inv_eps_coeff, n_pw)
            score = _compute_gap_ratio(bands, lower_band=4)
            return score
        except Exception:
            return -float('inf')
    
    def try_cell(quarter):
        nonlocal best_cell, best_score
        cell = _make_symmetric_cell(size, quarter)
        score = evaluate_cell(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
        return score
    
    # Phase 1: Try different geometry families with a few representative parameters
    # Four-rod pattern (proven to work well)
    for offset in [0.20, 0.25, 0.30]:
        for radius in [0.12, 0.15, 0.18]:
            if time.time() > deadline - 60.0:
                break
            quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
            try_cell(quarter)
    
    # Square pattern
    for half_size in [0.3, 0.4, 0.5]:
        if time.time() > deadline - 60.0:
            break
        quarter = _square_pattern(size, half_size)
        try_cell(quarter)
    
    # Cross pattern
    for arm_width in [0.15, 0.2, 0.25]:
        for arm_length in [0.4, 0.5, 0.6]:
            if time.time() > deadline - 60.0:
                break
            quarter = _cross_pattern(size, arm_width, arm_length)
            try_cell(quarter)
    
    # Ring pattern
    for inner in [0.05, 0.1, 0.15]:
        for outer in [0.25, 0.3, 0.35]:
            if time.time() > deadline - 60.0:
                break
            if inner < outer:
                quarter = _ring_pattern(size, inner, outer)
                try_cell(quarter)
    
    # Hexagon pattern
    for radius in [0.2, 0.25, 0.3]:
        if time.time() > deadline - 60.0:
            break
        quarter = _hexagon_pattern(size, radius)
        try_cell(quarter)
    
    # Diamond pattern
    for radius in [0.2, 0.25, 0.3]:
        if time.time() > deadline - 60.0:
            break
        quarter = _diamond_pattern(size, radius)
        try_cell(quarter)
    
    # Phase 2: Local refinement around the best found configuration
    if best_cell is not None and time.time() < deadline - 30.0:
        # Refine around four-rod parameters (most likely to be best)
        for offset in [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28]:
            for radius in [0.13, 0.14, 0.15, 0.16, 0.17]:
                if time.time() > deadline - 20.0:
                    break
                quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
                try_cell(quarter)
    
    if best_cell is None:
        # Fallback: four-rod pattern with known good parameters
        quarter = _four_rod_pattern(size, 0.15, 0.15, 0.15, 0.15, 0.25)
        best_cell = _make_symmetric_cell(size, quarter)
    
    return best_cell

# EVOLVE-BLOCK-END
