"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
import scipy.linalg as sla

# EVOLVE-BLOCK-START

def _make_symmetric_cell(size, quarter):
    """Build a 32x32 cell from a 16x16 top-left quarter, enforcing full square symmetry."""
    n = size // 2  # 16
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
    """Compute Fourier coefficients of 1/epsilon for the pixel map.
    
    For TM polarization, the eigenvalue problem requires 1/eps.
    eps = 1 (air, pixel 0) or 12 (dielectric, pixel 1)
    1/eps = 1 (air) or 1/12 (dielectric)
    """
    half = n_pw // 2
    inv_eps_coeff = {}
    for mx in range(-half, half + 1):
        for my in range(-half, half + 1):
            s = 0.0
            for i in range(size):
                x = (i + 0.5) / size
                for j in range(size):
                    y = (j + 0.5) / size
                    eps = 12.0 if cell[i][j] == 1 else 1.0
                    inv_eps = 1.0 / eps
                    angle = 2.0 * math.pi * (mx * x + my * y)
                    s += inv_eps * (math.cos(angle) + 1j * math.sin(angle))
            inv_eps_coeff[(mx, my)] = s / (size * size)
    return inv_eps_coeff


def _band_structure_tm(inv_eps_coeff, n_pw, k_points=None):
    """Compute TM band structure using plane-wave Galerkin method.
    
    Returns list of (kx, ky, [omega_1, ..., omega_n]) for each k-point.
    """
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
        M = np.zeros((n_pw_total, n_pw_total), dtype=np.float64)
        for i in range(n_pw_total):
            kx_i = kx + G_vectors[i][0]
            ky_i = ky + G_vectors[i][1]
            for j in range(i, n_pw_total):
                kx_j = kx + G_vectors[j][0]
                ky_j = ky + G_vectors[j][1]
                G_diff_x = G_vectors[i][0] - G_vectors[j][0]
                G_diff_y = G_vectors[i][1] - G_vectors[j][1]
                mx = int(round(G_diff_x / (2.0 * math.pi)))
                my = int(round(G_diff_y / (2.0 * math.pi)))
                f_val = inv_eps_coeff.get((mx, my), 0.0)
                val = (kx_i * kx_j + ky_i * ky_j) * f_val.real
                M[i, j] = val
                if i != j:
                    M[j, i] = val
        eigenvalues = np.linalg.eigvalsh(M)
        omegas = np.sqrt(np.maximum(0.0, eigenvalues))
        bands.append((kx, ky, omegas))
    
    return bands


def _compute_gap_ratio(bands, lower_band=4):
    """Compute the gap-midgap ratio between bands 4 and 5 (0-indexed: 3 and 4)."""
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


def _fourier_harmonic_cell(size, a20, a02, a22, phase20=0.0, phase02=0.0, phase22=0.0, threshold=0.5):
    """Create a cell from superposition of low-order Fourier harmonics.
    
    The dielectric distribution is:
    eps(x,y) = eps_avg + a20*cos(2*pi*2*x + phase20) + a02*cos(2*pi*2*y + phase02) 
                + a22*cos(2*pi*(2*x+2*y) + phase22)
    
    We threshold at 'threshold' to get 0/1 pixels.
    """
    n = size // 2
    quarter = [[0] * n for _ in range(n)]
    
    for i in range(n):
        x = (i + 0.5) / size
        for j in range(n):
            y = (j + 0.5) / size
            val = (a20 * math.cos(2.0 * math.pi * 2 * x + phase20) +
                   a02 * math.cos(2.0 * math.pi * 2 * y + phase02) +
                   a22 * math.cos(2.0 * math.pi * (2 * x + 2 * y) + phase22))
            # Threshold: if val > threshold, set to 1 (dielectric)
            if val > threshold:
                quarter[i][j] = 1
            else:
                quarter[i][j] = 0
    
    return quarter


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


def _evaluate_cell_fast(cell, size, n_pw=17, n_kpoints=30):
    """Fast evaluation of gap ratio for a given cell."""
    try:
        inv_eps_coeff = _compute_1over_eps_fourier(cell, size, n_pw)
        bands = _band_structure_tm(inv_eps_coeff, n_pw, n_kpoints=n_kpoints)
        score = _compute_gap_ratio(bands, lower_band=4)
        return score
    except Exception:
        return -float('inf')


def _local_optimize_quarter(quarter, size, n_pw=17, max_iter=50, step_size=0.05):
    """Local optimization of quarter cell pixels to maximize gap ratio.
    
    Uses a simple hill-climbing approach: try flipping each pixel and keep
    the flip if it improves the score.
    """
    n = size // 2
    best_quarter = [row[:] for row in quarter]
    best_score = -float('inf')
    
    # Evaluate initial
    cell = _make_symmetric_cell(size, best_quarter)
    best_score = _evaluate_cell_fast(cell, size, n_pw)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(n):
                # Try flipping pixel (i, j)
                old_val = best_quarter[i][j]
                best_quarter[i][j] = 1 - old_val
                cell = _make_symmetric_cell(size, best_quarter)
                score = _evaluate_cell_fast(cell, size, n_pw)
                
                if score > best_score:
                    best_score = score
                    improved = True
                else:
                    best_quarter[i][j] = old_val
        
        if not improved:
            break
    
    return best_quarter, best_score


def design(problem):
    """Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210.0  # Leave 20s margin for the 230s limit
    
    best_cell = None
    best_score = -float('inf')
    
    def try_cell(quarter, label=""):
        nonlocal best_cell, best_score
        cell = _make_symmetric_cell(size, quarter)
        score = _evaluate_cell_fast(cell, size)
        if score > best_score:
            best_score = score
            best_cell = cell
        return score
    
    # Strategy 1: Fourier harmonic parameterization
    # Try different combinations of (a20, a02, a22, phase20, phase02, phase22, threshold)
    harmonic_configs = [
        # (a20, a02, a22, phase20, phase02, phase22, threshold)
        (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        (0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0),
        (0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, math.pi/2, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.0, math.pi/2, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.0, 0.0, math.pi/2, 0.0),
        (0.4, 0.4, 0.4, 0.0, 0.0, 0.0, -0.1),
        (0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.1),
        (0.6, 0.6, 0.0, 0.0, 0.0, 0.0, -0.1),
        (0.6, 0.0, 0.6, 0.0, 0.0, 0.0, -0.1),
        (0.0, 0.6, 0.6, 0.0, 0.0, 0.0, -0.1),
        (0.5, 0.5, 0.5, math.pi/4, math.pi/4, 0.0, 0.0),
        (0.5, 0.5, 0.5, -math.pi/4, -math.pi/4, 0.0, 0.0),
    ]
    
    for config in harmonic_configs:
        if time.time() > deadline - 30.0:
            break
        a20, a02, a22, ph20, ph02, ph22, thresh = config
        quarter = _fourier_harmonic_cell(size, a20, a02, a22, ph20, ph02, ph22, thresh)
        try_cell(quarter, "harmonic")
    
    # Strategy 2: Four-rod patterns (from current best approach)
    rod_configs = [
        (0.25, 0.15),
        (0.25, 0.12),
        (0.25, 0.18),
        (0.25, 0.20),
        (0.20, 0.15),
        (0.30, 0.15),
        (0.25, 0.10),
        (0.22, 0.15),
        (0.28, 0.15),
        (0.25, 0.22),
        (0.25, 0.25),
    ]
    
    for offset, radius in rod_configs:
        if time.time() > deadline - 30.0:
            break
        quarter = _four_rod_pattern(size, radius, radius, radius, radius, offset)
        try_cell(quarter, "four_rod")
    
    # Strategy 3: Single rod patterns
    for radius in [0.15, 0.18, 0.20, 0.22, 0.25]:
        if time.time() > deadline - 30.0:
            break
        quarter = _single_rod_pattern(size, radius)
        try_cell(quarter, "single_rod")
    
    # Strategy 4: Local optimization on best cell found so far
    if best_cell is not None and time.time() < deadline - 40.0:
        # Extract quarter from best_cell
        n = size // 2
        best_quarter = [[best_cell[i][j] for j in range(n)] for i in range(n)]
        
        optimized_quarter, opt_score = _local_optimize_quarter(best_quarter, size, max_iter=30)
        if opt_score > best_score:
            best_score = opt_score
            best_cell = _make_symmetric_cell(size, optimized_quarter)
    
    # Strategy 5: If time permits, try local optimization on a few other promising cells
    if time.time() < deadline - 40.0:
        # Try local optimization on a harmonic-based cell
        quarter_h = _fourier_harmonic_cell(size, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
        cell_h = _make_symmetric_cell(size, quarter_h)
        score_h = _evaluate_cell_fast(cell_h, size)
        
        if score_h > -float('inf') and score_h > best_score - 0.5:
            n = size // 2
            opt_q, opt_s = _local_optimize_quarter(quarter_h, size, max_iter=20)
            if opt_s > best_score:
                best_score = opt_s
                best_cell = _make_symmetric_cell(size, opt_q)
    
    if best_cell is None:
        # Fallback: original single rod
        best_cell = _make_symmetric_cell(size, _single_rod_pattern(size, 0.2))
    
    return best_cell

# EVOLVE-BLOCK-END
