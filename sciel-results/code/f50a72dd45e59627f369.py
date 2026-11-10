"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _make_cell(radius, thickness=0.0):
    """Create a 32x32 cell with a centered dielectric disk of given radius (in lattice units)."""
    size = 32
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                row.append(1)
            elif abs(abs(x) - radius) < thickness / 2 and abs(abs(y) - radius) < thickness / 2:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell

def _make_ring(inner_r, outer_r):
    """Create a 32x32 cell with a dielectric ring between inner_r and outer_r."""
    size = 32
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            if inner_r * inner_r <= r2 <= outer_r * outer_r:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return cell

def _estimate_band_gap(cell, eps_high=12):
    """
    Rough estimate of the band structure using a simple plane-wave expansion.
    This is a fast heuristic to guide the search - not the exact evaluator.
    We use a small number of plane waves for speed.
    """
    size = 32
    N = 9  # number of plane waves in each direction (small for speed)
    
    # Compute Fourier coefficients of 1/epsilon for TM mode
    # For TM: eigenvalue equation involves (1/epsilon) * k^2
    # We'll use a simplified approach
    
    # Get the dielectric map
    eps_map = np.array(cell, dtype=float)
    eps_map[eps_map == 0] = 1.0
    eps_map[eps_map == 1] = eps_high
    
    inv_eps = 1.0 / eps_map
    
    # Compute Fourier coefficients
    # Use a grid of G-vectors
    G_list = []
    for gx in range(-N//2, N//2 + 1):
        for gy in range(-N//2, N//2 + 1):
            G_list.append((gx, gy))
    
    N_G = len(G_list)
    
    # Fourier coefficients of 1/epsilon
    fourier = np.zeros((N_G, N_G), dtype=complex)
    for i, (gx, gy) in enumerate(G_list):
        # Compute integral of inv_eps * exp(-i G . r)
        # Approximate with pixel sum
        phase_x = np.exp(-2j * np.pi * gx * np.arange(size) / size)
        phase_y = np.exp(-2j * np.pi * gy * np.arange(size) / size)
        phase = np.outer(phase_x, phase_y)
        fourier[i, :] = np.sum(inv_eps * phase, axis=(0, 1)) / (size * size)
    
    # Now compute bands at several k-points in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= pi/a
    # In units of 2pi/a: 0 <= kx <= ky, kx + ky <= 0.5
    
    k_points = []
    # Along Gamma-X
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0))
    # Along X-M
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Along Gamma-M
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    
    band_4_max = 0.0
    band_5_min = float('inf')
    
    for kx, ky in k_points:
        # Build the matrix A where A[i,j] = delta_ij * |k+G_i|^2 * (1/eps_0) + (1/eps)_{G_i - G_j}
        # For TM: (k+G)^2 / eps(k+G)
        
        A = np.zeros((N_G, N_G), dtype=complex)
        for i, (gx_i, gy_i) in enumerate(G_list):
            kx_i = kx + gx_i
            ky_i = ky + gy_i
            k2_i = kx_i * kx_i + ky_i * ky_i
            for j, (gx_j, gy_j) in enumerate(G_list):
                # Fourier coefficient of 1/eps at (G_i - G_j)
                dgx = gx_i - gx_j
                dgy = gy_i - gy_j
                # Find index for this G-vector
                found = False
                for idx, (gx, gy) in enumerate(G_list):
                    if gx == dgx and gy == dgy:
                        A[i, j] += fourier[idx, idx] * k2_i / (2 * np.pi)
                        found = True
                        break
                if not found:
                    # Approximate: if G-vector is outside our range, use 0
                    A[i, j] += 0.0
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            # We want bands 4 and 5 (0-indexed: 3 and 4)
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > band_4_max:
                    band_4_max = w4
                if w5 < band_5_min:
                    band_5_min = w5
        except:
            continue
    
    if band_5_min == float('inf'):
        return 0.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return gap / midgap

def design(problem):
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Deadline guard: 220s limit to be safe
    deadline = 220.0
    
    # Start with the seed design
    best_cell = _make_cell(0.2)
    best_score = _estimate_band_gap(best_cell, eps_high)
    
    # Try various radii
    radii = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for r in radii:
        if _elapsed() > deadline - 30:
            break
        cell = _make_cell(r)
        score = _estimate_band_gap(cell, eps_high)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Try rings
    rings = [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.25, 0.45), (0.3, 0.5)]
    for inner, outer in rings:
        if _elapsed() > deadline - 30:
            break
        cell = _make_ring(inner, outer)
        score = _estimate_band_gap(cell, eps_high)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Try disk with some holes (inverse)
    # Air holes in dielectric background
    for hole_r in [0.1, 0.15, 0.2, 0.25]:
        if _elapsed() > deadline - 20:
            break
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            row = []
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < hole_r * hole_r:
                    row.append(0)
                else:
                    row.append(1)
            cell.append(row)
        score = _estimate_band_gap(cell, eps_high)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Try a few more complex patterns if time permits
    if _elapsed() < deadline - 60:
        # Square rod
        for half_w in [0.15, 0.2, 0.25, 0.3]:
            if _elapsed() > deadline - 40:
                break
            cell = []
            for i in range(size):
                x = (i + 0.5) / size - 0.5
                row = []
                for j in range(size):
                    y = (j + 0.5) / size - 0.5
                    if abs(x) < half_w and abs(y) < half_w:
                        row.append(1)
                    else:
                        row.append(0)
                cell.append(row)
            score = _estimate_band_gap(cell, eps_high)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    return best_cell
# EVOLVE-BLOCK-END
