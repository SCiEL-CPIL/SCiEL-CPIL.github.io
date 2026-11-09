import time
import numpy as np
import math

_SIZE = 32
_EPS_HIGH = 12.0
_EPS_LOW = 1.0

# Plane-wave basis: 289 = 17x17 grid of G vectors
_G_RANGE = 8
_GV = [(m, n) for m in range(-_G_RANGE, _G_RANGE+1) for n in range(-_G_RANGE, _G_RANGE+1)]
_NPW = len(_GV)  # 289

# Precompute G index lookup
_G_INDEX = {}
for idx, (gm, gn) in enumerate(_GV):
    _G_INDEX[(gm, gn)] = idx

# Precompute difference indices for M matrix construction
# M[i,j] needs mu[G[i] - G[j]]
_DIFF_IDX = np.zeros((_NPW, _NPW), dtype=np.int32)
for i in range(_NPW):
    gi, gj_ = _GV[i]
    for j in range(_NPW):
        gj, gk = _GV[j]
        dm = gi - gj
        dn = gj_ - gk
        if -_G_RANGE <= dm <= _G_RANGE and -_G_RANGE <= dn <= _G_RANGE:
            _DIFF_IDX[i, j] = _G_INDEX[(dm, dn)]
        else:
            _DIFF_IDX[i, j] = -1

# K-points: 30 points in the irreducible BZ
# Irreducible BZ: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2*pi/a)
_KPOINTS = []
# High symmetry points and grid
# Triangle vertices: (0,0), (0.25, 0.25), (0, 0.5)
# Wait, standard square BZ is [-0.5, 0.5] x [-0.5, 0.5].
# Irreducible wedge: 0 <= kx <= ky, kx+ky <= 0.5.
# Vertices: (0,0), (0.25, 0.25), (0, 0.5).
# Let's sample this triangle.
for i in range(6):
    for j in range(6):
        u = i / 5.0
        v = j / 5.0
        if u + v <= 1.0:
            # Map (u,v) to triangle.
            # Basis vectors from (0,0): e1=(0.25, 0.25), e2=(0, 0.5).
            # Point = u*e1 + v*e2 ? No, that's a parallelogram.
            # Let's use barycentric or simple grid.
            # kx = u * 0.25
            # ky = u * 0.25 + v * 0.5
            # Constraint: kx <= ky -> u*0.25 <= u*0.25 + v*0.5 (True if v>=0)
            # Constraint: kx+ky <= 0.5 -> u*0.25 + u*0.25 + v*0.5 <= 0.5 -> 0.5u + 0.5v <= 0.5 -> u+v <= 1.
            kx = u * 0.25
            ky = u * 0.25 + v * 0.5
            _KPOINTS.append((kx, ky))

# Deduplicate
_seen = set()
_unique_kpts = []
for p in _KPOINTS:
    key = (round(p[0], 6), round(p[1], 6))
    if key not in _seen:
        _seen.add(key)
        _unique_kpts.append(p)
_KPOINTS = _unique_kpts[:30]

def _compute_fourier_coeffs(cell_flat):
    """Compute Fourier coefficients of 1/eps for a given cell (flat array)."""
    # cell_flat is 0/1. 1 is dielectric.
    eps = np.where(cell_flat == 1, _EPS_HIGH, _EPS_LOW)
    eps_inv = 1.0 / eps
    # Reshape to 2D for FFT
    eps_inv_2d = eps_inv.reshape(_SIZE, _SIZE)
    # FFT
    fft_coeffs = np.fft.fft2(eps_inv_2d) / (_SIZE * _SIZE)
    
    coeffs = np.zeros(_NPW, dtype=np.complex128)
    for idx, (gm, gn) in enumerate(_GV):
        # Map G to FFT index
        # FFT index k corresponds to freq k if k <= N/2, else k-N
        # We want freq gm.
        idx_m = gm % _SIZE
        idx_n = gn % _SIZE
        coeffs[idx] = fft_coeffs[idx_m, idx_n]
    return coeffs

def _tm_bands_fast(cell):
    """Compute TM band-4/5 gap ratio using vectorized NumPy solver."""
    cell_flat = np.array(cell, dtype=np.float64).flatten()
    coeffs = _compute_fourier_coeffs(cell_flat)
    
    band_4_max = -np.inf
    band_5_min = np.inf
    
    for kx, ky in _KPOINTS:
        # Compute (k+G)^2 for all G
        kg_sq = np.zeros(_NPW, dtype=np.float64)
        kg_mag = np.zeros(_NPW, dtype=np.float64)
        for i in range(_NPW):
            gm, gn = _GV[i]
            kx_tot = kx + gm
            ky_tot = ky + gn
            kg_sq[i] = kx_tot * kx_tot + ky_tot * ky_tot
            kg_mag[i] = math.sqrt(kg_sq[i])
        
        # Construct Hermitian matrix H = D^{1/2} M D^{1/2}
        # M_{ij} = coeffs[DIFF_IDX[i,j]]
        # H_{ij} = M_{ij} * kg_mag[i] * kg_mag[j]
        
        # Build M first (sparse-ish, but dense 289x289 is fine)
        # valid = _DIFF_IDX >= 0
        # M = np.zeros((_NPW, _NPW), dtype=np.complex128)
        # M[_DIFF_IDX >= 0] = coeffs[_DIFF_IDX[_DIFF_IDX >= 0]]
        
        # Direct construction of H
        # H = M * outer(kg_mag, kg_mag)
        # This is efficient.
        
        M = np.zeros((_NPW, _NPW), dtype=np.complex128)
        valid = _DIFF_IDX >= 0
        M[valid] = coeffs[_DIFF_IDX[valid]]
        
        H = M * np.outer(kg_mag, kg_mag)
        H = 0.5 * (H + H.conj().T) # Ensure Hermitian
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(H)
        
        if len(eigvals) >= 5:
            w4_sq = eigvals[3]
            w5_sq = eigvals[4]
            if w4_sq > 0 and w5_sq > 0:
                w4 = math.sqrt(w4_sq)
                w5 = math.sqrt(w5_sq)
                band_4_max = max(band_4_max, w4)
                band_5_min = min(band_5_min, w5)
    
    if band_4_max <= 0 or band_5_min <= 0:
        return -1.0
    
    gap = band_5_min - band_4_max
    midgap = (band_5_min + band_4_max) / 2.0
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / (band_5_min + band_4_max)

def _get_orbit(i, j):
    """Get the symmetry orbit of pixel (i, j) under C4v."""
    N = _SIZE
    points = [
        (i, j),
        (N-1-i, j),
        (i, N-1-j),
        (N-1-i, N-1-j),
        (j, i),
        (j, N-1-i),
        (N-1-j, i),
        (N-1-j, N-1-i)
    ]
    return list(set(points))

def _make_initial_cell():
    """Create initial cell based on 4-circle design."""
    cell = np.zeros((_SIZE, _SIZE), dtype=np.int8)
    centers = [
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25)
    ]
    r = 0.15
    for i in range(_SIZE):
        for j in range(_SIZE):
            x = (i + 0.5) / _SIZE - 0.5
            y = (j + 0.5) / _SIZE - 0.5
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < r * r:
                    cell[i, j] = 1
                    break
    return cell

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell."""
    start_time = time.time()
    deadline = start_time + 200.0 # Safety margin
    
    size = problem.get("pixels", 32)
    if size != _SIZE:
        # Fallback for other sizes (not expected)
        cell = [[0]*size for _ in range(size)]
        return cell

    # Initialize with 4-circle design
    cell = _make_initial_cell()
    best_cell = cell.copy()
    best_score = _tm_bands_fast(best_cell)
    
    # Local search
    no_improve_count = 0
    max_no_improve = 100
    step_size = 1 # We flip pixels
    
    while time.time() < deadline - 10.0:
        # Pick random pixel
        i = np.random.randint(_SIZE)
        j = np.random.randint(_SIZE)
        
        orbit = _get_orbit(i, j)
        
        # Flip
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
            
        score = _tm_bands_fast(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            no_improve_count = 0
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
            no_improve_count += 1
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            cell = best_cell.copy()
            # Random perturbation
            for _ in range(5):
                ri = np.random.randint(_SIZE)
                rj = np.random.randint(_SIZE)
                orbit = _get_orbit(ri, rj)
                for (r, c) in orbit:
                    cell[r, c] = 1 - cell[r, c]
            no_improve_count = 0

    return best_cell.tolist()
