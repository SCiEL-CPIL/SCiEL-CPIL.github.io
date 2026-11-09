import numpy as np
import time
import math

_DEADLINE = 225.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

# ---- Fast plane-wave TM surrogate (7x7 = 49 modes) ----
def _build_pw_evaluator(size=32, K=3):
    """Precompute everything needed for fast TM band evaluation."""
    G = [(h, k) for h in range(-K, K + 1) for k in range(-K, K + 1)]
    N = len(G)
    Gx = np.array([g[0] for g in G], dtype=np.float64)
    Gy = np.array([g[1] for g in G], dtype=np.float64)
    
    # Precompute G differences for FFT indexing
    Gdiff_h = np.zeros((N, N), dtype=int)
    Gdiff_k = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            Gdiff_h[i, j] = (G[i][0] - G[j][0]) % size
            Gdiff_k[i, j] = (G[i][1] - G[j][1]) % size
    
    # K-points in irreducible BZ (subset for speed)
    k_points = []
    k_points.append((0.0, 0.0))
    for t in np.linspace(0.1, 0.5, 5):
        k_points.append((0.0, t))
    for t in np.linspace(0.1, 0.5, 5):
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    kx_arr = np.array([p[0] for p in k_points])
    ky_arr = np.array([p[1] for p in k_points])
    nk = len(k_points)
    
    def evaluate(cell_grid):
        """Compute gap-midgap ratio for bands 4 and 5."""
        eps_grid = np.where(cell_grid == 1, 12.0, 1.0)
        inv_eps = 1.0 / eps_grid
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[Gdiff_h, Gdiff_k] / (size * size)
        coeffs_real = coeffs.real
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for idx in range(nk):
            kx = kx_arr[idx]
            ky = ky_arr[idx]
            Kx = Gx + kx
            Ky = Gy + ky
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs_real
            # Symmetrize to ensure Hermitian
            H = 0.5 * (H + H.T)
            eigenvalues = np.linalg.eigvalsh(H)
            if len(eigenvalues) >= 5:
                w4 = np.sqrt(max(0.0, eigenvalues[3]))
                w5 = np.sqrt(max(0.0, eigenvalues[4]))
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5
        
        if min_w5 == float('inf') or max_w4 <= 0 or min_w5 <= 0:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / (min_w5 + max_w4)
    
    return evaluate, k_points, nk

# ---- C4v symmetry enforcement ----
def _enforce_c4v(cell, size=32):
    """Enforce C4v symmetry: cell == cell.T, cell == cell[::-1,:], cell == cell[:,::-1]."""
    c = np.asarray(cell, dtype=np.float64)
    sym = c.copy()
    sym = sym + sym.T
    sym = sym + sym[::-1, :]
    sym = sym + sym[:, ::-1]
    sym = sym + sym.T[::-1, :]
    sym = sym + sym.T[:, ::-1]
    sym = sym + sym[::-1, :].T
    sym = sym + sym[:, ::-1].T
    return (sym >= 4).astype(int)

# ---- Get unique C4v orbits for pixel-level optimization ----
def _get_orbits(size=32):
    """Get all unique C4v orbits. Each orbit is a list of (i,j) positions."""
    visited = set()
    orbits = []
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = []
            # Generate all 8 symmetric points
            pts = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, j),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            unique_pts = []
            for p in pts:
                if p not in unique_pts:
                    unique_pts.append(p)
            orbits.append(unique_pts)
            for p in unique_pts:
                visited.add(p)
    return orbits

# ---- Chebyshev parameterized mask (smooth initialization) ----
def _make_cell_chebyshev(p, size=32):
    a0, a1, a2, a3, a4, a5, r0, thr = p
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size - 0.5
    y = (yy + 0.5) / size - 0.5
    r = np.sqrt(x * x + y * y)
    f = (a0 + a1 * (2 * r * r - 1.0) + a2 * (4 * x * y) + 
         a3 * (4 * (x * x - y * y)) + a4 * (8 * x * x * y * y) + 
         a5 * (8 * (x * x * x * y + x * y * y * y)))
    f = f + (1.0 - r * r / max(r0 * r0, 1e-9))
    cell = (f > thr).astype(int)
    return _enforce_c4v(cell, size)

# ---- Main design function ----
def design(problem):
    size = problem["pixels"]  # 32
    
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin
    
    # Build fast evaluator
    evaluate, k_points, nk = _build_pw_evaluator(size, K=3)
    
    # ---- Phase 1: Chebyshev parameter search (smooth landscape) ----
    _BOUNDS = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 
               (-2.0, 2.0), (-2.0, 2.0), (0.15, 0.55), (-2.0, 2.0)]
    
    best_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0])
    best_cell = _make_cell_chebyshev(best_p, size)
    best_score = evaluate(best_cell)
    
    # Coarse-to-fine hill climbing on Chebyshev parameters
    for step in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        if time.time() > deadline - 60.0:
            break
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 60.0:
                break
            for d in range(len(best_p)):
                if time.time() > deadline - 60.0:
                    break
                lo, hi = _BOUNDS[d]
                for sgn in (1.0, -1.0):
                    if time.time() > deadline - 60.0:
                        break
                    cand = best_p.copy()
                    cand[d] += sgn * step
                    if cand[d] < lo or cand[d] > hi:
                        continue
                    c = _make_cell_chebyshev(cand, size)
                    s = evaluate(c)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_p = cand
                        best_cell = c
                        improved = True
    
    # ---- Phase 2: Pixel-level optimization on C4v orbits ----
    # Get orbits
    orbits = _get_orbits(size)
    n_orbits = len(orbits)
    
    # Precompute orbit indices for fast flip
    orbit_indices = []
    for orbit in orbits:
        idxs = [(i, j) for i, j in orbit]
        orbit_indices.append(idxs)
    
    current_cell = best_cell.copy()
    current_score = best_score
    
    # Simulated annealing on pixel orbits
    temperature = 0.3
    cooling_rate = 0.998
    min_temp = 0.02
    
    # Precompute which orbits are "active" (have both 0s and 1s nearby)
    # For efficiency, we'll just sample randomly
    
    while time.time() < deadline - 5.0:
        # Pick a random orbit
        oi = np.random.randint(n_orbits)
        orbit = orbit_indices[oi]
        
        # Create new cell by flipping this orbit
        new_cell = current_cell.copy()
        for (i, j) in orbit:
            new_cell[i, j] = 1 - new_cell[i, j]
        
        new_score = evaluate(new_cell)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0 or (temperature > min_temp and np.random.random() < math.exp(delta / temperature)):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell.copy()
        
        # Cool down
        temperature = max(temperature * cooling_rate, min_temp)
    
    # ---- Finalize: enforce symmetry and return ----
    final_cell = _enforce_c4v(best_cell, size)
    return final_cell.tolist()
