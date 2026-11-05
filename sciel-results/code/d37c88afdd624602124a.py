import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _make_symmetric_cell(size, base):
    """
    Enforces full C4v symmetry (square symmetry group).
    Input: base (size x size array of 0/1)
    Output: cell (size x size list of lists of 0/1)
    """
    b = np.array(base, dtype=np.int8)
    N = size
    cell = np.zeros((N, N), dtype=np.int8)
    processed = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(N):
            if processed[i, j]:
                continue
            
            # Generate the 8 symmetric positions for (i, j)
            # 1. (i, j)
            # 2. (N-1-i, j)  [Reflect across horizontal midline]
            # 3. (j, i)      [Reflect across main diagonal]
            # 4. (N-1-i, N-1-j) [Rotate 180]
            # 5. (i, N-1-j)  [Reflect across vertical midline]
            # 6. (j, N-1-i)  [Rotate 90 CW]
            # 7. (N-1-j, i)  [Rotate 90 CCW]
            # 8. (N-1-j, N-1-i) [Reflect across anti-diagonal]
            
            orbit = [
                (i, j),
                (N-1-i, j),
                (j, i),
                (N-1-i, N-1-j),
                (i, N-1-j),
                (j, N-1-i),
                (N-1-j, i),
                (N-1-j, N-1-i)
            ]
            
            # Remove duplicates (happens on symmetry axes)
            unique_orbit = list(set(orbit))
            
            # Majority vote from input base
            vals = [b[r, c] for r, c in unique_orbit]
            if sum(vals) >= len(vals) / 2.0:
                val = 1
            else:
                val = 0
            
            # Assign value to all unique positions in the orbit
            for r, c in unique_orbit:
                cell[r, c] = val
                processed[r, c] = True
                
    return cell.tolist()

def _compute_tm_bands_fast(cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=12.0, eps_low=1.0):
    """
    Fast plane-wave TM band calculation.
    n_pw=8 gives 17x17=289 plane waves, matching evaluator.
    """
    # Build reciprocal lattice vectors
    G_range = range(-n_pw, n_pw + 1)
    Gs = np.array([(mx, my) for mx in G_range for my in G_range], dtype=float)
    nG = len(Gs)
    
    # Compute inv_eps Fourier coefficients using FFT
    cell_arr = np.array(cell, dtype=float)
    eps_arr = np.where(cell_arr > 0.5, eps_high, eps_low)
    inv_eps_arr = 1.0 / eps_arr
    
    inv_eps_fft = np.fft.fft2(inv_eps_arr)
    
    # Map G indices to FFT bin indices
    G_bins = np.array([(mx % size, my % size) for mx, my in Gs], dtype=int)
    inv_eps_G = inv_eps_fft[G_bins[:, 0], G_bins[:, 1]]
    
    # Build inv_eps matrix: M[i,j] = inv_eps(G_i - G_j)
    dmx = Gs[:, 0][:, None] - Gs[:, 0][None, :]
    dmy = Gs[:, 1][:, None] - Gs[:, 1][None, :]
    
    # Vectorized index calculation for diff_bins
    # We need to map (dmx, dmy) to (dmx % size, dmy % size)
    diff_bins_row = dmx % size
    diff_bins_col = dmy % size
    
    # inv_eps_mat[i, j] = inv_eps_fft[diff_bins_row[i, j], diff_bins_col[i, j]]
    inv_eps_mat = inv_eps_fft[diff_bins_row, diff_bins_col]
    
    # K-points: Irreducible BZ of square lattice
    # Triangle: 0 <= ky <= kx <= pi
    k_points = []
    k_points.append((0.0, 0.0))
    # Line Gamma-X (kx from 0 to pi, ky=0)
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((t * np.pi, 0.0))
    # Line Gamma-M (kx=ky, from 0 to pi)
    for t in np.linspace(0.1, 1.0, n_ky):
        k_points.append((t * np.pi, t * np.pi))
    # Line X-M (kx=pi, ky from 0 to pi)
    for t in np.linspace(0.1, 1.0, n_kx):
        k_points.append((np.pi, t * np.pi))
    
    max_band4 = 0.0
    min_band5 = np.inf
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        kGs = k + Gs
        
        # Compute dot products: (k+Gi)·(k+Gj)
        # dot[i, j] = (k+Gi) dot (k+Gj)
        # Using broadcasting
        kGs_x = kGs[:, 0]
        kGs_y = kGs[:, 1]
        dot = np.outer(kGs_x, kGs_x) + np.outer(kGs_y, kGs_y)
        
        A = dot * inv_eps_mat
        # A should be Hermitian
        A = (A + A.conj().T) / 2.0
        
        # Solve eigenvalue problem
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.real(eigvals)
        # Keep only positive eigenvalues
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) >= 5:
            b4 = eigvals[3]
            b5 = eigvals[4]
            max_band4 = max(max_band4, b4)
            min_band5 = min(min_band5, b5)
    
    if min_band5 == np.inf or max_band4 == 0:
        return -1.0
    
    gap = min_band5 - max_band4
    midgap = 0.5 * (min_band5 + max_band4)
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap

def _generate_initial_candidates(size):
    """Generate a diverse set of initial candidates."""
    candidates = []
    center = size // 2
    
    def make_cell_from_blobs(blobs, bg_val):
        """blobs: list of (cx, cy, radius, val)"""
        base = np.full((size, size), bg_val, dtype=np.int8)
        for cx, cy, r, val in blobs:
            # cx, cy are offsets from center
            for i in range(size):
                for j in range(size):
                    dx = i - center - cx
                    dy = j - center - cy
                    if dx*dx + dy*dy <= r*r:
                        base[i, j] = val
        return _make_symmetric_cell(size, base)

    # Case 1: Dielectric background (1), Air holes (0)
    # 1 central hole
    for r in [2, 3, 4, 5, 6]:
        candidates.append(make_cell_from_blobs([(0, 0, r, 0)], 1))
    
    # 4 holes at (d, 0)
    for d in [4, 5, 6, 7, 8]:
        for r in [1, 2, 3]:
            blobs = [(-d, 0, r, 0), (d, 0, r, 0), (0, -d, r, 0), (0, d, r, 0)]
            candidates.append(make_cell_from_blobs(blobs, 1))
            
    # 4 holes at (d, d)
    for d in [4, 5, 6, 7]:
        for r in [1, 2, 3]:
            blobs = [(-d, -d, r, 0), (d, -d, r, 0), (-d, d, r, 0), (d, d, r, 0)]
            candidates.append(make_cell_from_blobs(blobs, 1))
            
    # Central hole + 4 corner holes
    for r1 in [2, 3, 4]:
        for d in [6, 7, 8]:
            for r2 in [1, 2]:
                blobs = [(0, 0, r1, 0), (-d, -d, r2, 0), (d, -d, r2, 0), (-d, d, r2, 0), (d, d, r2, 0)]
                candidates.append(make_cell_from_blobs(blobs, 1))

    # Case 2: Air background (0), Dielectric rods (1)
    # 1 central rod
    for r in [4, 5, 6, 7, 8]:
        candidates.append(make_cell_from_blobs([(0, 0, r, 1)], 0))
        
    # 4 rods at (d, 0)
    for d in [4, 5, 6, 7]:
        for r in [1, 2, 3]:
            blobs = [(-d, 0, r, 1), (d, 0, r, 1), (0, -d, r, 1), (0, d, r, 1)]
            candidates.append(make_cell_from_blobs(blobs, 0))
            
    # 4 rods at (d, d)
    for d in [4, 5, 6]:
        for r in [1, 2, 3]:
            blobs = [(-d, -d, r, 1), (d, -d, r, 1), (-d, d, r, 1), (d, d, r, 1)]
            candidates.append(make_cell_from_blobs(blobs, 0))

    # Crosses
    def make_cross(arm_width, length, bg_val, cross_val):
        base = np.full((size, size), bg_val, dtype=np.int8)
        for i in range(size):
            for j in range(size):
                dx = abs(i - center)
                dy = abs(j - center)
                if dx < length and dy < arm_width / 2:
                    base[i, j] = cross_val
                elif dy < length and dx < arm_width / 2:
                    base[i, j] = cross_val
        return _make_symmetric_cell(size, base)
        
    for aw in [2, 3, 4, 5]:
        for ln in [6, 8, 10, 12]:
            candidates.append(make_cross(aw, ln, 0, 1))
            candidates.append(make_cross(aw, ln, 1, 0))

    return candidates

def design(problem):
    """Time-bounded local search for TM band gap maximization."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(size)
    
    best_cell = None
    best_score = -1.0
    
    # Evaluate initial candidates
    for cell in candidates:
        if time.time() > deadline:
            break
        try:
            score = _compute_tm_bands_fast(cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=eps_high, eps_low=eps_low)
        except Exception:
            score = -1.0
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search: random mutations
    if best_cell is not None:
        current_cell = best_cell
        current_score = best_score
        rng = np.random.default_rng(42)
        
        # We will do a few rounds of local search
        for round_num in range(10):
            if time.time() > deadline:
                break
            improved = False
            for step in range(50):
                if time.time() > deadline:
                    break
                # Random mutation: flip some pixels
                mut_cell = np.array(current_cell, dtype=np.int8)
                n_flips = rng.integers(1, 4)
                for _ in range(n_flips):
                    i = rng.integers(0, size)
                    j = rng.integers(0, size)
                    mut_cell[i, j] = 1 - mut_cell[i, j]
                sym_cell = _make_symmetric_cell(size, mut_cell)
                
                try:
                    score = _compute_tm_bands_fast(sym_cell, size, n_pw=8, n_kx=5, n_ky=5, eps_high=eps_high, eps_low=eps_low)
                except Exception:
                    score = -1.0
                
                if score > current_score:
                    current_cell = sym_cell
                    current_score = score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_cell = sym_cell
            
            if not improved:
                break
    
    if best_cell is None:
        # Fallback to a simple circle
        base = np.zeros((size, size), dtype=np.int8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                if dx*dx + dy*dy < 16:
                    base[i, j] = 1
        best_cell = _make_symmetric_cell(size, base)
    
    return best_cell
# EVOLVE-BLOCK-END
