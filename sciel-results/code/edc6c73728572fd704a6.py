import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=10, pw_half=8):
    """
    Fast approximate gap calculation.
    Uses a square PW cutoff of [-pw_half, pw_half] to better match the evaluator's 289 PWs (17x17).
    """
    N = len(cell)
    # epsilon: 1 for air (0), 12 for dielectric (1)
    eps = 1.0 + 11.0 * np.array(cell, dtype=float)
    inv_eps = 1.0 / eps
    
    # FFT of inv_eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Generate PW indices
    # Square grid from -pw_half to pw_half
    Gx_vals = np.arange(-pw_half, pw_half + 1)
    Gy_vals = np.arange(-pw_half, pw_half + 1)
    Gx_grid, Gy_grid = np.meshgrid(Gx_vals, Gy_vals, indexing='ij')
    G_list = np.vstack([Gx_grid.ravel(), Gy_grid.ravel()]).T
    n_pw = len(G_list)
    
    # Precompute difference matrices
    # G_arr is (n_pw, 2)
    G_arr = G_list
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Build inv_eps matrix
    # We need inv_eps[G_i - G_j]
    # The FFT is periodic with N.
    # Index into fft_inv using (diff % N)
    idx_qx = (Gx_diff % N).astype(int)
    idx_qy = (Gy_diff % N).astype(int)
    inv_eps_mat = fft_inv[idx_qx, idx_qy]
    
    # K-points in irreducible Brillouin zone
    # The IBZ for square lattice is 0 <= kx <= ky <= 0.5 (in units of 1/a)
    # Or 0 <= ky <= kx <= 0.5
    # Let's use a grid covering the triangle (0,0)-(0.5,0)-(0.5,0.5)
    k_points = []
    # Boundary: (0,0) to (0.5,0)
    for i in range(6):
        k_points.append((i/10.0, 0.0))
    # Boundary: (0.5,0) to (0.5,0.5)
    for i in range(1, 6):
        k_points.append((0.5, i/10.0))
    # Boundary: (0.5,0.5) to (0,0) -> (0.5, 0.5) to (0,0) is not straight in k-space, it's (0.5,0.5) to (0,0) via (0.25,0.25)?
    # The IBZ is bounded by kx=0, ky=kx, ky=0.5.
    # So points: (0,0), (0.5,0), (0.5,0.5), (0,0) is wrong.
    # Vertices: Gamma(0,0), X(0.5,0), M(0.5,0.5).
    # Edges: Gamma-X, X-M, M-Gamma.
    # Gamma-X: (t, 0) for t in [0, 0.5]
    # X-M: (0.5, t) for t in [0, 0.5]
    # M-Gamma: (0.5-t, 0.5-t) for t in [0, 0.5] -> (0.5,0.5) to (0,0)
    
    k_points = []
    # Gamma to X
    for i in range(6):
        k_points.append((i/10.0, 0.0))
    # X to M
    for i in range(1, 6):
        k_points.append((0.5, i/10.0))
    # M to Gamma
    for i in range(1, 6):
        k_points.append((0.5 - i/10.0, 0.5 - i/10.0))
    
    # Add some interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.3, 0.2))
    k_points.append((0.4, 0.3))
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = (kG ** 2).sum(axis=1)
        
        M = dots[:, None] * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                # Bands are 0-indexed. 4th band is index 3, 5th is index 4.
                # Wait, "4th and 5th TM bands".
                # Band 1 is lowest. So Band 4 is index 3. Band 5 is index 4.
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _gen_rod(r):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i, j] = 1
    return cell.tolist()

def _gen_square(r):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < r and abs(y) < r:
                cell[i, j] = 1
    return cell.tolist()

def _gen_cross(arm_w, arm_l):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if (abs(x) < arm_w/2 and abs(y) < arm_l/2) or \
               (abs(y) < arm_w/2 and abs(x) < arm_l/2):
                cell[i, j] = 1
    return cell.tolist()

def _gen_ring(inner_r, outer_r):
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x*x + y*y
            if inner_r*inner_r < r2 < outer_r*outer_r:
                cell[i, j] = 1
    return cell.tolist()

def _mutate_symmetric(cell, rate=0.05):
    size = len(cell)
    c = np.array(cell, dtype=int)
    # Fundamental domain is top-left quadrant (i < 16, j < 16)
    # Symmetries:
    # (i, j) <-> (size-1-i, j) <-> (i, size-1-j) <-> (size-1-i, size-1-j)
    for i in range(size//2):
        for j in range(size//2):
            if np.random.random() < rate:
                val = 1 - c[i, j]
                c[i, j] = val
                c[size-1-i, j] = val
                c[i, size-1-j] = val
                c[size-1-i, size-1-j] = val
    return c.tolist()

def design(problem):
    start_time = time.time()
    deadline = start_time + 210.0 # Safety margin
    
    size = problem["pixels"]
    
    # 1. Generate initial candidates
    candidates = []
    
    # Rods
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_gen_rod(r))
        
    # Squares
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(_gen_square(r))
        
    # Crosses
    for w in [0.1, 0.15, 0.2]:
        for l in [0.3, 0.4, 0.5]:
            candidates.append(_gen_cross(w, l))
            
    # Rings
    for i_r in [0.1, 0.2]:
        for o_r in [0.3, 0.4, 0.5]:
            if o_r > i_r:
                candidates.append(_gen_ring(i_r, o_r))
                
    # Evaluate all
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        if time.time() > deadline - 60:
            break
        try:
            score = _compute_gap_fast(cell, n_k=10, pw_half=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # 2. Local Search
    if best_cell is not None:
        current = best_cell
        current_score = best_score
        
        # Run multiple restarts of local search if time permits
        num_restarts = 3
        for _ in range(num_restarts):
            if time.time() > deadline - 30:
                break
            
            # If not first restart, perturb the best found so far
            if _ > 0:
                current = _mutate_symmetric(best_cell, rate=0.2)
                # Re-evaluate to get a starting point
                try:
                    current_score = _compute_gap_fast(current, n_k=10, pw_half=8)
                except:
                    current_score = -1.0
            
            # Hill climbing
            for step in range(500):
                if time.time() > deadline - 15:
                    break
                
                mutated = _mutate_symmetric(current, rate=0.05)
                try:
                    score = _compute_gap_fast(mutated, n_k=10, pw_half=8)
                    if score > current_score:
                        current_score = score
                        current = mutated
                        if score > best_score:
                            best_score = score
                            best_cell = current
                except:
                    pass
                    
    if best_cell is None:
        best_cell = _gen_rod(0.25)
        
    return best_cell
