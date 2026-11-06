import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget (with margin)
    
    def quadrant_to_full(q):
        """Convert a 16x16 symmetric quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    def compute_gap_fast(cell_2d):
        """
        Fast approximate gap evaluation using fewer plane waves and k-points.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        npw = 4
        G = []
        mx_list = []
        my_list = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
                mx_list.append(mx)
                my_list.append(my)
        G = np.array(G)
        mx_arr = np.array(mx_list)
        my_arr = np.array(my_list)
        nG = len(G)
        
        inv_eps = 1.0 / eps
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vectorized Fourier Coeffs
        phase = G[:, 0][None, None, :] * X[None, :, :] + G[:, 1][None, None, :] * Y[None, :, :]
        exp_phase = np.exp(-1j * phase)
        inv_eps_coeffs = np.sum(inv_eps[None, :, :] * exp_phase, axis=(1, 2)) / (N * N)
        
        # Precompute A indices
        dmx = mx_arr[:, None] - mx_arr[None, :]
        dmy = my_arr[:, None] - my_arr[None, :]
        valid = (np.abs(dmx) <= npw) & (np.abs(dmy) <= npw)
        idx = (dmx + npw) * (2*npw + 1) + (dmy + npw)
        idx[~valid] = 0
        A = np.zeros((nG, nG), dtype=complex)
        A[valid] = inv_eps_coeffs[idx[valid]]
        
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi/2, np.pi/2),
            (np.pi/2, 0.0),
            (np.pi/4, np.pi/4)
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            H = A * kG_sq[None, :]
            H = (H + H.conj().T) / 2
            
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

    def make_rod_design(radius, cx, cy):
        """Create a 2x2 rod design in the top-left quadrant."""
        q = np.zeros((16, 16), dtype=int)
        # The quadrant is 16x16. The full cell is 32x32.
        # The rod center in the full cell is (cx, cy).
        # In the quadrant, the center is (cx*16, cy*16) ?
        # Full cell coords: 0..31. Center 0.25 -> 8.
        # Quadrant coords: 0..15. Center 0.25*32 = 8.
        # So center in quadrant is (cx*16, cy*16).
        cx_q = cx * 16
        cy_q = cy * 16
        
        for i in range(16):
            for j in range(16):
                dx = i - cx_q
                dy = j - cy_q
                if dx*dx + dy*dy < radius*radius:
                    q[i, j] = 1
        
        # Enforce symmetry q == q.T
        q = np.maximum(q, q.T)
        return q

    def local_search(quadrant, max_iter=200):
        """Flip pixels to improve score."""
        q = quadrant.copy()
        best_score = compute_gap_fast(quadrant_to_full(q))
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(16):
                for j in range(i, 16): # Only upper triangle to handle symmetry
                    # Flip q[i,j] and q[j,i]
                    old_val = q[i, j]
                    q[i, j] = 1 - old_val
                    q[j, i] = 1 - old_val
                    
                    score = compute_gap_fast(quadrant_to_full(q))
                    
                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        q[i, j] = old_val
                        q[j, i] = old_val
            if not improved:
                break
        return q, best_score

    # Initial designs to try
    initial_quadrants = []
    
    # 1. Standard 2x2 rods
    q1 = make_rod_design(3.5, 0.25, 0.25)
    initial_quadrants.append(q1)
    
    # 2. Slightly larger rods
    q2 = make_rod_design(4.0, 0.25, 0.25)
    initial_quadrants.append(q2)
    
    # 3. Smaller rods
    q3 = make_rod_design(3.0, 0.25, 0.25)
    initial_quadrants.append(q3)
    
    # 4. Shifted rods
    q4 = make_rod_design(3.5, 0.3, 0.3)
    initial_quadrants.append(q4)
    
    best_q = initial_quadrants[0]
    best_score = -np.inf
    
    for q in initial_quadrants:
        q_opt, score = local_search(q, max_iter=50)
        if score > best_score:
            best_score = score
            best_q = q_opt
            
        if time.time() > deadline - 10:
            break
            
    # Final local search with more iterations on the best
    if time.time() < deadline - 10:
        best_q, best_score = local_search(best_q, max_iter=100)
        
    best_full_cell = quadrant_to_full(best_q)
    return best_full_cell.tolist()
