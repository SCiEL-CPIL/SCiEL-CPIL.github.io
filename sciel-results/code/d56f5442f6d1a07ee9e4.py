"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 218.0  # safety margin before 230s kill
    
    def time_left():
        return DEADLINE - (time.time() - t_start)
    
    # Precompute pixel coordinates
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    XX, YY = np.meshgrid(coords, coords, indexing='ij')
    
    def enforce_symmetry(cell):
        """Enforce D4 symmetry by averaging over the symmetry group."""
        sym_cells = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1]
        ]
        avg = np.mean(sym_cells, axis=0)
        return (avg >= 0.5).astype(int)
    
    def calculate_gap(cell, n_pw=7):
        """
        Calculate the band 4-5 gap ratio using the plane-wave method.
        Uses a subset of G-vectors for speed.
        """
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps = 1.0 / eps
        
        N = size
        half = n_pw // 2
        
        # G-vectors
        G_list = []
        for gx in range(-half, half + 1):
            for gy in range(-half, half + 1):
                G_list.append((gx, gy))
        
        nG = len(G_list)
        G_idx = {(gx, gy): i for i, (gx, gy) in enumerate(G_list)}
        
        # Compute full FFT of inv_eps once
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        # Build inv_eps_F dictionary for needed G differences
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        # Key k-points in the irreducible BZ
        k_points_list = [
            (0.0, 0.0),      # Gamma
            (0.1, 0.0),
            (0.2, 0.0),
            (0.3, 0.0),
            (0.4, 0.0),
            (0.5, 0.0),      # X
            (0.5, 0.1),
            (0.5, 0.2),
            (0.5, 0.3),
            (0.5, 0.4),
            (0.5, 0.5),      # M
            (0.3, 0.1),
            (0.4, 0.2),
            (0.3, 0.2),
            (0.2, 0.1),
            (0.4, 0.3),
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points_list:
            # Build eigenvalue matrix
            M = np.zeros((nG, nG), dtype=complex)
            
            for ig, (gx, gy) in enumerate(G_list):
                Gkx = gx - kx
                Gky = gy - ky
                for igp, (gxp, gyp) in enumerate(G_list):
                    Gpkx = gxp - kx
                    Gpky = gyp - ky
                    dGx = gx - gxp
                    dGy = gy - gyp
                    if (dGx, dGy) in inv_eps_F:
                        M[ig, igp] = (Gkx * Gpkx + Gky * Gpky) * inv_eps_F[(dGx, dGy)]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 5:
                omega_4 = np.sqrt(max(0, eigvals[3]))
                omega_5 = np.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return -1.0
        
        gap_ratio = 2.0 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio
    
    def make_cell_from_radii(r1, r2, r3, r4, r5, shape_type=0):
        """
        Create a cell from concentric symmetric patterns.
        shape_type: 0=circle, 1=square, 2=cross, 3=diamond, 4=ring
        r1-r5: radii for 5 concentric layers
        """
        cell = np.zeros((size, size), dtype=int)
        
        dist = np.sqrt(XX**2 + YY**2)
        
        if shape_type == 0:  # Circles
            for r in [r1, r2, r3, r4, r5]:
                cell += (dist <= r).astype(int)
        elif shape_type == 1:  # Squares
            max_coord = np.maximum(np.abs(XX), np.abs(YY))
            for r in [r1, r2, r3, r4, r5]:
                cell += (max_coord <= r).astype(int)
        elif shape_type == 2:  # Crosses
            min_coord = np.minimum(np.abs(XX), np.abs(YY))
            for r in [r1, r2, r3, r4, r5]:
                cell += (min_coord <= r).astype(int)
        elif shape_type == 3:  # Diamonds (L1 norm)
            l1_dist = np.abs(XX) + np.abs(YY)
            for r in [r1, r2, r3, r4, r5]:
                cell += (l1_dist <= r).astype(int)
        elif shape_type == 4:  # Rings
            for i, r in enumerate([r1, r2, r3, r4, r5]):
                if i % 2 == 0:
                    cell += (dist <= r).astype(int)
                else:
                    cell -= (dist <= r).astype(int)
        
        # Threshold to binary
        cell = (cell > len([r1, r2, r3, r4, r5]) // 2).astype(int)
        
        return enforce_symmetry(cell)
    
    def ca_evolve(seed_cell, rule, steps):
        """
        Evolve a 2D cellular automaton with a custom symmetric rule.
        rule: integer encoding the rule (simplified for 2D)
        """
        cell = seed_cell.copy().astype(int)
        for _ in range(steps):
            new_cell = np.zeros_like(cell)
            for i in range(size):
                for j in range(size):
                    # Get 3x3 neighborhood (toroidal)
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni = (i + di) % size
                            nj = (j + dj) % size
                            neighbors = (neighbors << 1) | cell[ni, nj]
                    # Apply rule: use center-weighted majority
                    center = cell[i, j]
                    neighbor_sum = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % size
                            nj = (j + dj) % size
                            neighbor_sum += cell[ni, nj]
                    
                    # Simple rule: if majority of neighbors agree with center, keep; else flip
                    if neighbor_sum >= 4:
                        new_cell[i, j] = 1
                    elif neighbor_sum <= 3:
                        new_cell[i, j] = 0
                    else:
                        new_cell[i, j] = 1 - center
            cell = new_cell
        return enforce_symmetry(cell)
    
    # Strategy: Try multiple structural families
    best_cell = None
    best_score = -float('inf')
    
    # Family 1: Concentric geometric patterns
    print("Trying geometric patterns...")
    for shape_type in range(5):
        if time_left() < 15:
            break
        for r1 in np.linspace(0.05, 0.45, 8):
            if time_left() < 10:
                break
            for r2 in np.linspace(0.05, 0.45, 8):
                if time_left() < 10:
                    break
                for r3 in np.linspace(0.05, 0.45, 6):
                    if time_left() < 10:
                        break
                    for r4 in np.linspace(0.05, 0.45, 6):
                        if time_left() < 10:
                            break
                        for r5 in np.linspace(0.05, 0.45, 6):
                            if time_left() < 10:
                                break
                            cell = make_cell_from_radii(r1, r2, r3, r4, r5, shape_type)
                            score = calculate_gap(cell)
                            if score > best_score:
                                best_score = score
                                best_cell = cell.copy()
    
    # Family 2: Cellular automaton exploration
    if time_left() > 30:
        print("Trying CA patterns...")
        np.random.seed(42)
        for trial in range(min(int(time_left() / 8), 20)):
            if time_left() < 10:
                break
            
            # Random seed pattern with symmetry
            seed = np.random.randint(0, 2, (size, size))
            seed = enforce_symmetry(seed)
            
            # Evolve with different step counts
            for steps in [1, 2, 3, 4, 5]:
                if time_left() < 10:
                    break
                cell = ca_evolve(seed, 0, steps)
                score = calculate_gap(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
    
    # Family 3: Fourier-based (keep as fallback)
    if time_left() > 20 and (best_cell is None or best_score < 0.3):
        print("Trying Fourier patterns...")
        
        def make_cell_from_fourier(params):
            A1, A2, A3, A4, A5, p1, p2, p3, p4, p5, thresh = params
            field = np.zeros((size, size), dtype=float)
            field += A1 * np.cos(2 * np.pi * (XX + YY) + p1)
            field += A2 * np.cos(2 * np.pi * (XX - YY) + p2)
            field += A3 * np.cos(4 * np.pi * XX + p3)
            field += A4 * np.cos(4 * np.pi * YY + p4)
            field += A5 * np.cos(4 * np.pi * (XX + YY) + p5)
            cell = (field >= thresh).astype(int)
            return enforce_symmetry(cell)
        
        candidates = [
            [0.4, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0],
            [0.3, 0.3, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0],
            [0.0, 0.0, 0.4, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.0],
            [-0.4, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0],
        ]
        
        for cand in candidates:
            if time_left() < 10:
                break
            cell = make_cell_from_fourier(cand)
            score = calculate_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Final fallback
    if best_cell is None or best_score < 0:
        r = 0.2
        val = XX**2 + YY**2
        cell = (val <= r**2).astype(int)
        best_cell = enforce_symmetry(cell)
    
    return best_cell.tolist()
