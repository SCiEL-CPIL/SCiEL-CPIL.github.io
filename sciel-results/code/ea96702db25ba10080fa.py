"""Design a 32x32 photonic crystal unit cell to maximize TM band 4-5 gap."""
import numpy as np
import time
import math

def design(problem):
    """
    Design a symmetric 32x32 unit cell for a 2D square-lattice photonic crystal
    with epsilon=12 dielectric in air, maximizing the gap between TM bands 4 and 5.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill
    
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
    
    def calculate_gap(cell, n_pw=9):
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
        
        # Precompute inv_eps Fourier coefficients using FFT
        inv_eps_F = np.zeros((nG, nG), dtype=complex)
        for ig, (gx, gy) in enumerate(G_list):
            # Compute FFT of inv_eps
            fft_vals = np.fft.fft2(inv_eps)
            # The FFT gives coefficients for G = (gx, gy) where gx, gy are in [-N/2, N/2)
            # We need to map our G-vectors to FFT indices
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[ig, ig] = fft_vals[fft_gx, fft_gy] / (N * N)
        
        # Actually, let's compute the full 2D FFT once and extract the coefficients we need
        inv_eps_fft = np.fft.fft2(inv_eps)
        
        # Rebuild inv_eps_F properly
        inv_eps_F = {}
        for gx, gy in G_list:
            fft_gx = gx % N
            fft_gy = gy % N
            inv_eps_F[(gx, gy)] = inv_eps_fft[fft_gx, fft_gy] / (N * N)
        
        # Key k-points in the irreducible BZ (in units where BZ is [0, 0.5]x[0, 0.5])
        # Irreducible BZ: 0 <= ky <= kx <= 0.5
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
            # M_{G,G'} = (G-k)·(G'-k) * inv_eps_F[G-G']
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
    
    # Strategy: Parameterize using low-order Fourier harmonics
    # The dielectric distribution is: eps(x,y) = eps_mid + sum of harmonics
    # We'll use a superposition of modes that couple to bands 4 and 5
    
    # For TM modes in a square lattice, the relevant G-vectors for the 4-5 gap
    # are typically (1,0), (0,1), (1,1), (2,0), (0,2), (2,2), etc.
    
    # Let's parameterize the cell as a sum of sinusoidal terms
    # cell(x,y) = threshold(eps_base + A1*cos(2*pi*(x+y)) + A2*cos(2*pi*(x-y)) + 
    #                         A3*cos(4*pi*x) + A4*cos(4*pi*y) + A5*cos(4*pi*(x+y)))
    
    def make_cell_from_fourier(params):
        """
        Create a cell from Fourier harmonic parameters.
        params: [A1, A2, A3, A4, A5, phase1, phase2, phase3, phase4, phase5, threshold]
        """
        A1, A2, A3, A4, A5, p1, p2, p3, p4, p5, thresh = params
        
        # Compute the field
        field = np.zeros((size, size), dtype=float)
        field += A1 * np.cos(2 * np.pi * (XX + YY) + p1)
        field += A2 * np.cos(2 * np.pi * (XX - YY) + p2)
        field += A3 * np.cos(4 * np.pi * XX + p3)
        field += A4 * np.cos(4 * np.pi * YY + p4)
        field += A5 * np.cos(4 * np.pi * (XX + YY) + p5)
        
        # Threshold to get 0/1 cell
        cell = (field >= thresh).astype(int)
        
        return enforce_symmetry(cell)
    
    # Initial candidates
    best_cell = None
    best_score = -float('inf')
    
    # Try a grid of Fourier parameters
    # Amplitudes: -0.5 to 0.5
    # Phases: 0, pi/2, pi, 3pi/2
    # Threshold: -0.5 to 0.5
    
    # Start with a few hand-tuned candidates
    candidates = []
    
    # Candidate 1: Strong (1,1) mode
    candidates.append([0.4, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0])
    
    # Candidate 2: Strong (1,0) and (0,1) modes
    candidates.append([0.3, 0.3, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0])
    
    # Candidate 3: Strong (2,0) mode
    candidates.append([0.0, 0.0, 0.4, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0])
    
    # Candidate 4: Strong (2,2) mode
    candidates.append([0.0, 0.0, 0.0, 0.0, 0.4, 0, 0, 0, 0, 0, 0.0])
    
    # Candidate 5: Mixed modes
    candidates.append([0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.0])
    
    # Candidate 6: With phase shifts
    candidates.append([0.3, 0.3, 0.0, 0.0, 0.0, 0, math.pi/2, 0, 0, 0, 0.0])
    
    # Candidate 7: Different threshold
    candidates.append([0.4, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.1])
    
    # Candidate 8: Negative amplitude
    candidates.append([-0.4, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0])
    
    # Evaluate initial candidates
    for cand in candidates:
        if time_left() < 10:
            break
        cell = make_cell_from_fourier(cand)
        score = calculate_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local optimization using coordinate descent
    if best_cell is not None and time_left() > 30:
        # Get the best parameters
        best_params = candidates[0].copy()
        best_score_local = best_score
        
        # Try to find which candidate gave the best score
        for cand in candidates:
            cell = make_cell_from_fourier(cand)
            if np.array_equal(cell, best_cell):
                best_params = cand.copy()
                break
        
        # Coordinate descent on each parameter
        param_ranges = {
            0: (-0.5, 0.5, 0.05),   # A1
            1: (-0.5, 0.5, 0.05),   # A2
            2: (-0.5, 0.5, 0.05),   # A3
            3: (-0.5, 0.5, 0.05),   # A4
            4: (-0.5, 0.5, 0.05),   # A5
            5: (0, 2*math.pi, math.pi/4),  # p1
            6: (0, 2*math.pi, math.pi/4),  # p2
            7: (0, 2*math.pi, math.pi/4),  # p3
            8: (0, 2*math.pi, math.pi/4),  # p4
            9: (0, 2*math.pi, math.pi/4),  # p5
            10: (-0.5, 0.5, 0.05),  # threshold
        }
        
        # Do a few rounds of coordinate descent
        for round_num in range(3):
            if time_left() < 15:
                break
            
            improved = False
            for param_idx in range(11):
                if time_left() < 10:
                    break
                
                lo, hi, step = param_ranges[param_idx]
                current_val = best_params[param_idx]
                
                # Try values around current
                for delta in [-step, -step/2, step/2, step, 2*step]:
                    new_val = current_val + delta
                    if new_val < lo or new_val > hi:
                        continue
                    
                    new_params = best_params.copy()
                    new_params[param_idx] = new_val
                    
                    cell = make_cell_from_fourier(new_params)
                    score = calculate_gap(cell)
                    
                    if score > best_score_local:
                        best_score_local = score
                        best_params = new_params
                        best_cell = cell.copy()
                        improved = True
                        break  # Move to next parameter
        
            if not improved:
                break
    
    # If we still have time, try a random search
    if time_left() > 20:
        np.random.seed(42)
        n_random = int(time_left() / 5)  # ~5s per random trial
        
        for _ in range(min(n_random, 50)):
            if time_left() < 10:
                break
            
            # Random parameters
            params = [
                np.random.uniform(-0.5, 0.5),  # A1
                np.random.uniform(-0.5, 0.5),  # A2
                np.random.uniform(-0.5, 0.5),  # A3
                np.random.uniform(-0.5, 0.5),  # A4
                np.random.uniform(-0.5, 0.5),  # A5
                np.random.uniform(0, 2*math.pi),  # p1
                np.random.uniform(0, 2*math.pi),  # p2
                np.random.uniform(0, 2*math.pi),  # p3
                np.random.uniform(0, 2*math.pi),  # p4
                np.random.uniform(0, 2*math.pi),  # p5
                np.random.uniform(-0.5, 0.5),  # threshold
            ]
            
            cell = make_cell_from_fourier(params)
            score = calculate_gap(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    # Final fallback: if we didn't find a good cell, use a simple circular rod
    if best_cell is None or best_score < 0:
        # Circular rod, radius 0.2
        r = 0.2
        val = XX**2 + YY**2
        cell = (val <= r**2).astype(int)
        best_cell = enforce_symmetry(cell)
    
    return best_cell.tolist()
