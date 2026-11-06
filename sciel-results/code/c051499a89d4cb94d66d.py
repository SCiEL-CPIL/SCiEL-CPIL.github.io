import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # Use 15x15 = 225 plane waves for the internal search loop.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                c_sym[i, j] = val
                c_sym[j, i] = val
                c_sym[31-i, 31-j] = val
                c_sym[31-j, 31-i] = val
                c_sym[i, 31-j] = val
                c_sym[31-j, i] = val
                c_sym[31-i, j] = val
                c_sym[j, 31-i] = val
        return c_sym

    # 2. NEW STRATEGY: Parametric Search using Product of 1D Gratings
    # Represent the cell as the threshold of a function f(x,y) = A*sin(2*pi*fx*x) * B*sin(2*pi*fy*y)
    # This generates checkerboard-like structures which are excellent for coupling (2,0) and (2,2) modes.
    # We search over fx, fy, A, B, and threshold T.
    
    # Precompute x and y coordinates
    x_coords = np.arange(size, dtype=float)
    y_coords = np.arange(size, dtype=float)
    
    def generate_cell(fx, fy, phase_x, phase_y, threshold):
        """
        Generate a symmetric cell from parameters.
        fx, fy: spatial frequencies (in cycles per unit cell)
        phase_x, phase_y: phase shifts
        threshold: threshold for binarization
        """
        # We want to ensure symmetry about the center.
        # The function must be symmetric.
        # A simple symmetric function is cos(2*pi*fx*x) * cos(2*pi*fy*y)
        # For integer fx, fy, this is symmetric about x=0, 0.5, etc.
        # We need symmetry about the center of the 32x32 grid.
        # The center is at 15.5.
        # cos(2*pi*fx*(x - 15.5)) is symmetric about 15.5 if fx is integer?
        # cos(2*pi*fx*(15.5 - d)) = cos(2*pi*fx*15.5 - 2*pi*fx*d)
        # If fx is integer, 2*pi*fx*15.5 is a multiple of pi.
        # cos(pi - a) = -cos(a). Not symmetric.
        # cos(2*pi*fx*x) with fx integer is periodic.
        # Let's just generate the pattern and enforce symmetry afterwards.
        
        # Use a grid of coordinates
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Define the potential function
        # We want to maximize Fourier components at (2,0) and (2,2).
        # A product of two cosines with frequencies fx and fy will have components at (fx, fy), (fx, -fy), etc.
        # If we pick fx=1, fy=1, we get (1,1).
        # If we pick fx=2, fy=0, we get (2,0).
        # To get both, maybe a sum? Or a product of different frequencies?
        # Let's try: f = cos(2*pi*fx*X) * cos(2*pi*fy*Y)
        # If fx=1, fy=1, components at (1,1), (1,-1), (-1,1), (-1,-1).
        # This helps with (2,2) if we consider second harmonics or if the band structure couples them.
        # Actually, the 4-5 gap is often driven by (2,0) and (2,2) in the *inverse* permittivity or permittivity.
        
        # Let's try a more flexible form:
        # f = a*cos(2*pi*fx*X + px) + b*cos(2*pi*fy*Y + py)
        # Or product:
        f = np.cos(2.0 * np.pi * fx * X) * np.cos(2.0 * np.pi * fy * Y)
        
        # Binarize
        cell = (f > threshold).astype(int)
        
        # Enforce symmetry
        cell = enforce_full_symmetry(cell)
        
        return cell

    best_score = -1.0
    best_cell = np.zeros((size, size), dtype=int)
    
    # Define search space
    # Frequencies: 1, 2, 3, 4
    # Thresholds: -1.0 to 1.0
    # We will do a coarse grid search first, then local refinement.
    
    freqs = [1.0, 1.5, 2.0, 2.5, 3.0]
    thresholds = np.linspace(-0.8, 0.8, 9)
    
    # Coarse Search
    for fx in freqs:
        for fy in freqs:
            for th in thresholds:
                if time.time() > deadline - 10.0:
                    break
                
                cell = generate_cell(fx, fy, 0.0, 0.0, th)
                score = compute_score(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    best_params = (fx, fy, th)
                    
                if time.time() > deadline - 10.0:
                    break

    # Local Refinement around best parameters
    if best_score > -0.5:
        fx0, fy0, th0 = best_params
        # Refine fx, fy, th
        for _ in range(3):
            fx_step = 0.1
            fy_step = 0.1
            th_step = 0.05
            
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for dx in [-fx_step, 0, fx_step]:
                    for dy in [-fy_step, 0, fy_step]:
                        for dt in [-th_step, 0, th_step]:
                            if dx == 0 and dy == 0 and dt == 0:
                                continue
                            
                            fx_new = fx0 + dx
                            fy_new = fy0 + dy
                            th_new = th0 + dt
                            
                            if fx_new < 0.5 or fy_new < 0.5:
                                continue
                            if th_new < -1.0 or th_new > 1.0:
                                continue
                                
                            cell = generate_cell(fx_new, fy_new, 0.0, 0.0, th_new)
                            score = compute_score(cell)
                            
                            if score > best_score:
                                best_score = score
                                best_cell = cell.copy()
                                fx0, fy0, th0 = fx_new, fy_new, th_new
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    # Final check: if best_cell is invalid or score is bad, try a few random restarts?
    # No, we have time. Let's just return the best found.
    
    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
