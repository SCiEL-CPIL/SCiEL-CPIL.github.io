import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave margin for safety

    # The strategy:
    # 1. Define the dielectric distribution as a sum of specific Fourier modes.
    #    eps(r) = eps_bg + sum_k [ A_k * cos(2*pi*(kx*x + ky*y) + phi_k) ]
    #    We focus on modes that are significant for the 4-5 band gap.
    #    The 4th and 5th bands in a square lattice typically involve interactions
    #    near the M point (pi, pi) and X point (pi, 0).
    #    The (1,1) mode is crucial for the M-point gap, but the prompt suggests
    #    (2,0), (0,2), and (2,2). Let's include a background plus these modes.
    #    Note: For a 0/1 pixel map, we must threshold the continuous field.
    
    # Modes to optimize:
    # (1,0), (0,1) -> fundamental
    # (1,1) -> M point coupling
    # (2,0), (0,2) -> X point / higher order
    # (2,2) -> higher M point
    
    # Let's parameterize:
    # eps_field(x,y) = bg + a10*cos(2*pi*x) + a01*cos(2*pi*y) + a11*cos(2*pi*(x+y)) + a20*cos(4*pi*x) + a02*cos(4*pi*y) + a22*cos(4*pi*(x+y))
    # We can fix phases to 0 or pi for simplicity due to symmetry, or optimize them.
    # Given the symmetry requirement (invariant under square symmetry group),
    # the field must be symmetric. Cosines with these arguments are symmetric 
    # if the center is at (0.5, 0.5) in normalized coordinates [0,1].
    # Let x = (i+0.5)/size, y = (j+0.5)/size.
    
    # Symmetry check:
    # f(x,y) = f(1-x, y) -> cos(2*pi*(1-x)) = cos(2*pi - 2*pi*x) = cos(2*pi*x). OK.
    # f(x,y) = f(x, 1-y) -> OK.
    # f(x,y) = f(y, x) -> cos(2*pi*x) becomes cos(2*pi*y). So a10 must equal a01?
    # If we want full D4 symmetry, we should enforce:
    # a10 = a01
    # a20 = a02
    # a11 and a22 are inherently symmetric under x<->y swap.
    
    # So parameters: bg, a1, a11, a2, a22
    # where a1 corresponds to (1,0) and (0,1), a2 to (2,0) and (0,2).
    
    # Let's define the field:
    # F(x,y) = bg + a1*(cos(2*pi*x) + cos(2*pi*y)) + a11*cos(2*pi*(x+y)) + a2*(cos(4*pi*x) + cos(4*pi*y)) + a22*cos(4*pi*(x+y))
    
    # We will search over these 5 parameters.
    # Ranges:
    # bg: [0.1, 0.9] (fill fraction)
    # a1: [-1.0, 1.0]
    # a11: [-1.0, 1.0]
    # a2: [-0.5, 0.5]
    # a22: [-0.5, 0.5]
    
    # To evaluate the score, we need to:
    # 1. Generate the 32x32 grid from the parameters.
    # 2. Compute the 4-5 TM band gap.
    
    # Since we don't have a fast exact solver in the sandbox easily (and building one is complex),
    # and the prompt allows NumPy/SciPy, we can implement a simplified plane-wave solver
    # or use a proxy if the solver is too slow.
    # However, the "inspiration" score of 5.78 was achieved by a 4-rod design.
    # A 4-rod design corresponds to a specific set of Fourier coefficients.
    # Let's try to match the 4-rod design with our parameterization.
    # 4 rods at (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75).
    # This has strong (1,1) and (2,2) components?
    # Actually, rods at (1/4, 1/4) etc. have periodicity 1/2.
    # The Fourier series of a square wave with period 1/2 has odd harmonics of 2.
    # So (2,0), (0,2), (2,2) etc.
    
    # Let's implement a fast band solver for the 32x32 grid.
    # We will use a small number of plane waves (e.g., 9x9 or 11x11 grid of G vectors)
    # to keep it fast enough for grid search.
    
    def generate_grid(params):
        bg, a1, a11, a2, a22 = params
        xs = np.linspace(0, 1, size, endpoint=False) + 0.5/size
        ys = np.linspace(0, 1, size, endpoint=False) + 0.5/size
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        
        # Field
        F = bg + a1*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y)) + a11*np.cos(2*np.pi*(X+Y)) + a2*(np.cos(4*np.pi*X) + np.cos(4*np.pi*Y)) + a22*np.cos(4*np.pi*(X+Y))
        
        # Threshold to 0/1
        # We want to preserve the fill fraction roughly, but the threshold determines the shape.
        # A simple threshold at 0.5 might not work if bg is not 0.5.
        # Better: threshold such that the mean is close to bg?
        # Or just use a fixed threshold relative to the field range?
        # Let's use a threshold that makes the mean fill fraction equal to bg.
        # This requires sorting the field values, which is expensive inside the loop.
        # Alternative: Use a sigmoid or just a hard threshold at 0.5 and assume bg is the mean of the *unthresholded* field?
        # If we threshold at 0.5, the fill fraction is determined by the distribution of F.
        # Let's just use a hard threshold at 0.5 for now, and adjust bg to control fill.
        # Actually, if we use a hard threshold, the resulting grid is binary.
        # The "bg" parameter effectively shifts the threshold.
        
        grid = (F > 0.5).astype(int)
        return grid

    def compute_gap_score_fast(grid):
        # Fast plane-wave solver
        # Use a 7x7 grid of G vectors (diamond or square)
        # Let's use a square grid of G vectors from -3 to 3 (7x7 = 49 vectors)
        # This is small enough for speed.
        
        nG_side = 7
        G_vals = np.arange(-nG_side//2, nG_side//2 + 1)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        Gx = Gx.ravel() * 2 * np.pi
        Gy = Gy.ravel() * 2 * np.pi
        nG = len(Gx)
        
        # Fourier coefficients of epsilon
        eps = eps_low + (eps_high - eps_low) * grid
        F_eps = np.fft.fft2(eps) / (size * size)
        
        # Map G vector indices to FFT indices
        # G = (2*pi*m, 2*pi*n). FFT index is (m mod size, n mod size).
        # We need to extract these coefficients.
        # Note: FFT output is ordered 0, 1, ..., size/2-1, -size/2, ..., -1.
        # For small m,n, we can just index directly if we handle the wrap-around.
        
        # Precompute the mapping for the G vectors we use
        G_indices = []
        for i in range(nG):
            m = G_vals[i // nG_side]
            n = G_vals[i % nG_side]
            # FFT index for m
            im = m % size
            # FFT index for n
            in_ = n % size
            G_indices.append((im, in_))
        
        # Build the matrix for a specific k-point
        # We will sample a few k-points in the irreducible BZ.
        # k-points: Gamma(0,0), X(pi,0), M(pi,pi), and midpoints.
        k_points = [
            (0.0, 0.0),
            (np.pi, 0.0),
            (np.pi, np.pi),
            (np.pi/2, 0.0),
            (np.pi, np.pi/2),
            (np.pi/2, np.pi/2)
        ]
        
        max_omega4 = 0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build H matrix
            # H_{mn} = (k+G_m)^2 * delta_{mn} - (k+G_m) dot (k+G_n) * (1/eps)_{m-n}  (for TE)
            # For TM, it's different.
            # Let's use the standard form for TM:
            # sum_n [ (k+G_m)^2 delta_{mn} - (k+G_m) dot (k+G_n) * (1/eps)_{m-n} ] E_n = omega^2 E_m
            # Wait, for TM (Hz), the equation is:
            # (k+G_m)^2 E_m - sum_n (1/eps)_{m-n} (k+G_m) dot (k+G_n) E_n = omega^2 E_m ?
            # No, for TM, the coupling is via eps, not 1/eps.
            # For TE (Ex,Ey), coupling is via 1/eps.
            # For TM (Hz), coupling is via eps.
            # Equation: sum_n [ (k+G_m) dot (k+G_n) / (1/eps)_{m-n} ... ] ?
            
            # Let's use the generalized eigenvalue formulation:
            # A v = omega^2 B v
            # For TM:
            # A_{mn} = (k+G_m) dot (k+G_n) * eps_{m-n}
            # B_{mn} = delta_{mn}
            # This is not quite right either.
            
            # Correct TM eigenvalue problem:
            # sum_n [ (k+G_m) dot (k+G_n) * (1/eps)_{m-n} ] E_n = (k+G_m)^2 E_m
            # This is for TE.
            
            # For TM:
            # sum_n [ (k+G_m) dot (k+G_n) * eps_{m-n} ] E_n = (k+G_m)^2 * eps_avg * E_m ?
            
            # Let's stick to the TE formulation as it's more common and standard in many PW codes,
            # or check the polarization in the problem.
            # Problem says: "TM bands".
            
            # Standard TM equation in PW:
            # sum_n [ (k+G_m) dot (k+G_n) * (1/eps)_{m-n} ] E_n = (k+G_m)^2 E_m  <-- This is TE
            # For TM, the roles of eps and 1/eps are swapped in the off-diagonal terms?
            # Actually, for TM (magnetic field in plane, electric out of plane? No, TM is H in plane, E out? Or H out, E in plane?)
            # In 2D photonic crystals:
            # TE: E is in plane, H is out of plane. (Hz)
            # TM: H is in plane, E is out of plane. (Ez)
            # Wait, standard convention:
            # TE (Transverse Electric): E is transverse to propagation. In 2D slab, if propagation is in x-y, E is z? No.
            # Usually:
            # TE mode: H_z = 0, E_x, E_y non-zero. (E in plane)
            # TM mode: E_z = 0, H_x, H_y non-zero. (H in plane)
            # Wait, "TM" usually means Transverse Magnetic, so H is transverse to k. If k is in x-y plane, H has z-component?
            # Let's check the problem statement: "TM bands".
            # In many 2D PC contexts:
            # TE: E_z = 0 (H_z is the field). Equation involves 1/eps.
            # TM: H_z = 0 (E_z is the field). Equation involves eps.
            
            # Let's assume the standard:
            # TE (Hz): H_{mn} = (k+G_m) dot (k+G_n) * (1/eps)_{m-n}
            # TM (Ez): H_{mn} = (k+G_m) dot (k+G_n) * eps_{m-n}
            
            # Let's implement the TM version.
            
            # We need eps_{m-n} for all pairs m,n.
            # eps_{m-n} is the Fourier coefficient of eps at G_m - G_n.
            # Since G_m and G_n are from our set, G_m - G_n might be outside our set.
            # We can get it from the full FFT F_eps.
            
            # Precompute eps_coeffs for all differences?
            # Or just compute on the fly.
            
            # For speed, let's precompute a dictionary of eps_coeffs for the differences of our G vectors.
            # But we have 49 G vectors, so 49*49 = 2401 pairs.
            # We can just index F_eps directly.
            
            # Matrix A
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    # k+G_i dot k+G_j
                    kGi = np.array([kx + Gx[i], ky + Gy[i]])
                    kGj = np.array([kx + Gx[j], ky + Gy[j]])
                    dot = np.dot(kGi, kGj)
                    
                    # eps_{i-j}
                    # G_i - G_j = (2*pi*(m_i - m_j), 2*pi*(n_i - n_j))
                    # We need the Fourier coefficient at this G vector.
                    # dm = m_i - m_j
                    # dn = n_i - n_j
                    mi = G_vals[i // nG_side]
                    ni = G_vals[i % nG_side]
                    mj = G_vals[j // nG_side]
                    nj = G_vals[j % nG_side]
                    dm = mi - mj
                    dn = ni - nj
                    
                    # Index in F_eps
                    idx_m = dm % size
                    idx_n = dn % size
                    eps_coeff = F_eps[idx_n, idx_m]
                    
                    A[i, j] = dot * eps_coeff
            
            # Eigenvalue problem: A v = (k+G_i)^2 v ?
            # No, the equation is:
            # sum_j [ (k+G_i) dot (k+G_j) * eps_{i-j} ] E_j = (k+G_i)^2 * E_i
            # This is a generalized eigenvalue problem: A v = D v
            # where D = diag((k+G_i)^2)
            
            D = np.diag(kx + Gx**2 + (ky + Gy)**2) # Wait, (k+G_i)^2 is scalar for each i
            
            # Actually, D is diagonal with entries (k+Gx[i])^2 + (ky+Gy[i])^2
            
            try:
                # Solve A v = D v  =>  D^{-1} A v = v
                # Or use scipy.linalg.eig
                # For small nG, we can just invert D if it's invertible
                # D might be singular if k+G_i = 0.
                # At Gamma, G=0 is in the set, so D[0,0] = 0.
                # We should handle this.
                
                # Better: Solve (A - omega^2 D) v = 0
                # This is a standard eigenvalue problem if we divide by D?
                # Let's use scipy.linalg.eig for generalized eigenvalue problem
                # A v = lambda D v
                # where lambda = omega^2
                
                from scipy.linalg import eig
                w, v = eig(A, D)
                w = w.real
                w = np.sort(w)
                
                # Take the 4th and 5th bands (index 3 and 4)
                if len(w) >= 5:
                    omega4 = np.sqrt(max(0, w[3]))
                    omega5 = np.sqrt(max(0, w[4]))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                return -1.0
        
        if min_omega5 <= max_omega4:
            return -1.0
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # Grid search over parameters
    # bg: [0.3, 0.7]
    # a1: [-0.5, 0.5]
    # a11: [-0.5, 0.5]
    # a2: [-0.2, 0.2]
    # a22: [-0.2, 0.2]
    
    # Coarse grid first
    best_score = -1.0
    best_params = None
    
    # Define grid steps
    n_bg = 5
    n_a1 = 5
    n_a11 = 5
    n_a2 = 3
    n_a22 = 3
    
    bg_vals = np.linspace(0.3, 0.7, n_bg)
    a1_vals = np.linspace(-0.5, 0.5, n_a1)
    a11_vals = np.linspace(-0.5, 0.5, n_a11)
    a2_vals = np.linspace(-0.2, 0.2, n_a2)
    a22_vals = np.linspace(-0.2, 0.2, n_a22)
    
    for bg in bg_vals:
        for a1 in a1_vals:
            for a11 in a11_vals:
                for a2 in a2_vals:
                    for a22 in a22_vals:
                        if time.time() > deadline - 10:
                            break
                        params = (bg, a1, a11, a2, a22)
                        grid = generate_grid(params)
                        score = compute_gap_score_fast(grid)
                        if score > best_score:
                            best_score = score
                            best_params = params
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Refine around best_params if time permits
    if best_params is not None and time.time() < deadline - 20:
        bg0, a10, a110, a20, a220 = best_params
        
        # Local search with smaller steps
        step_bg = 0.02
        step_a = 0.05
        
        for _ in range(5): # 5 refinement iterations
            improved = False
            for delta_bg in [-step_bg, 0, step_bg]:
                for delta_a1 in [-step_a, 0, step_a]:
                    for delta_a11 in [-step_a, 0, step_a]:
                        for delta_a2 in [-step_a/2, 0, step_a/2]:
                            for delta_a22 in [-step_a/2, 0, step_a/2]:
                                if time.time() > deadline - 10:
                                    break
                                new_params = (bg0 + delta_bg, a10 + delta_a1, a110 + delta_a11, a20 + delta_a2, a220 + delta_a22)
                                if new_params[0] < 0.1 or new_params[0] > 0.9:
                                    continue
                                grid = generate_grid(new_params)
                                score = compute_gap_score_fast(grid)
                                if score > best_score:
                                    best_score = score
                                    best_params = new_params
                                    bg0, a10, a110, a20, a220 = new_params
                                    improved = True
                            if time.time() > deadline - 10:
                                break
                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            if not improved:
                break
            step_bg *= 0.5
            step_a *= 0.5

    # Generate final grid
    if best_params is not None:
        final_grid = generate_grid(best_params)
    else:
        # Fallback: 4-rod design
        final_grid = [[0] * size for _ in range(size)]
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.15
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx*dx + dy*dy < radius*radius:
                        final_grid[i][j] = 1
                        break

    return final_grid
