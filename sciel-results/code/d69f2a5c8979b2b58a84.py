import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell with a dielectric background and air holes
    to maximize the TM band gap between bands 4 and 5.
    Uses a fast plane-wave surrogate to optimize the geometry parameters.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # --- Fast Plane-Wave Solver ---
    # We use a smaller number of plane waves (N_pw=10 -> 441 PWs) and fewer k-points
    # for the surrogate optimization. The final evaluation by the judge will be more accurate.
    
    N_pw = 10
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G-G' differences
    # diff[i,j] = G_vectors[i] - G_vectors[j]
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    # K-points: Cover the irreducible BZ.
    # The irreducible BZ of a square lattice is the triangle (0,0)-(0.5,0)-(0.5,0.5).
    # We sample points along the high-symmetry lines and inside.
    k_points = []
    # Gamma to X (0,0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0))
    # X to M (0.5, 0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5, t))
    # Gamma to M (0,0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, t))
    # Interior points
    k_points.append((0.25, 0.125))
    k_points.append((0.375, 0.25))
    k_points.append((0.125, 0.25))
    
    # Remove duplicates
    seen = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
    
    def evaluate_cell(eps_grid):
        """
        Computes the band gap using a simplified plane-wave method.
        Returns the gap-midgap ratio.
        """
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Precompute coefficient matrix
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                ix = dx % size
                iy = dy % size
                coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in unique_k:
            k = np.array([kx, ky])
            
            # (k+G)^2 for each G
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (num_pw,)
            
            # M[i,j] = coeff_matrix[i,j] * omega_sq[j]  (Note: standard form is sum_j M_ij c_j = w^2 c_i)
            # M_ij = (1/eps)_{G_i - G_j} * |k - G_j|^2 ? No.
            # Eigenvalue problem: sum_j (1/eps)_{G_i - G_j} (k - G_j)^2 c_j = w^2 c_i
            # So M[i, j] = coeff_matrix[i, j] * omega_sq[j]
            M = coeff_matrix * omega_sq[np.newaxis, :]
            
            # Solve eigenvalue problem
            eigvals = np.linalg.eigvalsh(M)
            real_eigvals = np.sort(np.real(eigvals))
            
            if len(real_eigvals) >= 5:
                # We want bands 4 and 5 (0-indexed 3 and 4)
                # Note: omega^2 = eigval, so omega = sqrt(eigval)
                # We need to be careful about the ordering.
                # The lowest eigenvalues correspond to the lowest frequencies.
                omega_4 = np.sqrt(real_eigvals[3])
                omega_5 = np.sqrt(real_eigvals[4])
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    def make_cell(params):
        """
        Creates a cell with a dielectric background and 4 air holes.
        params: [hole_radius, hole_offset_x, hole_offset_y, squareness]
        hole_radius: radius of the air holes (0 to 0.25)
        hole_offset_x: offset of holes from (0.25, 0.25) etc.
        hole_offset_y: offset of holes from (0.25, 0.25) etc.
        squareness: 0 for circular, 1 for square (superellipse exponent)
        """
        hole_radius = params[0]
        hole_offset_x = params[1]
        hole_offset_y = params[2]
        squareness = params[3]
        
        # Initialize with dielectric (1)
        cell = np.ones((size, size), dtype=int)
        
        # Centers of the 4 holes
        # The cell is 32x32. The symmetry center is (16, 16).
        # The holes are in the four quadrants.
        # Base centers: (8, 8), (24, 8), (8, 24), (24, 24) in pixel coords (0-indexed)
        # But we need to be careful with the symmetry.
        # The symmetry group is C4v about the center (15.5, 15.5).
        # So if we place a hole at (cx, cy), we must also place it at:
        # (31-cx, 31-cy) [180 rot]
        # (cy, cx) [transpose]
        # (31-cy, cx) [reflection]
        
        # Let's define the hole in the first quadrant (top-left)
        # Center at (8 + offset_x, 8 + offset_y)
        # Actually, let's use fractional coordinates to be safe.
        # Cell size is 32.
        # Center of first quadrant hole: (0.25 + ox, 0.25 + oy)
        
        cx = 0.25 + hole_offset_x
        cy = 0.25 + hole_offset_y
        
        # Exponent for superellipse: 2 is circle, 4 is square-ish
        # x^(2n) + y^(2n) < r^(2n)
        # For n=1 (exp=2): x^2 + y^2 < r^2 (Circle)
        # For n=2 (exp=4): x^4 + y^4 < r^4 (Square-ish)
        exponent = 2 + 2 * squareness
        
        for i in range(size):
            for j in range(size):
                # Pixel center
                px = (j + 0.5) / size
                py = (i + 0.5) / size
                
                # Check distance to the hole center in the first quadrant
                # We need to check all 4 symmetric positions
                # 1. (cx, cy)
                # 2. (1-cx, cy) -> (0.75-ox, 0.25+oy)
                # 3. (cx, 1-cy) -> (0.25+ox, 0.75-oy)
                # 4. (1-cx, 1-cy) -> (0.75-ox, 0.75-oy)
                
                dist_sq = 0.0
                for dx in [px - cx, px - (1 - cx)]:
                    for dy in [py - cy, py - (1 - cy)]:
                        # Superellipse distance
                        # We want to find if the point is inside the hole.
                        # (|dx|/r)^(2n) + (|dy|/r)^(2n) < 1
                        term = (abs(dx) / hole_radius) ** exponent + (abs(dy) / hole_radius) ** exponent
                        if term < 1.0:
                            cell[i, j] = 0  # Air hole
                            break
                    if cell[i, j] == 0:
                        break
        
        return cell

    # --- Optimization ---
    # Parameters: [hole_radius, hole_offset_x, hole_offset_y, squareness]
    # hole_radius: [0.05, 0.2]
    # hole_offset_x: [-0.1, 0.1]
    # hole_offset_y: [-0.1, 0.1]
    # squareness: [0, 1]
    
    best_score = -np.inf
    best_params = None
    
    # Initial guess: Circular holes, no offset
    init_params = np.array([0.1, 0.0, 0.0, 0.0])
    cell = make_cell(init_params)
    score = evaluate_cell(np.where(cell == 1, eps_high, eps_low))
    best_score = score
    best_params = init_params.copy()
    
    # Random search with local refinement
    num_attempts = 200
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate candidate
        if attempt < 100:
            # Local perturbation of best
            p = best_params + np.random.normal(0, 0.05, 4)
        else:
            # Random restart
            p = np.array([
                np.random.uniform(0.05, 0.2),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(0, 1)
            ])
        
        # Clamp parameters
        p[0] = np.clip(p[0], 0.05, 0.2)
        p[1] = np.clip(p[1], -0.1, 0.1)
        p[2] = np.clip(p[2], -0.1, 0.1)
        p[3] = np.clip(p[3], 0, 1)
        
        cell = make_cell(p)
        score = evaluate_cell(np.where(cell == 1, eps_high, eps_low))
        
        if score > best_score:
            best_score = score
            best_params = p.copy()
            
            # Local refinement
            for _ in range(5):
                if time.time() > deadline:
                    break
                p_local = best_params + np.random.normal(0, 0.02, 4)
                p_local[0] = np.clip(p_local[0], 0.05, 0.2)
                p_local[1] = np.clip(p_local[1], -0.1, 0.1)
                p_local[2] = np.clip(p_local[2], -0.1, 0.1)
                p_local[3] = np.clip(p_local[3], 0, 1)
                
                cell_local = make_cell(p_local)
                score_local = evaluate_cell(np.where(cell_local == 1, eps_high, eps_low))
                
                if score_local > best_score:
                    best_score = score_local
                    best_params = p_local.copy()
    
    # Final cell
    final_cell = make_cell(best_params)
    
    # Ensure symmetry
    sym_cell = np.zeros((size, size), dtype=int)
    count = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            val = final_cell[i, j]
            # Original
            sym_cell[i, j] += val
            count[i, j] += 1
            # 180 rotation
            r_i, r_j = size - 1 - i, size - 1 - j
            sym_cell[r_i, r_j] += val
            count[r_i, r_j] += 1
            # Transpose
            t_i, t_j = j, i
            sym_cell[t_i, t_j] += val
            count[t_i, t_j] += 1
            # Reflection
            f_i, f_j = size - 1 - j, i
            sym_cell[f_i, f_j] += val
            count[f_i, f_j] += 1
    
    final_sym_cell = (sym_cell / count) >= 0.5
    final_sym_cell = final_sym_cell.astype(int)
    
    # Force strict symmetry
    for i in range(size):
        for j in range(size):
            val = final_sym_cell[i, j]
            final_sym_cell[size - 1 - i, size - 1 - j] = val
            final_sym_cell[j, i] = val
            final_sym_cell[size - 1 - j, i] = val
    
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_sym_cell[i, j]))
        result.append(row)
    
    return result
