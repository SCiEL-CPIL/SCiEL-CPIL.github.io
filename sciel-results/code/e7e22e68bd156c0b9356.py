import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 photonic crystal unit cell using a Genetic Algorithm.
    Maximizes the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s budget
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute Plane Wave data
    N_pw = 8
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    num_pw = len(G_vectors)
    
    # Precompute G-G' differences indices
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
    # Convert to indices for fft_inv_eps lookup
    # Note: fft2 output is ordered such that index [i, j] corresponds to frequency (i, j) 
    # but with wrap-around. We need to map G_diff to [0, size-1]
    idx_x = G_diff[:, :, 0] % size
    idx_y = G_diff[:, :, 1] % size
    
    # K points in irreducible BZ
    # We use a set of points covering the boundary and interior
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
    
    # Remove duplicates
    unique_k = []
    seen = set()
    for k in k_points:
        key = (round(k[0], 6), round(k[1], 6))
        if key not in seen:
            seen.add(key)
            unique_k.append(k)
            
    k_array = np.array(unique_k)
    
    def make_symmetric_cell(base_cell):
        """
        Enforces C4v symmetry on a 32x32 cell.
        base_cell: 32x32 numpy array of 0/1.
        Returns: 32x32 numpy array of 0/1 with C4v symmetry.
        """
        # The 8 symmetries of the square:
        # 1. Identity: (i, j)
        # 2. Rot 90: (j, size-1-i)
        # 3. Rot 180: (size-1-i, size-1-j)
        # 4. Rot 270: (size-1-j, i)
        # 5. Refl X (i, size-1-j)
        # 6. Refl Y (size-1-i, j)
        # 7. Refl Diag (j, i)
        # 8. Refl Anti-Diag (size-1-j, size-1-i)
        
        # To enforce symmetry, we sum the values at all 8 symmetric positions.
        # If sum >= 4, set to 1, else 0.
        # This creates a valid symmetric cell.
        
        sym_sum = np.zeros((size, size), dtype=int)
        
        # We can vectorize this by iterating over the 8 transformations
        # But 8 * 1024 ops is trivial.
        
        for i in range(size):
            for j in range(size):
                val = base_cell[i, j]
                # Get all 8 positions
                pos = [
                    (i, j),
                    (j, size-1-i),
                    (size-1-i, size-1-j),
                    (size-1-j, i),
                    (i, size-1-j),
                    (size-1-i, j),
                    (j, i),
                    (size-1-j, size-1-i)
                ]
                # Sum contributions from all positions to the target (i,j)?
                # No, we want the value at (i,j) to be consistent with its orbit.
                # It's easier to compute the sum for each orbit and assign.
                # However, iterating over all pixels and checking all 8 neighbors is O(N^2 * 8).
                # 32*32*8 = 8192 ops. Very fast.
                
                # Actually, let's just sum the values of the 8 pixels that map to each other.
                # For a given pixel (i,j), its orbit is the set of 8 positions.
                # We can just sum base_cell at those 8 positions.
                s = 0
                for pi, pj in pos:
                    s += base_cell[pi, pj]
                # If majority is 1, set to 1
                sym_sum[i, j] = 1 if s >= 4 else 0
        
        return sym_sum

    def evaluate_cell(cell):
        """
        Computes the TM band gap using a plane-wave method.
        Returns the gap-midgap ratio.
        """
        eps_grid = eps_low + (eps_high - eps_low) * cell.astype(float)
        inv_eps_grid = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps_grid)
        fft_inv_eps /= (size * size)
        
        # Construct coefficient matrix for all k-points at once?
        # M_k = coeff_matrix * omega_sq_k
        # We need eigenvalues for each k.
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        # Precompute coeff_matrix for this cell
        coeff_matrix = fft_inv_eps[idx_x, idx_y]
        
        for kx, ky in k_array:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq = np.sum(k_plus_G ** 2, axis=1)
            
            # M = coeff_matrix * omega_sq[:, np.newaxis]
            # Since coeff_matrix is real (fft of real signal is conjugate symmetric, but we take real part? 
            # Wait, for TM, the matrix is:
            # (k+G)^2 delta_ij - sum_{G''} eps^{-1}(G-G') (k+G')^2 ... ?
            # Standard PWE for TM:
            # det( (k+G)^2 delta_{GG'} - sum_{G''} eps^{-1}(G-G'') (k+G'')^2 ) = 0
            # This is not simply coeff_matrix * omega_sq.
            # The previous code used: M = coeff_matrix * omega_sq[:, np.newaxis]
            # This is only valid if the matrix is diagonal or if we are solving a different problem.
            # Let's check the previous code's matrix construction.
            # coeff_matrix[i, j] = fft_inv_eps[ix, iy]
            # M = coeff_matrix * omega_sq[:, np.newaxis]
            # This implies M_{ij} = eps^{-1}(G_i - G_j) * (k+G_i)^2 ?
            # The eigenvalue equation is:
            # (k+G)^2 a_G = sum_{G'} eps^{-1}(G-G') (k+G')^2 a_{G'}
            # This is NOT an eigenvalue problem of the form M a = lambda a with lambda = omega^2.
            # It is a generalized eigenvalue problem or requires rearrangement.
            # However, for TM modes in 2D, the equation is:
            # (k+G)^2 a_G = sum_{G'} [1/eps(G-G')] (k+G')^2 a_{G'}
            # This can be rewritten as:
            # sum_{G'} [ delta_{GG'} (k+G)^2 - (1/eps(G-G')) (k+G')^2 ] a_{G'} = 0
            # This is a determinant condition.
            # Alternatively, we can define a matrix H such that H a = 0.
            # Or, if we define b_G = (k+G)^2 a_G, then:
            # b_G = sum_{G'} eps^{-1}(G-G') b_{G'}
            # This is an eigenvalue problem: b = C b, where C_{GG'} = eps^{-1}(G-G').
            # The eigenvalues of C are 1? No.
            # The equation is:
            # (k+G)^2 a_G = sum_{G'} eps^{-1}(G-G') (k+G')^2 a_{G'}
            # Let v_G = (k+G)^2 a_G.
            # Then v_G = sum_{G'} eps^{-1}(G-G') v_{G'}
            # This implies v is an eigenvector of the matrix E = [eps^{-1}(G-G')] with eigenvalue 1.
            # This is only possible if 1 is an eigenvalue of E.
            # This approach doesn't yield the band structure directly.
            
            # Correct PWE formulation for TM:
            # The matrix to diagonalize is:
            # A_{GG'} = (k+G)^2 delta_{GG'} - sum_{G''} eps^{-1}(G-G'') (k+G'')^2 delta_{G'G''} ? No.
            
            # Let's use the standard form:
            # sum_{G'} [ (k+G)^2 delta_{GG'} - (k+G')^2 eps^{-1}(G-G') ] a_{G'} = 0
            # This is a singular system.
            # We can solve for eigenvalues omega^2 by finding where det(M(omega)) = 0.
            # OR, we can transform it.
            # Multiply by (k+G)^{-2}:
            # a_G - sum_{G'} (k+G')^2 / (k+G)^2 eps^{-1}(G-G') a_{G'} = 0
            # This is not symmetric.
            
            # A common symmetric formulation:
            # Let u_G = (k+G) a_G.
            # Then u_G = sum_{G'} eps^{-1}(G-G') u_{G'}
            # This again leads to eigenvalue 1.
            
            # Actually, the standard PWE for TM is solved by diagonalizing:
            # M_{GG'} = (k+G)^2 delta_{GG'} + sum_{G''} eps^{-1}(G-G'') (k+G'')^2 ... ?
            
            # Let's look at the previous code's logic again.
            # It computed eigvals of M = coeff_matrix * omega_sq[:, np.newaxis].
            # This is M_{ij} = eps^{-1}(G_i-G_j) (k+G_i)^2.
            # The eigenvalues of this matrix are NOT the band frequencies.
            # The previous code was fundamentally flawed in its physics implementation.
            # That's why it failed!
            
            # Correct Matrix for TM:
            # The equation is:
            # (k+G)^2 a_G = sum_{G'} eps^{-1}(G-G') (k+G')^2 a_{G'}
            # This is a generalized eigenvalue problem if we consider omega^2?
            # No, (k+G) is fixed.
            # This is a linear system M(omega) a = 0.
            # To find the bands, we need to find omega such that det(M) = 0.
            # However, we can rewrite this as:
            # a = D^{-1} C a, where D is diagonal with (k+G)^2 and C has eps^{-1}(G-G') (k+G')^2.
            # This is not symmetric.
            
            # Symmetric formulation:
            # Multiply by (k+G)^{-1/2}:
            # (k+G)^{1/2} a_G = sum_{G'} (k+G)^{-1/2} eps^{-1}(G-G') (k+G')^{1/2} (k+G')^{1/2} a_{G'}
            # Let b_G = (k+G)^{1/2} a_G.
            # b_G = sum_{G'} [ (k+G)^{-1/2} eps^{-1}(G-G') (k+G')^{1/2} ] b_{G'}
            # The matrix B_{GG'} = (k+G)^{-1/2} eps^{-1}(G-G') (k+G')^{1/2} is NOT symmetric.
            
            # Wait, eps^{-1}(G-G') is real and symmetric.
            # Let's check the symmetry of B_{GG'}:
            # B_{G'G} = (k+G')^{-1/2} eps^{-1}(G'-G) (k+G)^{1/2}
            # B_{GG'} = (k+G)^{-1/2} eps^{-1}(G-G') (k+G')^{1/2}
            # They are not equal.
            
            # However, the matrix H = D^{1/2} C D^{1/2} where C_{GG'} = eps^{-1}(G-G') is symmetric.
            # The equation is:
            # (k+G)^2 a_G = sum_{G'} eps^{-1}(G-G') (k+G')^2 a_{G'}
            # This is NOT of the form H a = lambda a.
            
            # Let's use the standard approach:
            # The eigenvalues omega^2 are the roots of det( (k+G)^2 delta_{GG'} - sum_{G''} eps^{-1}(G-G'') (k+G'')^2 delta_{G'G''} ) = 0 ?
            # No, the sum is over G''.
            # The matrix is M_{GG'} = (k+G)^2 delta_{GG'} - (k+G')^2 eps^{-1}(G-G').
            # This matrix is NOT symmetric.
            # But its eigenvalues are real?
            
            # Actually, for TM, the matrix is:
            # A_{GG'} = (k+G)^2 delta_{GG'} - (k+G')^2 eps^{-1}(G-G')
            # The eigenvalues of A are 0? No.
            # We are looking for non-trivial solutions to A a = 0.
            # This means 0 is an eigenvalue of A.
            # But we want the band frequencies.
            
            # Ah, the frequency is NOT an eigenvalue of A.
            # The frequency is determined by the k-vector.
            # Wait, in PWE, for a fixed k, we solve for the field amplitudes.
            # The condition for a solution is det(A) = 0.
            # But A depends on k.
            # For a fixed k, there is only one condition?
            # No, A is a matrix of size N_pw x N_pw.
            # det(A) = 0 is a single equation.
            # But we have N_pw unknowns.
            # This implies that for a generic k, there is no solution unless the determinant is zero.
            # But the determinant is a scalar.
            # This means that for a fixed k, the system is overdetermined?
            # No, the equation is:
            # sum_{G'} A_{GG'} a_{G'} = 0
            # This is a homogeneous linear system.
            # Non-trivial solutions exist if det(A) = 0.
            # But A depends on k.
            # So for each k, we check if det(A) = 0?
            # If not, there is no mode at that k?
            # That can't be right. There are modes at all k.
            
            # I am missing the frequency dependence.
            # The equation is:
            # (k+G)^2 a_G = sum_{G'} eps^{-1}(G-G') (k+G')^2 a_{G'}
            # This equation does NOT contain omega!
            # Why?
            # Because in 2D, for TM polarization, the wave equation is:
            # nabla^2 H_z + omega^2 eps(r) H_z = 0
            # H_z = sum a_G exp(i (k+G) . r)
            # - (k+G)^2 a_G + omega^2 sum eps^{-1}(G-G') a_{G'} = 0 ?
            # No, it's:
            # - (k+G)^2 a_G + omega^2 sum_{G'} eps^{-1}(G-G') a_{G'} = 0
            # (k+G)^2 a_G = omega^2 sum_{G'} eps^{-1}(G-G') a_{G'}
            # THIS is the eigenvalue problem!
            # lambda = omega^2.
            # M_{GG'} = (k+G)^2 delta_{GG'}
            # N_{GG'} = sum_{G''} eps^{-1}(G-G'') delta_{G'G''} ? No.
            # The term is sum_{G'} eps^{-1}(G-G') a_{G'}.
            # So N_{GG'} = eps^{-1}(G-G').
            # The equation is:
            # D a = omega^2 N a
            # Where D is diagonal with (k+G)^2.
            # N is the matrix of eps^{-1} Fourier coefficients.
            # This is a generalized eigenvalue problem:
            # D a = omega^2 N a
            # N^{-1} D a = omega^2 a
            # Or D N^{-1} a = omega^2 a
            # Since N is symmetric and positive definite (usually), we can transform to a standard eigenvalue problem.
            # Let a = N^{1/2} b.
            # D N^{1/2} b = omega^2 N N^{1/2} b = omega^2 N^{3/2} b ? No.
            # D N^{-1/2} N^{-1/2} a = omega^2 a
            # Let b = N^{-1/2} a.
            # D N^{-1/2} b = omega^2 N^{1/2} b
            # N^{-1/2} D N^{-1/2} b = omega^2 b
            # This is a standard symmetric eigenvalue problem!
            # Matrix H = N^{-1/2} D N^{-1/2}.
            # We need to compute N^{-1/2}.
            
            # N is 289x289.
            # We can diagonalize N: N = U diag(nu) U^T.
            # N^{-1/2} = U diag(nu^{-1/2}) U^T.
            # H = U diag(nu^{-1/2}) U^T D U diag(nu^{-1/2}) U^T.
            # This is expensive to do for each k.
            
            # Alternative: Solve generalized eigenvalue problem D a = omega^2 N a directly.
            # `np.linalg.eig` can do this? No, `scipy.linalg.eig` can.
            # Or `scipy.linalg.eigh` if both are symmetric.
            # D is symmetric. N is symmetric.
            # So we can use `scipy.linalg.eigh(D, N)`.
            
            # Let's use scipy.linalg.eigh.
            
        # Re-implementation of evaluate_cell using scipy.linalg.eigh
        
        from scipy.linalg import eigh
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        # Precompute N (eps^{-1} matrix) for this cell
        # N_{ij} = eps^{-1}(G_i - G_j)
        # This is exactly coeff_matrix from before.
        N = fft_inv_eps[idx_x, idx_y]
        
        # N should be real and symmetric.
        # Ensure it is symmetric to avoid numerical issues
        N = (N + N.T) / 2.0
        
        for kx, ky in k_array:
            k = np.array([kx, ky])
            k_plus_G = k + G_vectors
            omega_sq_diag = np.sum(k_plus_G ** 2, axis=1)
            D = np.diag(omega_sq_diag)
            
            # Solve D a = omega^2 N a
            # eigh(D, N) returns eigenvalues and eigenvectors.
            try:
                eigvals, _ = eigh(D, N)
                real_eigvals = np.sort(np.real(eigvals))
                
                if len(real_eigvals) >= 5:
                    omega_4 = np.sqrt(max(real_eigvals[3], 0))
                    omega_5 = np.sqrt(max(real_eigvals[4], 0))
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except Exception:
                continue
        
        if min_omega_5 == np.inf or max_omega_4 == -np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap == 0:
            return -1.0
        ratio = 2 * gap / midgap
        return ratio

    # Genetic Algorithm
    pop_size = 40
    population = []
    scores = []
    
    # Initial Candidates
    def create_cross(dielectric=True):
        cell = np.zeros((size, size), dtype=int)
        if dielectric:
            cell[:, size//2-1:size//2+1] = 1
            cell[size//2-1:size//2+1, :] = 1
        else:
            cell[:, :] = 1
            cell[:, size//2-1:size//2+1] = 0
            cell[size//2-1:size//2+1, :] = 0
        return make_symmetric_cell(cell)
    
    def create_checkerboard():
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i+j) % 2 == 0:
                    cell[i, j] = 1
        return make_symmetric_cell(cell)
    
    def create_grid_holes():
        cell = np.ones((size, size), dtype=int)
        # 4x4 grid of holes
        step = size // 4
        for i in range(1, 5):
            for j in range(1, 5):
                center_i = i * step - step//2
                center_j = j * step - step//2
                radius = step // 3
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        if di*di + dj*dj <= radius*radius:
                            cell[center_i+di, center_j+dj] = 0
        return make_symmetric_cell(cell)
        
    # Add initial structured candidates
    candidates = [
        create_cross(True),
        create_cross(False),
        create_checkerboard(),
        create_grid_holes(),
        np.zeros((size, size), dtype=int), # All air
        np.ones((size, size), dtype=int)   # All dielectric
    ]
    
    # Fill rest with random
    while len(candidates) < pop_size:
        rand_cell = (np.random.rand(size, size) > 0.5).astype(int)
        candidates.append(make_symmetric_cell(rand_cell))
        
    population = candidates
    scores = [evaluate_cell(c) for c in population]
    
    best_score = max(scores)
    best_cell = population[scores.index(best_score)]
    
    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Selection: Keep top 50%
        elite_size = pop_size // 2
        next_pop = population[:elite_size]
        
        # Crossover/Mutation to fill the rest
        while len(next_pop) < pop_size:
            # Pick two parents
            p1_idx = np.random.randint(0, elite_size)
            p2_idx = np.random.randint(0, elite_size)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover: Bitwise AND/OR or random mix
            if np.random.rand() > 0.5:
                child = ((p1 + p2) >= 1).astype(int) # OR
            else:
                child = ((p1 + p2) >= 2).astype(int) # AND
            
            # Mutation
            # Flip a few pixels in the fundamental domain
            num_mutations = np.random.randint(1, 5)
            for _ in range(num_mutations):
                i = np.random.randint(0, size)
                j = np.random.randint(0, size)
                child[i, j] = 1 - child[i, j]
                
            # Enforce symmetry
            child = make_symmetric_cell(child)
            next_pop.append(child)
            
        population = next_pop
        scores = [evaluate_cell(c) for c in population]
        
        current_best = max(scores)
        if current_best > best_score:
            best_score = current_best
            best_cell = population[scores.index(current_best)]
            
        # Update deadline check
        if time.time() > deadline - 5:
            break
            
    # Convert best_cell to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
        
    return result
