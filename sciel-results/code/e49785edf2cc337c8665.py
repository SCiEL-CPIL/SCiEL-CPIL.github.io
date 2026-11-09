import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s limit
    
    N = 32
    eps_low = 1.0
    eps_high = 12.0
    lower_band = 4  # 1-indexed, so we want gap between index 3 and 4 (0-indexed)
    
    # Plane wave setup
    # 289 plane waves implies (2*8+1)^2 = 17^2 = 289. So cutoff is 8.
    N_PW = 8
    G_vals = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vecs = np.column_stack((Gx.ravel(), Gy.ravel()))
    
    # Precompute k-points
    # Irreducible BZ of square lattice: Gamma-X-M-Gamma
    # Gamma: (0,0), X: (pi, 0), M: (pi, pi)
    # Using 10 points per segment (excluding endpoints to avoid duplicates mostly, but including is fine)
    k_points = []
    # Gamma to X
    for i in range(10):
        kx = i * np.pi / 9.0
        ky = 0.0
        k_points.append((kx, ky))
    # X to M
    for i in range(1, 10):
        kx = np.pi
        ky = i * np.pi / 9.0
        k_points.append((kx, ky))
    # M to Gamma
    for i in range(1, 10):
        kx = np.pi * (1 - i/9.0)
        ky = np.pi * (1 - i/9.0)
        k_points.append((kx, ky))
    
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2 for each k-point
    # (k+G)^2 = kx^2 + ky^2 + 2*pi*(kx*Gx + ky*Gy) + (2*pi)^2*(Gx^2+Gy^2)
    # We need the matrix elements for the TM mode:
    # H = (1/eps) * (k+G)^2
    # In matrix form: D * E = omega^2 E, where D_ij = <i|1/eps|j> * |k+Gi|^2
    # Actually, the standard PWE for TM is:
    # (1/eps(r)) (k+G)^2 E_{k+G} = omega^2 sum_{G'} (1/eps_{G-G'}) E_{k+G'}
    # Let's use the form:
    # sum_{G'} M_{G,G'} E_{G'} = omega^2 E_G
    # M_{G,G'} = delta_{G,G'} (k+G)^2 + sum_{G''} (1/eps)_{G-G''} ... wait.
    
    # Correct PWE matrix construction:
    # For TM:
    # (1/eps) * (k+G)^2 * E_{k+G} = omega^2 * E_k
    # Expanding 1/eps in Fourier: 1/eps(r) = sum_G' (1/eps)_{G'} e^{i G' r}
    # E_k = sum_G E_G e^{i (k+G) r}
    # The product (1/eps) E_k involves convolution.
    # The eigenvalue problem is:
    # sum_{G'} [ (k+G)^2 delta_{G,G'} + sum_{G''} (1/eps)_{G-G''} (k+G')^2 delta_{G', G''} ] ... no.
    
    # Let's stick to the standard formulation found in literature (e.g., Joannopoulos):
    # For TM:
    # (k+G)^2 E_G = omega^2 sum_{G'} (1/eps)_{G-G'} E_{G'}
    # This is a generalized eigenvalue problem or can be converted to standard.
    # Actually, for TM, the matrix is:
    # D_{G,G'} = (k+G)^2 delta_{G,G'} - omega^2 (1/eps)_{G-G'}
    # This is implicit in omega.
    
    # Better form:
    # sum_{G'} (1/eps)_{G-G'} (k+G')^2 E_{G'} = omega^2 (k+G)^2 E_G ? No.
    
    # Let's use the formulation:
    # H = V * K^2 * V^{-1} ? No.
    
    # Standard form for TM:
    # (1/eps) (k+G)^2 E_G = omega^2 E_k
    # Multiply by e^{-i k r} and integrate...
    # The matrix equation is:
    # sum_{G'} (1/eps)_{G-G'} (k+G')^2 E_{G'} = omega^2 (k+G)^2 E_G
    # Wait, the operator (k+G)^2 acts on E_G.
    # So: sum_{G'} (1/eps)_{G-G'} (k+G')^2 E_{G'} = omega^2 (k+G)^2 E_G
    # This is: M E = omega^2 D E
    # Where M_{G,G'} = (1/eps)_{G-G'} (k+G')^2
    # And D_{G,G} = (k+G)^2
    
    # To solve: D^{-1/2} M D^{-1/2} u = omega^2 u
    # where E = D^{-1/2} u
    
    # So we need:
    # 1. Fourier coefficients of 1/eps: eps_inv_FFT
    # 2. For each k:
    #    a. Compute kg_sq = (k+G)^2 for all G in basis
    #    b. Construct M: M[i,j] = eps_inv_FFT[G_i - G_j] * kg_sq[j]
    #    c. Construct S = diag(1/sqrt(kg_sq)) * M * diag(1/sqrt(kg_sq))
    #    d. Eigenvalues of S are omega^2
    
    # Let's verify the indices.
    # G_i - G_j must be in the range of available Fourier coefficients.
    # Since we have N_PW=8, we have coefficients for G in [-8, 8].
    # G_i - G_j can range from -16 to 16.
    # We need to handle aliasing or ensure we have enough coefficients.
    # With N=32, we can resolve up to G=16.
    # So we should compute eps_inv_FFT with at least 16 harmonics.
    # Let's use N_FFT = 32, so we get coefficients for G in [-16, 15] (or similar).
    
    # Precompute G differences
    Gx_flat = G_vecs[:, 0]
    Gy_flat = G_vecs[:, 1]
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Map differences to FFT indices
    # FFT of size 32 gives indices 0..31.
    # Frequency f corresponds to index i = f % 32.
    # f ranges from -16 to 16.
    # idx_x = (Diff_x + 16) % 32
    # idx_y = (Diff_y + 16) % 32
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32
    
    # Precompute kg_sq and inv_sqrt for each k-point
    KG_SQ = []
    KG_INV_SQRT = []
    for k in k_points:
        kx, ky = k
        kg = k[None, :] + G_vecs * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        KG_SQ.append(kg_sq)
        KG_INV_SQRT.append(1.0 / np.sqrt(kg_sq))
    KG_SQ = np.array(KG_SQ)
    KG_INV_SQRT = np.array(KG_INV_SQRT)
    
    def evaluate_cell(cell):
        """
        Evaluate the gap-midgap ratio for a 32x32 cell.
        cell: list of lists of 0/1. 1=dielectric (eps=12), 0=air (eps=1).
        """
        try:
            eps_arr = np.array(cell, dtype=float)
            # Map 0/1 to eps values
            # 1 -> 12, 0 -> 1
            eps_arr = np.where(eps_arr > 0.5, eps_high, eps_low)
            mu = 1.0 / eps_arr
            
            # Compute Fourier coefficients of mu (1/eps)
            mu_fft = np.fft.fft2(mu)
            mu_fft = np.fft.fftshift(mu_fft)
            
            # Get coefficients for all G-G' combinations
            # mu_diff[i,j] = mu_fft[idx_x[i,j], idx_y[i,j]] / N^2
            mu_diff = mu_fft[idx_x, idx_y] / (N * N)
            
            min_omega_5 = float('inf')
            max_omega_4 = float('-inf')
            
            for ki in range(len(k_points)):
                # Construct S matrix
                # S[i,j] = mu_diff[i,j] * kg_sq[k,j] * inv_sqrt_kg_sq[k,i] * inv_sqrt_kg_sq[k,j]
                # Wait, the formula was:
                # S = D^{-1/2} M D^{-1/2}
                # M[i,j] = mu_diff[i,j] * kg_sq[j]
                # D^{-1/2}[i,i] = 1/sqrt(kg_sq[i])
                # So S[i,j] = (1/sqrt(kg_sq[i])) * mu_diff[i,j] * kg_sq[j] * (1/sqrt(kg_sq[j]))
                # S[i,j] = mu_diff[i,j] * sqrt(kg_sq[j] / kg_sq[i])
                
                # Let's use the precomputed arrays
                kg_sq_k = KG_SQ[ki]
                inv_sqrt_kg_sq_k = KG_INV_SQRT[ki]
                
                # S = mu_diff * (kg_sq[None, :] * inv_sqrt_kg_sq[:, None]) / kg_sq[None, :]? No.
                # S[i,j] = mu_diff[i,j] * kg_sq[j] * inv_sqrt_kg_sq[i] * inv_sqrt_kg_sq[j]
                
                # Efficiently:
                # Scale rows by inv_sqrt_kg_sq
                # Scale columns by sqrt(kg_sq)
                # Actually: S = diag(inv_sqrt) * mu_diff * diag(kg_sq) * diag(inv_sqrt)
                # S = diag(inv_sqrt) * mu_diff * diag(sqrt(kg_sq))
                
                # Let's just do the element-wise multiplication
                # factor[i,j] = inv_sqrt_kg_sq[i] * sqrt_kg_sq[j]
                sqrt_kg_sq_k = np.sqrt(kg_sq_k)
                factor = inv_sqrt_kg_sq_k[:, None] * sqrt_kg_sq_k[None, :]
                S = mu_diff * factor
                
                # Ensure symmetry (should be real and symmetric if mu is real)
                # mu_diff is complex? No, mu is real, so mu_fft is Hermitian.
                # But due to numerical errors, it might have small imaginary parts.
                # Take real part.
                S = np.real(S)
                S = (S + S.T) / 2.0 # Enforce symmetry
                
                # Solve eigenvalue problem
                # We only care about the 4th and 5th bands (indices 3 and 4)
                # But to be safe, we might need to compute more?
                # The lowest eigenvalues correspond to the lowest bands.
                # So we need the 4th and 5th smallest eigenvalues.
                
                evals = np.linalg.eigvalsh(S)
                
                # Sort is already done by eigvalsh
                w4_sq = evals[3]
                w5_sq = evals[4]
                
                w4 = np.sqrt(w4_sq)
                w5 = np.sqrt(w5_sq)
                
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
            
            gap = min_omega_5 - max_omega_4
            midgap = (min_omega_5 + max_omega_4) / 2.0
            if midgap == 0:
                return 0.0
            return gap / midgap
        except Exception:
            return 0.0

    # Search Strategy:
    # Use a parametric family of symmetric shapes to reduce the search space.
    # Then refine with pixel flips.
    
    def parametric_cell(params):
        """
        Generate a symmetric 32x32 cell from parameters.
        params: [cx, cy, rx, ry, angle, thickness]
        We use C4v symmetry, so we only need to define the top-left quadrant (8x8).
        """
        # For simplicity, let's start with a few hardcoded good candidates
        # and then refine.
        
        # Candidate 1: Large dielectric cross
        cell = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                # Cross shape
                if i > 14 and i < 18 or j > 14 and j < 18:
                    cell[i][j] = 1
        return cell

    def candidate_cross(thickness):
        cell = [[0]*N for _ in range(N)]
        half = N // 2
        for i in range(N):
            for j in range(N):
                # Dielectric background with air holes? Or air background with dielectric?
                # Knowledge says: "Inverting the material assignment (Dielectric background with Air holes ... leverages TM polarization ... likely opening a significantly larger band gap."
                # So let's try Dielectric Background (1) with Air Holes (0).
                cell[i][j] = 1
        # Carve out air holes in a cross pattern
        # Central hole
        hole_size = thickness
        for i in range(half - hole_size, half + hole_size):
            for j in range(half - hole_size, half + hole_size):
                # Make a cross shape hole? Or just square hole?
                # Let's try a square hole first
                cell[i][j] = 0
        # Symmetry is automatic for this construction
        return cell

    # Let's try a few initial candidates
    candidates = []
    
    # 1. Dielectric background, central square air hole (size 4x4)
    c1 = [[1]*N for _ in range(N)]
    for i in range(14, 18):
        for j in range(14, 18):
            c1[i][j] = 0
    candidates.append(c1)
    
    # 2. Dielectric background, central circular air hole (approx)
    c2 = [[1]*N for _ in range(N)]
    center = 16
    radius = 4.0
    for i in range(N):
        for j in range(N):
            if (i - center)**2 + (j - center)**2 < radius**2:
                c2[i][j] = 0
    candidates.append(c2)
    
    # 3. Dielectric background, four air holes in corners of the central region
    c3 = [[1]*N for _ in range(N)]
    hole_centers = [(12, 12), (12, 20), (20, 12), (20, 20)]
    r = 2.5
    for i in range(N):
        for j in range(N):
            for cx, cy in hole_centers:
                if (i - cx)**2 + (j - cy)**2 < r**2:
                    c3[i][j] = 0
    candidates.append(c3)
    
    # 4. Air background, central dielectric rod
    c4 = [[0]*N for _ in range(N)]
    for i in range(14, 18):
        for j in range(14, 18):
            c4[i][j] = 1
    candidates.append(c4)
    
    # 5. Air background, cross dielectric
    c5 = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if 14 <= i <= 17 or 14 <= j <= 17:
                c5[i][j] = 1
    candidates.append(c5)

    best_cell = None
    best_score = -np.inf
    
    # Evaluate all candidates
    for c in candidates:
        score = evaluate_cell(c)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in c]
        if time.time() > deadline:
            return best_cell

    # Local Search: Pixel Flips
    # We will perform a local search starting from the best candidate.
    # To maintain symmetry, we flip pixels in the top-left 8x8 quadrant and mirror them.
    
    def get_quadrant(cell):
        return [row[:16] for row in cell[:16]]
    
    def set_quadrant(cell, quad):
        new_cell = [row[:] for row in cell]
        for i in range(16):
            for j in range(16):
                val = quad[i][j]
                new_cell[i][j] = val
                new_cell[i][N-1-j] = val
                new_cell[N-1-i][j] = val
                new_cell[N-1-i][N-1-j] = val
        return new_cell

    current_cell = best_cell
    current_score = best_score
    
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try flipping each pixel in the quadrant
        # Random order to avoid bias
        indices = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indices)
        
        for i, j in indices:
            if time.time() > deadline - 10: # Leave time for final check
                break
                
            # Flip pixel (i,j) and its symmetric counterparts
            new_cell = [row[:] for row in current_cell]
            
            # Flip (i,j)
            new_cell[i][j] = 1 - new_cell[i][j]
            # Mirror to (i, N-1-j)
            new_cell[i][N-1-j] = new_cell[i][j]
            # Mirror to (N-1-i, j)
            new_cell[N-1-i][j] = new_cell[i][j]
            # Mirror to (N-1-i, N-1-j)
            new_cell[N-1-i][N-1-j] = new_cell[i][j]
            
            # Check if the cell changed
            if new_cell == current_cell:
                continue
                
            new_score = evaluate_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_cell = [row[:] for row in current_cell]
                    
            # Note: We don't backtrack immediately if it's worse, we just keep trying
            # This is a simple hill climbing. We might get stuck.
            # To escape, we could add simulated annealing, but let's keep it simple first.
            
    return best_cell
