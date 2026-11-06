"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy: Topological optimization using field intensity guidance
    1. Fast internal PWE solver for TM modes
    2. Start with a reasonable initial design
    3. Use field intensities of bands 4 and 5 to guide pixel flips
    4. Add dielectric where band 4 has high intensity, remove where band 5 has high intensity
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 200s limit
    
    # Fast internal PWE solver for TM modes
    def compute_tm_bands(cell, n_pw=13, n_kpts=15):
        """
        Compute TM band structure using plane-wave expansion.
        Returns gap-midgap ratio and field data for bands 4 and 5.
        """
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        # Get Fourier coefficients of inv_eps via FFT
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Plane wave indices
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        
        n_total = len(g_indices)
        
        # Precompute S matrix: S[i,j] = inv_eps(G_i - G_j)
        S = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            mi, ni = g_indices[i]
            for j in range(n_total):
                mj, nj = g_indices[j]
                gm = mi - mj
                gn = ni - nj
                idx_m = gm % N
                idx_n = gn % N
                S[i, j] = fft_coeffs[idx_m, idx_n]
        
        # k-points along the irreducible BZ boundary
        k_points = []
        n_seg = n_kpts // 3
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg):
            k_points.append((t, t))
        
        # Remove duplicates
        k_unique = []
        for kp in k_points:
            if not any(abs(kp[0]-k2[0]) < 1e-10 and abs(kp[1]-k2[1]) < 1e-10 for k2 in k_unique):
                k_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_k4 = None  # k-point where band 4 is maximized
        best_k5 = None  # k-point where band 5 is minimized
        field4 = None
        field5 = None
        
        for (kx, ky) in k_unique:
            Gk = np.array([(m + kx, n + ky) for m, n in g_indices])
            Gk_sq = Gk[:, 0]**2 + Gk[:, 1]**2
            
            A = S * Gk_sq[None, :]
            B = S
            
            try:
                from scipy.linalg import eig as sp_eig
                vals, vecs = sp_eig(A, B)
                vals = np.real(vals)
                vals = vals[vals > 1e-6]
                vals = np.sort(vals)
            except:
                try:
                    B_inv_A = np.linalg.solve(B, A)
                    vals = np.linalg.eigvals(B_inv_A)
                    vals = np.real(vals)
                    vals = vals[vals > 1e-6]
                    vals = np.sort(vals)
                except:
                    continue
            
            if len(vals) < 5:
                continue
            
            omega4 = vals[3]
            omega5 = vals[4]
            
            if omega4 > max_omega4:
                max_omega4 = omega4
                best_k4 = (kx, ky)
            if omega5 < min_omega5:
                min_omega5 = omega5
                best_k5 = (kx, ky)
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            return -1.0, None, None
        
        # Compute field intensities at critical k-points
        field4_int = None
        field5_int = None
        
        if best_k4 is not None:
            field4_int = _compute_field_intensity(cell, best_k4, S, g_indices, band_idx=3)
        if best_k5 is not None:
            field5_int = _compute_field_intensity(cell, best_k5, S, g_indices, band_idx=4)
        
        if min_omega5 + max_omega4 <= 0:
            return -1.0, field4_int, field5_int
        
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4), field4_int, field5_int
    
    def _compute_field_intensity(cell, kpoint, S, g_indices, band_idx=3):
        """Compute field intensity on real-space grid for a given band."""
        kx, ky = kpoint
        Gk = np.array([(m + kx, n + ky) for m, n in g_indices])
        Gk_sq = Gk[:, 0]**2 + Gk[:, 1]**2
        
        A = S * Gk_sq[None, :]
        B = S
        
        try:
            from scipy.linalg import eig as sp_eig
            vals, vecs = sp_eig(A, B)
            vals = np.real(vals)
            valid = vals > 1e-6
            vals = vals[valid]
            vecs = vecs[:, valid]
            order = np.argsort(vals)
            vecs = vecs[:, order]
        except:
            return None
        
        if band_idx >= vecs.shape[1]:
            return None
        
        c = vecs[:, band_idx]
        
        # Reconstruct field on real-space grid
        N = size
        field = np.zeros((N, N), dtype=complex)
        for idx, (m, n) in enumerate(g_indices):
            field += c[idx] * np.exp(2j * np.pi * ((m + kx) * np.arange(N)[:, None] / N + 
                                                    (n + ky) * np.arange(N)[None, :] / N))
        
        return np.abs(field)**2
    
    def apply_symmetric_flips(cell, flip_mask):
        """Apply flips with D4 symmetry enforcement."""
        new_cell = cell.copy()
        new_cell[flip_mask] = 1 - new_cell[flip_mask]
        
        # Enforce D4 symmetry
        new_cell = np.maximum(new_cell, new_cell[::-1, :])
        new_cell = np.maximum(new_cell, new_cell[:, ::-1])
        new_cell = np.maximum(new_cell, new_cell.T)
        new_cell = np.maximum(new_cell, new_cell[::-1, ::-1])
        
        return new_cell
    
    def make_initial_cell():
        """Create initial design: four small rods in quadrants."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = xx - cx
            dy = yy - cy
            r = np.sqrt(dx**2 + dy**2)
            cell[r <= 0.12] = 1
        
        return cell
    
    # Initialize
    current_cell = make_initial_cell()
    best_score, _, _ = compute_tm_bands(current_cell.tolist())
    best_candidate = current_cell.tolist()
    
    # Topological optimization loop
    n_flips = 4
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < deadline - 30:
        # Get field intensities
        _, field4_int, field5_int = compute_tm_bands(current_cell.tolist())
        
        if field4_int is None or field5_int is None:
            # Fallback: random flips
            flip_mask = np.zeros((size, size), dtype=int)
            indices = np.random.choice(size * size, n_flips, replace=False)
            for idx in indices:
                i, j = divmod(idx, size)
                if i >= j:
                    flip_mask[i, j] = 1
        else:
            # Field-guided flips
            # Add dielectric where band 4 has high intensity (pushes band 4 down)
            # Remove dielectric where band 5 has high intensity (pushes band 5 up)
            
            # Normalize fields
            f4 = field4_int / (np.max(field4_int) + 1e-10)
            f5 = field5_int / (np.max(field5_int) + 1e-10)
            
            # Score for adding dielectric: high f4, low f5
            add_score = f4 - 0.5 * f5
            # Score for removing dielectric: low f4, high f5
            remove_score = f5 - 0.5 * f4
            
            flip_mask = np.zeros((size, size), dtype=int)
            
            # Try to add dielectric at top add_score locations where cell is 0
            add_candidates = np.argwhere((current_cell == 0) & (add_score > 0))
            if len(add_candidates) > 0:
                add_scores = add_score[add_candidates[:, 0], add_candidates[:, 1]]
                top_add = add_candidates[np.argsort(add_scores)[-min(n_flips//2, len(add_candidates)):]]
                for i, j in top_add:
                    if i >= j:  # Upper triangle for symmetry
                        flip_mask[i, j] = 1
            
            # Try to remove dielectric at top remove_score locations where cell is 1
            remove_candidates = np.argwhere((current_cell == 1) & (remove_score > 0))
            if len(remove_candidates) > 0:
                remove_scores = remove_score[remove_candidates[:, 0], remove_candidates[:, 1]]
                top_remove = remove_candidates[np.argsort(remove_scores)[-min(n_flips//2, len(remove_candidates)):]]
                for i, j in top_remove:
                    if i >= j:
                        flip_mask[i, j] = 1  # This will flip 1->0
        
        # Apply flips
        new_cell = apply_symmetric_flips(current_cell, flip_mask)
        
        # Evaluate
        new_score, _, _ = compute_tm_bands(new_cell.tolist())
        
        if new_score > best_score:
            best_score = new_score
            best_candidate = new_cell.tolist()
            current_cell = new_cell
            no_improve_count = 0
            # Increase flips slightly
            n_flips = min(n_flips + 1, 8)
        else:
            no_improve_count += 1
            # Decrease flips if stuck
            if no_improve_count > 3:
                n_flips = max(2, n_flips - 1)
        
        if no_improve_count > max_no_improve:
            # Restart from best with perturbation
            no_improve_count = 0
            n_flips = 4
            current_cell = np.array(best_candidate, dtype=int)
            # Add random perturbation
            random_mask = np.random.rand(size, size) > 0.95
            current_cell = apply_symmetric_flips(current_cell, random_mask)
    
    return best_candidate

# EVOLVE-BLOCK-END
