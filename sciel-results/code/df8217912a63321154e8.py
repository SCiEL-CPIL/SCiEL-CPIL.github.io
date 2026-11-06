import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy (structural change):
    1. Fast internal PWE solver.
    2. New design family: connected cross-shaped dielectric (plus sign) through the
       cell center, parameterized by arm width, cross thickness, and corner rounding.
       This creates a multi-scale dielectric distribution with strong in-plane
       connectivity that scatters TM modes differently than isolated rods.
    3. Coarse-to-fine parameter search over the continuous design space.
    4. Topological refinement guided by band-edge field intensities.
    5. D4 symmetry enforced throughout.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 220.0
    
    def get_g_indices(n_pw):
        half = n_pw // 2
        g_indices = []
        for m in range(-half, half + 1):
            for n in range(-half, half + 1):
                if m * m + n * n <= half * half + half:
                    g_indices.append((m, n))
        return np.array(g_indices, dtype=int)
    
    # Cache G-indices
    _g_cache = {}
    def get_g_cached(n_pw):
        if n_pw not in _g_cache:
            _g_cache[n_pw] = get_g_indices(n_pw)
        return _g_cache[n_pw]
    
    def compute_tm_bands(cell, n_pw=13, n_kpts=10, return_eigvecs=False):
        """Fast PWE band-structure evaluation for TM polarization."""
        cell = np.array(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        N = size
        
        fft_coeffs = np.fft.fft2(inv_eps) / (N * N)
        g_indices = get_g_cached(n_pw)
        
        m_diff = g_indices[:, 0][:, None] - g_indices[:, 0][None, :]
        n_diff = g_indices[:, 1][:, None] - g_indices[:, 1][None, :]
        m_idx = np.mod(m_diff, N)
        n_idx = np.mod(n_diff, N)
        S = fft_coeffs[m_idx, n_idx]
        
        # k-points in irreducible BZ: triangle (0,0)-(0.5,0)-(0.5,0.5)
        k_points = []
        n_seg = max(2, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((t, 0.0))
        for t in np.linspace(0, 0.5, n_seg + 1):
            k_points.append((0.5, t))
        for t in np.linspace(0.5, 0, n_seg + 1):
            k_points.append((t, t))
        
        k_points_unique = []
        for kp in k_points:
            if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
                k_points_unique.append(kp)
        
        min_omega5 = np.inf
        max_omega4 = -np.inf
        best_gap = -np.inf
        best_k = None
        best_vec4 = None
        best_vec5 = None
        
        m_vals = g_indices[:, 0]
        n_vals = g_indices[:, 1]
        
        for (kx, ky) in k_points_unique:
            k_mag_sq = (m_vals + kx)**2 + (n_vals + ky)**2
            M = S * k_mag_sq[None, :]
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                if len(eigvals) >= 5:
                    w4 = np.sqrt(np.real(eigvals[3]))
                    w5 = np.sqrt(np.real(eigvals[4]))
                    max_omega4 = max(max_omega4, w4)
                    min_omega5 = min(min_omega5, w5)
                    gap = w5 - w4
                    if gap > best_gap:
                        best_gap = gap
                        best_k = (kx, ky)
                        if return_eigvecs:
                            best_vec4 = eigvecs[:, 3]
                            best_vec5 = eigvecs[:, 4]
            except Exception:
                continue
        
        if min_omega5 == np.inf or max_omega4 == -np.inf:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        if min_omega5 + max_omega4 <= 0:
            if return_eigvecs:
                return -1.0, None, None, None
            return -1.0
        
        score = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        if return_eigvecs:
            return score, best_k, best_vec4, best_vec5
        return score
    
    def enforce_d4_symmetry(cell):
        """Enforce D4 symmetry via majority vote over all 8 symmetry operations."""
        cell = np.array(cell, dtype=float)
        symmetries = [
            cell,
            cell[::-1, :],
            cell[:, ::-1],
            cell[::-1, ::-1],
            cell.T,
            cell.T[::-1, :],
            cell.T[:, ::-1],
            cell.T[::-1, ::-1],
        ]
        avg = np.sum(symmetries, axis=0) / 8.0
        return (avg >= 0.5).astype(int)
    
    def make_cross_cell(arm_width, cross_thick, corner_round=0.0):
        """
        Create a D4-symmetric cross-shaped dielectric cell.
        
        Parameters:
        - arm_width: normalized half-width of the cross arms (0 to 0.5)
        - cross_thick: normalized thickness of the central cross region
        - corner_round: rounding factor for the corners (0 = sharp, 1 = fully rounded)
        
        The cross is centered at the cell center and extends to the cell edges.
        """
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='ij')
        
        # Horizontal arm: |Y| < arm_width/2
        # Vertical arm: |X| < arm_width/2
        h_arm = np.abs(Y) < (arm_width / 2.0)
        v_arm = np.abs(X) < (arm_width / 2.0)
        
        # Central thick region: |X| < cross_thick/2 AND |Y| < cross_thick/2
        central = (np.abs(X) < (cross_thick / 2.0)) & (np.abs(Y) < (cross_thick / 2.0))
        
        # Base cross
        cell = (h_arm | v_arm).astype(float)
        
        # Add central thickening
        cell[central] = 1.0
        
        # Corner rounding: erode the corners of the arms
        if corner_round > 0:
            # Distance from each pixel to the nearest arm centerline
            # For horizontal arm: distance = |Y|, for vertical: |X|
            # At corners (where both arms meet), we can round
            # Use a superellipse-like erosion
            dx = np.abs(X)
            dy = np.abs(Y)
            
            # The "corner" regions are where both |X| and |Y| are between
            # arm_width/2 and some max. We erode based on Chebyshev distance.
            # Simple approach: erode pixels far from center in diagonal directions
            diag_dist = np.maximum(dx, dy)
            arm_half = arm_width / 2.0
            # In the corner regions (beyond arm width in one direction),
            # erode by corner_round * (distance beyond arm edge)
            beyond_x = np.maximum(0, dx - arm_half)
            beyond_y = np.maximum(0, dy - arm_half)
            
            # Round the outer corners of the arms
            corner_mask = (beyond_x > 0) | (beyond_y > 0)
            if np.any(corner_mask):
                # Chebyshev distance to the arm edge
                cheb = np.maximum(beyond_x, beyond_y)
                # Erode: if cheb > corner_round * max_cheb, set to 0
                max_cheb = np.max(cheb[corner_mask]) if np.any(corner_mask) else 1.0
                if max_cheb > 0:
                    erode_mask = cheb > corner_round * max_cheb
                    cell[erode_mask & corner_mask] = 0.0
        
        cell = (cell >= 0.5).astype(int)
        cell = enforce_d4_symmetry(cell)
        return cell
    
    def compute_field_intensity(vec, kx, ky, g_indices):
        """Compute |E|^2 on pixel grid from plane-wave eigenvector."""
        m_vals = g_indices[:, 0].astype(float)
        n_vals = g_indices[:, 1].astype(float)
        coords = (np.arange(size) + 0.5) / size - 0.5
        xx, yy = np.meshgrid(coords, coords, indexing='ij')
        
        intensity = np.zeros((size, size), dtype=float)
        m_k = m_vals + kx
        n_k = n_vals + ky
        
        chunk_size = max(1, 500000 // (size * size))
        for start in range(0, len(g_indices), chunk_size):
            end = min(start + chunk_size, len(g_indices))
            m_chunk = m_k[start:end]
            n_chunk = n_k[start:end]
            c_chunk = vec[start:end]
            phase = 2.0 * np.pi * (np.outer(m_chunk, xx.ravel()) + np.outer(n_chunk, yy.ravel()))
            phase = phase.reshape(end - start, size, size)
            e_contrib = np.sum(c_chunk[:, None, None] * np.exp(1j * phase), axis=0)
            intensity += np.abs(e_contrib)**2
        return intensity
    
    # === Phase 1: Coarse parameter search over cross design space ===
    best_score = -np.inf
    best_cell = None
    
    # Coarse grid
    arm_widths = np.linspace(0.10, 0.35, 8)
    cross_thicks = np.linspace(0.10, 0.40, 7)
    corner_rounds = [0.0, 0.3, 0.6, 1.0]
    
    for aw in arm_widths:
        if time.time() > deadline - 80:
            break
        for ct in cross_thicks:
            if time.time() > deadline - 70:
                break
            for cr in corner_rounds:
                if time.time() > deadline - 65:
                    break
                try:
                    cell = make_cross_cell(aw, ct, cr)
                    score = compute_tm_bands(cell, n_pw=11, n_kpts=8)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
                except Exception:
                    continue
    
    # Also try a few four-quadrant rod designs as alternatives
    if time.time() < deadline - 60:
        for r in [0.10, 0.13, 0.16]:
            for exp in [2.0, 3.0, 4.0]:
                try:
                    cell = np.zeros((size, size), dtype=int)
                    coords = (np.arange(size) + 0.5) / size - 0.5
                    X, Y = np.meshgrid(coords, coords, indexing='ij')
                    for cx, cy in [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]:
                        val = np.abs((X - cx)/r)**exp + np.abs((Y - cy)/r)**exp
                        cell[val <= 1.0 + 1e-9] = 1
                    cell = enforce_d4_symmetry(cell)
                    score = compute_tm_bands(cell, n_pw=11, n_kpts=8)
                    if score > best_score:
                        best_score = score
                        best_cell = cell
                except Exception:
                    continue
    
    if best_cell is None:
        best_cell = make_cross_cell(0.20, 0.25, 0.0)
        best_score = compute_tm_bands(best_cell, n_pw=11, n_kpts=8)
    
    # === Phase 2: Fine-tune around best cross parameters ===
    if time.time() < deadline - 50:
        # Re-evaluate with higher accuracy
        best_score = compute_tm_bands(best_cell, n_pw=13, n_kpts=12)
        
        # Local refinement: try nearby parameter values
        # We need to figure out what parameters produced best_cell.
        # Instead, do a local pixel-level search around the best cell.
        # Try small perturbations: flip small clusters of pixels.
        current_cell = best_cell.copy()
        current_score = best_score
        
        # Try adding/removing pixels at strategic locations
        for _ in range(15):
            if time.time() > deadline - 40:
                break
            improved = False
            # Try flipping each of the 4 symmetric positions in the first quadrant
            # to add or remove material
            for i in range(size // 4):
                for j in range(size // 4):
                    if time.time() > deadline - 40:
                        break
                    # Try adding
                    new_cell = current_cell.copy()
                    new_cell[i, j] = 1
                    new_cell = enforce_d4_symmetry(new_cell)
                    if np.array_equal(new_cell, current_cell):
                        continue
                    try:
                        s = compute_tm_bands(new_cell, n_pw=12, n_kpts=10)
                        if s > current_score:
                            current_cell = new_cell
                            current_score = s
                            best_cell = new_cell
                            best_score = s
                            improved = True
                            break
                    except Exception:
                        pass
                if improved:
                    break
            if not improved:
                break
    
    # === Phase 3: Topological optimization guided by field intensities ===
    if time.time() < deadline - 25:
        current_cell = best_cell.copy()
        current_score = best_score
        g_indices_13 = get_g_cached(13)
        
        stall_count = 0
        max_stall = 6
        
        for it in range(25):
            if time.time() > deadline - 20:
                break
            
            score, best_k, vec4, vec5 = compute_tm_bands(current_cell, n_pw=13, n_kpts=12, return_eigvecs=True)
            if best_k is None:
                break
            
            kx, ky = best_k
            I4 = compute_field_intensity(vec4, kx, ky, g_indices_13)
            I5 = compute_field_intensity(vec5, kx, ky, g_indices_13)
            I4 = I4 / (np.sum(I4) + 1e-12)
            I5 = I5 / (np.sum(I5) + 1e-12)
            
            mask_add = (current_cell == 0)
            mask_rem = (current_cell == 1)
            gain_add = np.where(mask_add, I4 - I5, -np.inf)
            gain_rem = np.where(mask_rem, I5 - I4, -np.inf)
            
            improved = False
            
            # Try top 5 add candidates
            flat_gain = gain_add.ravel()
            valid_add = np.where(np.isfinite(flat_gain))[0]
            if len(valid_add) > 0:
                top_add = np.argpartition(flat_gain[valid_add], -5)[-5:]
                top_add = valid_add[top_add]
                for idx in top_add:
                    if time.time() > deadline - 20:
                        break
                    pos = np.unravel_index(idx, gain_add.shape)
                    new_cell = current_cell.copy()
                    new_cell[pos] = 1
                    new_cell = enforce_d4_symmetry(new_cell)
                    if np.array_equal(new_cell, current_cell):
                        continue
                    try:
                        s = compute_tm_bands(new_cell, n_pw=13, n_kpts=12)
                        if s > current_score:
                            current_cell = new_cell
                            current_score = s
                            best_cell = current_cell
                            best_score = current_score
                            improved = True
                            break
                    except Exception:
                        pass
            
            if not improved:
                # Try top 5 remove candidates
                flat_rem = gain_rem.ravel()
                valid_rem = np.where(np.isfinite(flat_rem))[0]
                if len(valid_rem) > 0:
                    top_rem = np.argpartition(flat_rem[valid_rem], -5)[-5:]
                    top_rem = valid_rem[top_rem]
                    for idx in top_rem:
                        if time.time() > deadline - 20:
                            break
                        pos = np.unravel_index(idx, gain_rem.shape)
                        new_cell = current_cell.copy()
                        new_cell[pos] = 0
                        new_cell = enforce_d4_symmetry(new_cell)
                        if np.array_equal(new_cell, current_cell):
                            continue
                        try:
                            s = compute_tm_bands(new_cell, n_pw=13, n_kpts=12)
                            if s > current_score:
                                current_cell = new_cell
                                current_score = s
                                best_cell = current_cell
                                best_score = current_score
                                improved = True
                                break
                        except Exception:
                            pass
            
            if improved:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= max_stall:
                    break
    
    # Final verification
    if time.time() < deadline - 10:
        final_score = compute_tm_bands(best_cell, n_pw=15, n_kpts=15)
        if final_score > best_score:
            best_score = final_score
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
