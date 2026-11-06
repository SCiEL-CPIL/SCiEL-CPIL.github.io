"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 210.0  # 210s of 230s
    
    N = size
    half = N // 2  # 16
    
    # --- Symmetric cell construction ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant."""
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell
    
    # --- Fast band-structure evaluator ---
    # Precompute G vector indices for 17x17 = 289 plane waves
    max_g = 8
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)
    
    # Precompute G vector components
    G_x = np.array([2*math.pi*mx for mx, my in g_indices])
    G_y = np.array([2*math.pi*my for mx, my in g_indices])
    
    # Precompute index mapping for fast coefficient lookup
    # g_indices is ordered as mx from -8..8, my from -8..8
    # Index: mx_idx = mx + 8, my_idx = my + 8
    # In the flattened array: idx = mx_idx * 17 + my_idx
    
    def compute_eps_inv_coeffs(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0/eps_high if cell[i][j] == 1 else 1.0
        
        fft_coeff = np.fft.fft2(eps_inv) / (N*N)
        
        # Pre-extract needed coefficients into a lookup table
        # We need eps_inv(G_p - G_q) for all p,q
        # G_p - G_q has mx in -16..16, my in -16..16
        # But due to periodicity, we can wrap to -8..8 range
        coeff_table = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                coeff_table[p, q] = fft_coeff[dn, dm]
        
        return coeff_table
    
    def compute_tm_bands_fast(coeff_table):
        """Compute TM bands using precomputed coefficient table."""
        # K-points in irreducible BZ
        k_points = []
        n_per_line = 5
        for t in np.linspace(0, math.pi, n_per_line):
            k_points.append((0.0, t))
        for t in np.linspace(0, math.pi/2, n_per_line):
            k_points.append((t, t))
        for t in np.linspace(0, math.pi/2, n_per_line):
            k_points.append((math.pi/2 - t, math.pi/2 + t))
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        k_points.append((math.pi/8, 3*math.pi/8))
        k_points.append((math.pi/5, 2*math.pi/5))
        k_points.append((math.pi/3, 2*math.pi/3))
        k_points = k_points[:30]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # |k + G|^2 for each G
            kG_sq = (kx + G_x)**2 + (ky + G_y)**2
            
            # M[p,q] = coeff_table[p,q] * kG_sq[q]
            M = coeff_table * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # --- Design families ---
    def make_ring_quadrant(r_inner, r_outer):
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q
    
    def make_disk_quadrant(radius):
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r <= radius:
                    q[i][j] = 1
        return q
    
    def make_hole_ring_quadrant(r_inner, r_outer, hole_r):
        """Ring with a central hole."""
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer and r > hole_r:
                    q[i][j] = 1
        return q
    
    def make_square_ring_quadrant(r_inner, r_outer):
        """Square ring (Manhattan distance)."""
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = max(abs(x), abs(y))
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q
    
    def make_cross_quadrant(width):
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                if x < width or y < width:
                    q[i][j] = 1
        return q
    
    def make_ellipse_quadrant(rx, ry):
        """Elliptical ring."""
        half = size // 2
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt((x/rx)**2 + (y/ry)**2)
                if 0.8 <= r <= 1.2:
                    q[i][j] = 1
        return q
    
    # --- Evaluate a design quickly ---
    def score_quadrant(q):
        cell = build_cell_from_quadrant(q)
        coeff_table = compute_eps_inv_coeffs(cell)
        return compute_tm_bands_fast(coeff_table)
    
    # --- Coarse parameter search ---
    best_score = -1.0
    best_quadrant = None
    
    # Generate diverse initial designs
    candidates = []
    
    # Rings
    for r_in in [0.1, 0.15, 0.2, 0.25]:
        for r_out in [0.3, 0.35, 0.4, 0.45, 0.5]:
            if r_in < r_out:
                candidates.append(make_ring_quadrant(r_in, r_out))
    
    # Disks
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(make_disk_quadrant(r))
    
    # Hole rings
    for r_in in [0.15, 0.2]:
        for r_out in [0.35, 0.4, 0.45]:
            for hole in [0.05, 0.1]:
                candidates.append(make_hole_ring_quadrant(r_in, r_out, hole))
    
    # Square rings
    for r_in in [0.15, 0.2]:
        for r_out in [0.35, 0.4, 0.45]:
            candidates.append(make_square_ring_quadrant(r_in, r_out))
    
    # Crosses
    for w in [0.08, 0.1, 0.12, 0.15]:
        candidates.append(make_cross_quadrant(w))
    
    # Ellipses
    for rx in [0.2, 0.25, 0.3]:
        for ry in [0.15, 0.2, 0.25]:
            if rx != ry:
                candidates.append(make_ellipse_quadrant(rx, ry))
    
    # Evaluate all candidates
    for q in candidates:
        if time.time() > deadline - 60:
            break
        s = score_quadrant(q)
        if s > best_score:
            best_score = s
            best_quadrant = [row[:] for row in q]
    
    # --- Local search: pixel flips ---
    if best_quadrant is not None:
        current_q = [row[:] for row in best_quadrant]
        current_score = best_score
        
        positions = [(i, j) for i in range(half) for j in range(i, half)]
        
        import random
        rng = random.Random(42)
        
        no_improve = 0
        max_no_improve = 80
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            rng.shuffle(positions)
            
            for (i, j) in positions:
                if time.time() > deadline - 10:
                    break
                
                old_val = current_q[i][j]
                current_q[i][j] = 1 - old_val
                
                cell = build_cell_from_quadrant(current_q)
                coeff_table = compute_eps_inv_coeffs(cell)
                s = compute_tm_bands_fast(coeff_table)
                
                if s > current_score:
                    current_score = s
                    best_score = s
                    best_quadrant = [row[:] for row in current_q]
                    improved = True
                    no_improve = 0
                else:
                    current_q[i][j] = old_val
                    no_improve += 1
            
            if not improved:
                no_improve += 20
    
    # --- Final refinement: try small perturbations around best ---
    if best_quadrant is not None and time.time() < deadline - 30:
        # Try flipping pairs of nearby pixels
        current_q = [row[:] for row in best_quadrant]
        current_score = best_score
        
        positions = [(i, j) for i in range(half) for j in range(i, half)]
        
        import random
        rng2 = random.Random(123)
        
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            
            # Pick a random position and flip it
            i, j = rng2.choice(positions)
            old_val = current_q[i][j]
            current_q[i][j] = 1 - old_val
            
            cell = build_cell_from_quadrant(current_q)
            coeff_table = compute_eps_inv_coeffs(cell)
            s = compute_tm_bands_fast(coeff_table)
            
            if s > current_score:
                current_score = s
                best_score = s
                best_quadrant = [row[:] for row in current_q]
            else:
                current_q[i][j] = old_val
    
    # Build final cell
    if best_quadrant is None:
        # Fallback: simple ring
        best_quadrant = make_ring_quadrant(0.15, 0.35)
    
    final_cell = build_cell_from_quadrant(best_quadrant)
    
    # Verify symmetry
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N-1-i][j]:
                final_cell[N-1-i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][N-1-j]:
                final_cell[i][N-1-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
    
    return final_cell
# EVOLVE-BLOCK-END
