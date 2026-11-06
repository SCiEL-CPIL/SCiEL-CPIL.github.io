"""Rewritten program: parametric multi-rod TM band-gap search with fast PWE evaluator."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    half = size // 2  # 16

    t0 = time.time()
    deadline = t0 + 210.0  # safety margin

    # --- Symmetric cell construction ---
    # D4 symmetry: cell[i][j] determined by upper-triangle of top-left 16x16 quadrant
    # Independent pixels: (i,j) with 0<=i<=15, i<=j<=15 (136 pixels)
    
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant."""
        N = size
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                # D4 orbit: 8 symmetric positions
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    # --- Fast PWE evaluator ---
    # Precompute G vectors for 17x17 = 289 plane waves (|mx|,|my| <= 8)
    max_g = 8
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)  # 289
    G_vecs = np.array([(2.0*math.pi*mx, 2.0*math.pi*my) for mx, my in g_indices])
    
    # K-points: 30 points in irreducible BZ triangle (0,0)-(pi/2,pi/2)-(0,pi)
    k_points = []
    n_line = 5
    # Gamma-X: (0, t) for t in [0, pi]
    for t in np.linspace(0, math.pi, n_line):
        k_points.append((0.0, float(t)))
    # Gamma-M: (t, t) for t in [0, pi/2]
    for t in np.linspace(0, math.pi/2, n_line):
        k_points.append((float(t), float(t)))
    # M-X: (pi/2 - t, pi/2 + t) for t in [0, pi/2]
    for t in np.linspace(0, math.pi/2, n_line):
        k_points.append((math.pi/2 - float(t), math.pi/2 + float(t)))
    # Interior points
    k_points.append((math.pi/6, math.pi/3))
    k_points.append((math.pi/4, math.pi/2))
    k_points.append((math.pi/8, 3*math.pi/8))
    k_points.append((math.pi/5, 2*math.pi/5))
    k_points.append((math.pi/3, 2*math.pi/3))
    k_points.append((math.pi/7, 3*math.pi/7))
    k_points.append((math.pi/12, math.pi/3))
    k_points.append((math.pi/6, math.pi/2))
    k_points.append((math.pi/10, 3*math.pi/10))
    k_points = k_points[:30]
    
    # Precompute |k+G|^2 for each k-point and each G
    kG_sq_all = np.zeros((len(k_points), num_g))
    for ki, (kx, ky) in enumerate(k_points):
        for q in range(num_g):
            kG_sq_all[ki, q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2

    def compute_eps_inv_matrix(cell):
        """Compute eps_inv Fourier coefficient matrix for the cell."""
        N = size
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0/eps_high if cell[i][j] == 1 else 1.0
        
        # 2D FFT for Fourier coefficients
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        
        # Build eps_inv matrix: M[p,q] = eps_inv(G_p - G_q)
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                mx_diff = mx_p - mx_q
                my_diff = my_p - my_q
                mx_w = mx_diff % N
                my_w = my_diff % N
                eps_inv_mat[p, q] = fft_coeff[my_w, mx_w]
        
        return eps_inv_mat

    def compute_tm_bands(cell):
        """Compute TM bands and return gap-midgap ratio for bands 4 and 5."""
        eps_inv_mat = compute_eps_inv_matrix(cell)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for ki in range(len(k_points)):
            kG_sq = kG_sq_all[ki]
            # M[p,q] = eps_inv_mat[p,q] * |k+G_q|^2
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(float(eigvals[3]), 0.0))
            omega5 = math.sqrt(max(float(eigvals[4]), 0.0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Parametric design: multi-rod with D4 symmetry ---
    # Each rod is defined by center (cx, cy) in [0, 0.5]x[0, 0.5] (in cell coordinates)
    # and radius r. Due to D4 symmetry, a rod at (cx,cy) generates 8 copies.
    # We parameterize rods in the fundamental domain (upper triangle of top-left quadrant).
    
    def rods_to_quadrant(rods):
        """Convert list of (cx, cy, r) rods to 16x16 quadrant."""
        q = [[0]*half for _ in range(half)]
        for cx, cy, r in rods:
            # cx, cy in [0, 0.5] cell coordinates
            # Pixel (i,j) in quadrant: center at ((j+0.5)/32, (i+0.5)/32)
            for i in range(half):
                for j in range(i, half):
                    px = (j + 0.5) / size
                    py = (i + 0.5) / size
                    # Check distance to rod center (and its D4 mirrors)
                    # Rod at (cx, cy): distance
                    d = math.sqrt((px-cx)**2 + (py-cy)**2)
                    if d <= r:
                        q[i][j] = 1
        return q

    # --- Initial designs: try various multi-rod configurations ---
    def make_initial_rods():
        """Generate diverse initial rod configurations."""
        designs = []
        
        # Single ring (annulus approximated by ring of rods)
        # Actually, let's use filled disks at various positions
        
        # Design 1: Single central disk
        designs.append([(0.25, 0.25, 0.12)])
        
        # Design 2: Two rods
        designs.append([(0.15, 0.15, 0.08), (0.35, 0.35, 0.08)])
        
        # Design 3: Three rods in triangle
        designs.append([(0.1, 0.1, 0.07), (0.3, 0.1, 0.07), (0.2, 0.3, 0.07)])
        
        # Design 4: Four rods (corners of quadrant)
        designs.append([(0.08, 0.08, 0.06), (0.42, 0.08, 0.06), 
                        (0.08, 0.42, 0.06), (0.42, 0.42, 0.06)])
        
        # Design 5: Ring approximation with 8 rods
        ring_rods = []
        n_ring = 8
        ring_r = 0.2
        rod_r = 0.05
        for k in range(n_ring):
            angle = 2*math.pi*k/n_ring
            cx = 0.25 + ring_r*math.cos(angle)
            cy = 0.25 + ring_r*math.sin(angle)
            if 0 <= cx <= 0.5 and 0 <= cy <= 0.5:
                ring_rods.append((cx, cy, rod_r))
        if ring_rods:
            designs.append(ring_rods)
        
        # Design 6: Concentric pattern
        designs.append([(0.25, 0.25, 0.06), (0.15, 0.15, 0.04), (0.35, 0.35, 0.04)])
        
        # Design 7: Cross pattern
        designs.append([(0.25, 0.1, 0.04), (0.25, 0.4, 0.04),
                       (0.1, 0.25, 0.04), (0.4, 0.25, 0.04)])
        
        # Design 8: Dense grid
        grid_rods = []
        for gx in [0.1, 0.2, 0.3, 0.4]:
            for gy in [0.1, 0.2, 0.3, 0.4]:
                grid_rods.append((gx, gy, 0.035))
        designs.append(grid_rods)
        
        # Design 9: Sparse large rods
        designs.append([(0.12, 0.12, 0.1), (0.38, 0.38, 0.1)])
        
        # Design 10: Alternating pattern
        designs.append([(0.05, 0.25, 0.05), (0.25, 0.05, 0.05),
                       (0.25, 0.45, 0.05), (0.45, 0.25, 0.05)])
        
        return designs

    # Evaluate initial designs
    initial_rods = make_initial_rods()
    best_score = -1.0
    best_rods = initial_rods[0]
    
    for rods in initial_rods:
        if time.time() > deadline - 60:
            break
        q = rods_to_quadrant(rods)
        cell = build_cell_from_quadrant(q)
        score = compute_tm_bands(cell)
        if score > best_score:
            best_score = score
            best_rods = [list(r) for r in rods]
    
    # --- Local search: perturb rod parameters ---
    # Each rod: (cx, cy, r) -> 3 params per rod
    # Search: random perturbation of params, keep if improves
    
    current_rods = [list(r) for r in best_rods]
    current_score = best_score
    no_improve = 0
    max_no_improve = 80
    
    n_params = len(current_rods) * 3
    
    # Also try adding/removing rods occasionally
    for iteration in range(300):
        if time.time() > deadline - 30:
            break
        if no_improve > max_no_improve:
            break
        
        # Randomly select a modification
        mod_type = random.random()
        
        if mod_type < 0.7 and len(current_rods) > 0:
            # Perturb an existing rod
            rod_idx = random.randint(0, len(current_rods)-1)
            param_idx = random.randint(0, 2)  # cx, cy, or r
            old_val = current_rods[rod_idx][param_idx]
            
            # Perturbation size
            if param_idx == 2:  # radius
                delta = random.uniform(-0.02, 0.02)
                new_val = max(0.02, min(0.2, old_val + delta))
            else:  # cx or cy
                delta = random.uniform(-0.05, 0.05)
                new_val = max(0.02, min(0.48, old_val + delta))
            
            current_rods[rod_idx][param_idx] = new_val
            
        elif mod_type < 0.85 and len(current_rods) < 12:
            # Add a new rod
            cx = random.uniform(0.05, 0.45)
            cy = random.uniform(0.05, 0.45)
            r = random.uniform(0.03, 0.08)
            current_rods.append([cx, cy, r])
            # Mark for potential removal if not improving
            added_idx = len(current_rods) - 1
            
        elif mod_type < 0.95 and len(current_rods) > 2:
            # Remove a random rod
            remove_idx = random.randint(0, len(current_rods)-1)
            current_rods.pop(remove_idx)
        
        else:
            # Large random restart within bounds
            new_rods = []
            n_new = random.randint(2, 6)
            for _ in range(n_new):
                cx = random.uniform(0.05, 0.45)
                cy = random.uniform(0.05, 0.45)
                r = random.uniform(0.04, 0.1)
                new_rods.append([cx, cy, r])
            current_rods = new_rods
        
        # Evaluate
        q = rods_to_quadrant(current_rods)
        cell = build_cell_from_quadrant(q)
        score = compute_tm_bands(cell)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_rods = [list(r) for r in current_rods]
            no_improve = 0
        else:
            no_improve += 1
            # Revert if we added a rod and it didn't help
            if mod_type >= 0.7 and mod_type < 0.85 and len(current_rods) > 0:
                # Try to remove the last added rod if score worsened
                pass  # Keep simple: just revert by not updating
    
    # --- Final refinement: fine-tune best rods ---
    if best_score > -0.5 and time.time() < deadline - 20:
        # Fine perturbation of best design
        fine_rods = [list(r) for r in best_rods]
        fine_score = best_score
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            rod_idx = random.randint(0, len(fine_rods)-1)
            param_idx = random.randint(0, 2)
            old_val = fine_rods[rod_idx][param_idx]
            
            if param_idx == 2:
                delta = random.uniform(-0.005, 0.005)
                new_val = max(0.02, min(0.2, old_val + delta))
            else:
                delta = random.uniform(-0.01, 0.01)
                new_val = max(0.02, min(0.48, old_val + delta))
            
            fine_rods[rod_idx][param_idx] = new_val
            q = rods_to_quadrant(fine_rods)
            cell = build_cell_from_quadrant(q)
            score = compute_tm_bands(cell)
            
            if score > fine_score:
                fine_score = score
                if score > best_score:
                    best_score = score
                    best_rods = [list(r) for r in fine_rods]
            else:
                fine_rods[rod_idx][param_idx] = old_val
    
    # Build final cell from best rods
    q = rods_to_quadrant(best_rods)
    final_cell = build_cell_from_quadrant(q)
    
    # Verify and enforce symmetry
    for i in range(size):
        for j in range(size):
            if final_cell[i][j] != final_cell[size-1-i][j]:
                final_cell[size-1-i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][size-1-j]:
                final_cell[i][size-1-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
    
    return final_cell
# EVOLVE-BLOCK-END
