"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 210.0  # Safety margin before 230s limit

    half = size // 2  # 16
    N = size  # 32

    # --- Symmetric cell construction from 16x16 upper-triangle quadrant ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant (i<=j)."""
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

    # --- PWE solver with smaller basis for speed ---
    max_g = 5  # 11x11 = 121 plane waves (faster than 289)
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)
    G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
    
    # Fewer k-points: focus on critical points
    k_points = [
        (0.0, math.pi),           # X
        (0.0, 2*math.pi/3),       
        (0.0, math.pi/2),         # midpoint Gamma-X
        (math.pi/2, math.pi/2),   # M
        (math.pi/3, math.pi/3),
        (math.pi/6, math.pi/2),
        (0.0, math.pi*0.8),
        (0.0, math.pi*0.6),
        (math.pi/4, math.pi/2),
        (math.pi/6, math.pi/3),
    ]

    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        return fft_coeff

    def evaluate_gap(cell, fft_coeff=None):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        if fft_coeff is None:
            fft_coeff = compute_eps_inv_fourier(cell)
        
        # Build eps_inv difference matrix
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            
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

    # --- Initial designs: rings are known to work well for TM gaps ---
    def make_ring_quadrant(r_inner, r_outer):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    # Start with a few ring designs
    initial_designs = []
    for ri, ro in [(0.15, 0.35), (0.2, 0.4), (0.25, 0.45), (0.1, 0.3)]:
        initial_designs.append(make_ring_quadrant(ri, ro))

    # Evaluate initial designs
    best_score = -1.0
    best_quadrant = initial_designs[0]
    
    for q in initial_designs:
        if time.time() > deadline - 100:
            break
        cell = build_cell_from_quadrant(q)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]

    # --- Fast local search with pixel flipping ---
    current_quadrant = [row[:] for row in best_quadrant]
    current_score = best_score
    current_cell = build_cell_from_quadrant(current_quadrant)
    current_fft = compute_eps_inv_fourier(current_cell)
    
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    rng = random.Random(12345)
    
    # Multiple passes of local search
    for sweep in range(20):
        if time.time() > deadline - 30:
            break
        
        shuffled_pos = positions[:]
        rng.shuffle(shuffled_pos)
        
        improved_this_sweep = False
        for (i, j) in shuffled_pos:
            if time.time() > deadline - 25:
                break
            
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_fft = compute_eps_inv_fourier(new_cell)
            new_score = evaluate_gap(new_cell, new_fft)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                current_fft = new_fft
                improved_this_sweep = True
                if new_score > best_score:
                    best_score = new_score
                    best_quadrant = [row[:] for row in current_quadrant]
            else:
                current_quadrant[i][j] = old_val
        
        if not improved_this_sweep and sweep > 5:
            # Try a few random flips to escape local minima
            for _ in range(5):
                if time.time() > deadline - 25:
                    break
                i, j = rng.choice(positions)
                current_quadrant[i][j] = 1 - current_quadrant[i][j]
                new_cell = build_cell_from_quadrant(current_quadrant)
                new_fft = compute_eps_inv_fourier(new_cell)
                new_score = evaluate_gap(new_cell, new_fft)
                if new_score > current_score:
                    current_score = new_score
                    current_cell = new_cell
                    current_fft = new_fft
                    if new_score > best_score:
                        best_score = new_score
                        best_quadrant = [row[:] for row in current_quadrant]
                else:
                    current_quadrant[i][j] = 1 - current_quadrant[i][j]

    # Build final cell
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
