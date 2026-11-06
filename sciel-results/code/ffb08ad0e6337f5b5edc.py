"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.linalg import eigh
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 photonic crystal cell maximizing TM band gap 4-5."""
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210  # safety margin before 230s kill
    
    def time_left():
        return deadline - time.time()
    
    # Symmetry: D4 group on 32x32 cell
    # Independent region: top-left 16x16 (indices 0..15)
    # cell[i][j] determined by fundamental[min(i,31-i)][min(j,31-j)]
    # with the constraint that fundamental is symmetric: fund[i][j] = fund[j][i]
    
    def fund_to_cell(fund):
        """Expand 16x16 symmetric fundamental to 32x32 cell."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                fi = min(i, 31-i)
                fj = min(j, 31-j)
                cell[i][j] = fund[fi][fj]
        return cell
    
    # Plane wave setup
    # Use circular cutoff for G vectors
    def get_G_vectors(n_cutoff):
        """Get G vectors with |G| <= 2*pi*n_cutoff."""
        G_list = []
        for n in range(-n_cutoff, n_cutoff+1):
            for m in range(-n_cutoff, n_cutoff+1):
                if n*n + m*m <= n_cutoff*n_cutoff:
                    G_list.append((n, m))
        return G_list
    
    # Key k-points in irreducible BZ (0 <= kx <= ky <= pi)
    K_POINTS = [
        (0.0, 0.0),        # Gamma
        (np.pi, 0.0),      # X  
        (np.pi, np.pi),    # M
        (np.pi/2, 0.0),
        (np.pi/2, np.pi/2),
        (3*np.pi/4, np.pi/4),
    ]
    
    # Precompute pixel positions
    xs = np.array([(i+0.5)/size for i in range(size)])
    ys = np.array([(j+0.5)/size for j in range(size)])
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    def compute_tm_gap(cell, n_pw=4):
        """Compute TM gap-midgap ratio between bands 4 and 5."""
        G_list = get_G_vectors(n_pw)
        N = len(G_list)
        
        # Epsilon at each pixel
        eps_pix = np.where(np.array(cell, dtype=int) == 1, eps_high, eps_low)
        
        # Fourier coefficients: eps_G = (1/N) sum eps(x,y) exp(-i G·(x,y))
        eps_F = np.zeros(N, dtype=complex)
        for idx, (n, m) in enumerate(G_list):
            Gx = 2*np.pi*n
            Gy = 2*np.pi*m
            phase = np.exp(-1j*(Gx*X + Gy*Y))
            eps_F[idx] = np.sum(eps_pix * phase) / (size*size)
        
        # Build epsilon correlation matrix
        G_dict = {(n,m): idx for idx, (n,m) in enumerate(G_list)}
        eps_mat = np.zeros((N, N))
        for p in range(N):
            for q in range(N):
                dn = G_list[p][0] - G_list[q][0]
                dm = G_list[p][1] - G_list[q][1]
                key = (dn, dm)
                if key in G_dict:
                    eps_mat[p, q] = eps_F[G_dict[key]].real
                else:
                    eps_mat[p, q] = 0.0
        
        # Ensure symmetry
        eps_mat = (eps_mat + eps_mat.T) / 2
        
        # Compute bands at each k-point
        band4_max = 0
        band5_min = float('inf')
        
        for kx, ky in K_POINTS:
            k_vecs = np.array([(kx + 2*np.pi*G_list[p][0], ky + 2*np.pi*G_list[p][1]) for p in range(N)])
            k_sq = k_vecs[:,0]**2 + k_vecs[:,1]**2
            
            M = np.diag(k_sq)
            N_mat = eps_mat.copy()
            
            try:
                eigvals = eigh(M, N_mat, eigvals_only=True)
                eigvals = np.sort(eigvals)
            except:
                continue
            
            omegas = np.sqrt(np.maximum(eigvals, 0))
            
            if len(omegas) >= 5:
                b4 = omegas[3]  # 4th band (0-indexed)
                b5 = omegas[4]  # 5th band
                band4_max = max(band4_max, b4)
                band5_min = min(band5_min, b5)
        
        if band5_min == float('inf'):
            return -1.0
        
        gap = band5_min - band4_max
        midgap = (band5_min + band4_max) / 2
        if midgap <= 0:
            return -1.0
        return 2 * gap / midgap
    
    # Generate initial candidates
    def make_rod(radius):
        fund = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - i
                dy = 15.5 - j
                dist = np.sqrt(dx*dx + dy*dy) / 16.0
                if dist < radius:
                    fund[i][j] = 1
        return fund
    
    def make_square(half):
        fund = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                if abs(15.5 - i) < half and abs(15.5 - j) < half:
                    fund[i][j] = 1
        return fund
    
    def make_cross(w, l):
        fund = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - i
                dy = 15.5 - j
                if (abs(dy) < w/2.0 and abs(dx) < l/2.0) or \
                   (abs(dx) < w/2.0 and abs(dy) < l/2.0):
                    fund[i][j] = 1
        return fund
    
    def make_ring(r_inner, r_outer):
        fund = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - i
                dy = 15.5 - j
                dist = np.sqrt(dx*dx + dy*dy) / 16.0
                if r_inner < dist < r_outer:
                    fund[i][j] = 1
        return fund
    
    candidates = []
    # Rods
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        candidates.append(make_rod(r))
    # Squares
    for h in [3, 4, 5, 6, 7]:
        candidates.append(make_square(h))
    # Crosses
    for w in [2, 3, 4, 5]:
        for l in [6, 8, 10, 12]:
            candidates.append(make_cross(w, l))
    # Rings
    for ri, ro in [(0.1, 0.3), (0.2, 0.4), (0.15, 0.35), (0.25, 0.45)]:
        candidates.append(make_ring(ri, ro))
    
    # Evaluate candidates
    best_score = -999
    best_cell = None
    best_fund = None
    
    for fund in candidates:
        if time_left() < 5:
            break
        cell = fund_to_cell(fund)
        score = compute_tm_gap(cell, n_pw=4)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_fund = [row[:] for row in fund]
    
    if best_cell is None:
        # Fallback
        fund = make_rod(0.2)
        return fund_to_cell(fund)
    
    # Local search
    def local_search(start_fund, max_steps=100):
        current = [row[:] for row in start_fund]
        current_cell = fund_to_cell(current)
        current_score = compute_tm_gap(current_cell, n_pw=4)
        if current_score is None:
            return start_fund, -999
        
        best_fund_local = [row[:] for row in current]
        best_score_local = current_score
        no_improve = 0
        
        for step in range(max_steps):
            if time_left() < 30:
                break
            
            fi = np.random.randint(0, 16)
            fj = np.random.randint(0, 16)
            
            # Ensure symmetry of fundamental: fund[i][j] = fund[j][i]
            # So flipping (fi,fj) also flips (fj,fi) if different
            flip_positions = {(fi, fj)}
            if fi != fj:
                flip_positions.add((fj, fi))
            
            old_vals = {}
            for (pi, pj) in flip_positions:
                old_vals[(pi, pj)] = current[pi][pj]
                current[pi][pj] = 1 - current[pi][pj]
            
            new_cell = fund_to_cell(current)
            new_score = compute_tm_gap(new_cell, n_pw=4)
            
            if new_score is not None and new_score > current_score:
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_fund_local = [row[:] for row in current]
                no_improve = 0
            else:
                for (pi, pj) in flip_positions:
                    current[pi][pj] = old_vals[(pi, pj)]
                no_improve += 1
                if no_improve > 20:
                    break
        
        return best_fund_local, best_score_local
    
    # Multiple restarts
    for restart in range(5):
        if time_left() < 60:
            break
        
        if restart == 0:
            start_fund = [row[:] for row in best_fund]
        else:
            start_fund = [row[:] for row in best_fund]
            for _ in range(3 + restart):
                fi = np.random.randint(0, 16)
                fj = np.random.randint(0, 16)
                flip_positions = {(fi, fj)}
                if fi != fj:
                    flip_positions.add((fj, fi))
                for (pi, pj) in flip_positions:
                    start_fund[pi][pj] = 1 - start_fund[pi][pj]
        
        found_fund, found_score = local_search(start_fund, max_steps=80)
        if found_score > best_score:
            best_score = found_score
            best_fund = found_fund
            best_cell = fund_to_cell(best_fund)
    
    return best_cell
# EVOLVE-BLOCK-END
