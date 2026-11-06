import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    R_pw = 7
    G_vecs = []
    for n in range(-R_pw, R_pw+1):
        for m in range(-R_pw, R_pw+1):
            if n*n + m*m <= R_pw*R_pw:
                G_vecs.append((n, m))
    G_vecs = np.array(G_vecs, dtype=float)
    N_pw = len(G_vecs)
    
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.1, 0.1),
        (0.4, 0.4),
    ]

    def compute_gap(cell):
        eps_inv = np.zeros((size, size))
        eps_inv[cell == 1] = 1.0 / eps_high
        eps_inv[cell == 0] = 1.0 / eps_low
        
        R_diff = 2 * R_pw
        G_diff_vecs = []
        for n in range(-R_diff, R_diff+1):
            for m in range(-R_diff, R_diff+1):
                if n*n + m*m <= R_diff*R_diff:
                    G_diff_vecs.append((n, m))
        G_diff_vecs = np.array(G_diff_vecs, dtype=float)
        N_diff = len(G_diff_vecs)
        
        diff_to_idx = {}
        for idx, (n, m) in enumerate(G_diff_vecs):
            diff_to_idx[(int(n), int(m))] = idx
            
        i_idx = np.arange(size)
        r_i = (i_idx + 0.5) / size
        r_j = (i_idx + 0.5) / size
        
        n_g = G_diff_vecs[:, 0]
        m_g = G_diff_vecs[:, 1]
        
        phase_i = np.exp(-2j * np.pi * np.outer(n_g, r_i))
        phase_j = np.exp(-2j * np.pi * np.outer(m_g, r_j))
        
        A = phase_i @ eps_inv
        eps_inv_G = np.sum(A * phase_j, axis=1)
        
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = G_vecs + k
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            M = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                for j in range(N_pw):
                    dn = int(G_vecs[i, 0] - G_vecs[j, 0])
                    dm = int(G_vecs[i, 1] - G_vecs[j, 1])
                    idx_diff = diff_to_idx.get((dn, dm), -1)
                    if idx_diff != -1:
                        M[i, j] = k_plus_G_sq[i] * eps_inv_G[idx_diff]
                    else:
                        M[i, j] = 0.0
                        
            try:
                eigvals = np.linalg.eigvals(M)
                # Filter real, positive eigenvalues
                real_eigvals = np.real(eigvals)
                positive_eigvals = real_eigvals[real_eigvals > 0]
                if len(positive_eigvals) >= 5:
                    positive_eigvals.sort()
                    w4 = np.sqrt(positive_eigvals[3])
                    w5 = np.sqrt(positive_eigvals[4])
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
            except Exception:
                pass
        
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    def generate_cell(r_center, r_corner):
        cell = [[0]*size for _ in range(size)]
        cx = 15.5
        cy = 15.5
        
        for i in range(size):
            for j in range(size):
                pi = i + 0.5
                pj = j + 0.5
                is_dielectric = False
                
                if r_center > 0:
                    dx = pi - cx
                    dy = pj - cy
                    if dx*dx + dy*dy <= r_center*r_center:
                        is_dielectric = True
                
                if r_corner > 0:
                    d1_sq = pi*pi + pj*pj
                    d2_sq = (pi-32)**2 + pj*pj
                    d3_sq = pi*pi + (pj-32)**2
                    d4_sq = (pi-32)**2 + (pj-32)**2
                    min_d_sq = min(d1_sq, d2_sq, d3_sq, d4_sq)
                    if min_d_sq <= r_corner*r_corner:
                        is_dielectric = True
                
                cell[i][j] = 1 if is_dielectric else 0
        return cell

    best_score = -np.inf
    best_params = (0, 0)
    
    np.random.seed(42)
    for _ in range(100):
        rc = np.random.uniform(0, 16)
        rk = np.random.uniform(0, 16)
        cell = generate_cell(rc, rk)
        score = compute_gap(cell)
        if score > best_score:
            best_score = score
            best_params = (rc, rk)
            
    current_rc, current_rk = best_params
    step = 1.0
    
    while time.time() < deadline:
        improved = False
        for _ in range(10):
            d_rc = np.random.uniform(-step, step)
            d_rk = np.random.uniform(-step, step)
            new_rc = np.clip(current_rc + d_rc, 0, 16)
            new_rk = np.clip(current_rk + d_rk, 0, 16)
            cell = generate_cell(new_rc, new_rk)
            score = compute_gap(cell)
            if score > best_score:
                best_score = score
                best_params = (new_rc, new_rk)
                current_rc, current_rk = new_rc, new_rk
                improved = True
        if not improved:
            step *= 0.8
            if step < 0.1:
                break
                
    final_cell = generate_cell(best_params[0], best_params[1])
    return final_cell
