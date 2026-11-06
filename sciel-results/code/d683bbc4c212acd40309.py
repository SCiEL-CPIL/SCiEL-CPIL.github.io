import numpy as np
import time
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 115.0
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # 1. Precompute G vectors and Fourier coeffs
    G_max = 8
    G_vectors = []
    for gx in range(-G_max, G_max + 1):
        for gy in range(-G_max, G_max + 1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    N_pw = len(G_vectors)
    
    G_max_diff = 2 * G_max
    G_vectors_diff = []
    for gx in range(-G_max_diff, G_max_diff + 1):
        for gy in range(-G_max_diff, G_max_diff + 1):
            G_vectors_diff.append((gx, gy))
    G_vectors_diff = np.array(G_vectors_diff)
    N_pw_diff = len(G_vectors_diff)
    
    pixel_coords = []
    for i in range(size):
        for j in range(size):
            pixel_coords.append((i/size, (i+1)/size, j/size, (j+1)/size))
            
    C_coeffs_diff = np.zeros((size*size, N_pw_diff), dtype=complex)
    for p_idx, (x0, x1, y0, y1) in enumerate(pixel_coords):
        for k_idx, (Kx, Ky) in enumerate(G_vectors_diff):
            if Kx == 0 and Ky == 0:
                C_coeffs_diff[p_idx, k_idx] = 1.0
            else:
                term_x = 1.0
                if Kx != 0:
                    term_x = (np.exp(-1j * Kx * x1) - np.exp(-1j * Kx * x0)) / (-1j * Kx)
                term_y = 1.0
                if Ky != 0:
                    term_y = (np.exp(-1j * Ky * y1) - np.exp(-1j * Ky * y0)) / (-1j * Ky)
                C_coeffs_diff[p_idx, k_idx] = term_x * term_y
                
    def get_diff_idx(gx, gy):
        return (gx + G_max_diff) * (2*G_max_diff + 1) + (gy + G_max_diff)
        
    diff_indices = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            gx_diff = G_vectors[i, 0] - G_vectors[j, 0]
            gy_diff = G_vectors[i, 1] - G_vectors[j, 1]
            diff_indices[i, j] = get_diff_idx(gx_diff, gy_diff)
            
    idx_00 = get_diff_idx(0, 0)
    
    k_points = np.array([
        (0.0, 0.0), (0.5, 0.0), (0.5, 0.5),
        (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0),
        (0.6, 0.0), (0.7, 0.0), (0.8, 0.0), (0.9, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4),
        (0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.5, 0.9),
        (0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4),
        (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9),
        (0.25, 0.1), (0.3, 0.2), (0.4, 0.3), (0.45, 0.45)
    ])
    
    from scipy.linalg import eigh
    
    def evaluate_mask(mask):
        cell = np.zeros((size, size), dtype=int)
        idx = 0
        for i in range(16):
            for j in range(i + 1):
                val = (mask >> idx) & 1
                cell[i, j] = val
                cell[j, i] = val
                cell[31-i, 31-j] = val
                cell[31-j, 31-i] = val
                idx += 1
                
        mask_vec = cell.flatten()
        F_diel = C_coeffs_diff.T @ mask_vec
        
        inv_eps_diff = -(11.0/12.0) * F_diel
        inv_eps_diff[idx_00] += 1.0
        
        B = inv_eps_diff[diff_indices]
        
        omega_4s = []
        omega_5s = []
        
        for kx, ky in k_points:
            G_k = G_vectors + np.array([kx, ky])
            A_diag = np.sum(G_k**2, axis=1)
            A = np.diag(A_diag)
            
            try:
                eigvals, _ = eigh(A, B)
                top_lambdas = eigvals[-5:]
                top_k = 1.0 / np.sqrt(top_lambdas)
                omega_4s.append(top_k[1])
                omega_5s.append(top_k[0])
            except:
                return -1.0

        if not omega_4s:
            return -1.0
            
        max_omega_4 = max(omega_4s)
        min_omega_5 = min(omega_5s)
        gap = min_omega_5 - max_omega_4
        mid = (min_omega_5 + max_omega_4) / 2.0
        return 2 * gap / mid if mid != 0 else 0.0

    best_score = -1.0
    best_mask = 0
    
    seeds = [0, (1<<136)-1]
    random.seed(42)
    for _ in range(10):
        seeds.append(random.getrandbits(136))
        
    for seed in seeds:
        current_mask = seed
        current_score = evaluate_mask(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_neighbor_mask = current_mask
            best_neighbor_score = current_score
            
            for bit in range(136):
                if time.time() > deadline:
                    break
                neighbor_mask = current_mask ^ (1 << bit)
                score = evaluate_mask(neighbor_mask)
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor_mask = neighbor_mask
                    improved = True
                    
            if improved:
                current_mask = best_neighbor_mask
                current_score = best_neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    
    final_cell = np.zeros((size, size), dtype=int)
    idx = 0
    for i in range(16):
        for j in range(i + 1):
            val = (best_mask >> idx) & 1
            final_cell[i, j] = val
            final_cell[j, i] = val
            final_cell[31-i, 31-j] = val
            final_cell[31-j, 31-i] = val
            idx += 1
            
    return final_cell.tolist()
