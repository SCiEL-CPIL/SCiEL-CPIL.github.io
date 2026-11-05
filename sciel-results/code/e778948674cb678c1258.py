import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    half_size = size // 2 # 16
    
    def bits_to_cell(bits):
        """
        Convert a list of 136 bits (upper triangle of 16x16 top-left quadrant) 
        to a full 32x32 symmetric cell.
        """
        tl = [[0] * half_size for _ in range(half_size)]
        idx = 0
        for i in range(half_size):
            for j in range(i + 1):
                val = bits[idx]
                tl[i][j] = val
                tl[j][i] = val
                idx += 1
        
        cell = [[0] * size for _ in range(size)]
        for i in range(half_size):
            for j in range(half_size):
                val = tl[i][j]
                cell[i][j] = val
                cell[i][size - 1 - j] = val
                cell[size - 1 - i][j] = val
                cell[size - 1 - i][size - 1 - j] = val
        return cell

    def get_fourier_coeffs(cell, n_g):
        N = 2 * n_g + 1
        coeffs = np.zeros((N, N), dtype=complex)
        
        i_vals = np.arange(size)
        I, J = np.meshgrid(i_vals, i_vals, indexing='ij')
        M = np.array(cell, dtype=float)
        
        for n in range(-n_g, n_g + 1):
            for m in range(-n_g, n_g + 1):
                phase = np.exp(-1j * 2 * np.pi * (n * I + m * J) / size)
                val = np.sum(M * phase) / (size * size)
                coeffs[n + n_g, m + n_g] = val
        
        delta_eps = eps_high - eps_low
        eps_coeffs = coeffs * delta_eps
        eps_coeffs[n_g, n_g] += eps_low
        return eps_coeffs

    def compute_gap(cell, n_g, n_k):
        eps_coeffs = get_fourier_coeffs(cell, n_g)
        N_G = 2 * n_g + 1
        G_vecs = []
        for nx in range(-n_g, n_g + 1):
            for ny in range(-n_g, n_g + 1):
                G_vecs.append((2 * np.pi * nx, 2 * np.pi * ny))
        
        k_points = []
        if n_k > 1:
            for i in range(n_k):
                for j in range(n_k):
                    kx = i / (n_k - 1) * np.pi
                    ky = j / (n_k - 1) * np.pi
                    if kx <= ky:
                        k_points.append((kx, ky))
        else:
            k_points = [(0, 0)]
            
        # Pre-compute E indices and values
        E_indices = np.zeros((N_G * N_G, N_G * N_G), dtype=int)
        eps_flat = eps_coeffs.flatten()
        
        for i in range(N_G):
            for j in range(N_G):
                idx = i * N_G + j
                for ip in range(N_G):
                    for jp in range(N_G):
                        idxp = ip * N_G + jp
                        dx = i - ip
                        dy = j - jp
                        if -n_g <= dx <= n_g and -n_g <= dy <= n_g:
                            E_indices[idx, idxp] = (dx + n_g) * N_G + (dy + n_g)
                        else:
                            E_indices[idx, idxp] = -1
                            
        # Mask valid indices
        valid_mask = E_indices != -1
        
        all_omegas = []
        
        for kx, ky in k_points:
            D = np.zeros(N_G * N_G)
            for i in range(N_G):
                for j in range(N_G):
                    idx = i * N_G + j
                    kx_g = kx + G_vecs[i][0]
                    ky_g = ky + G_vecs[i][1]
                    D[idx] = kx_g * kx_g + ky_g * ky_g
            
            E = np.zeros((N_G * N_G, N_G * N_G), dtype=complex)
            # Fill E using indices
            # This is a bit slow in Python loop, but N_G is small for surrogate
            for i in range(N_G * N_G):
                for j in range(N_G * N_G):
                    if valid_mask[i, j]:
                        E[i, j] = eps_flat[E_indices[i, j]]
                        
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                mu_valid = mu_real[mu_real > 1e-10]
                
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio

    # Phase 1: Initial Candidates
    candidates = []
    candidates.append([0] * 136)
    candidates.append([1] * 136)
    for _ in range(5):
        candidates.append([np.random.randint(0, 2) for _ in range(136)])
        
    # Add a structured guess: Dielectric cross
    bits_cross = [0] * 136
    idx = 0
    for i in range(half_size):
        for j in range(i + 1):
            # Center is 15,15. Cross along diagonals?
            # Let's try: 1 if i==15 or j==15 (in quadrant coords)
            # But quadrant is 0..15. Center of cell is 15.5.
            # So i=15, j=15 is the corner closest to center.
            bits_cross[idx] = 1 if (i == 15 or j == 15) else 0
            idx += 1
    candidates.append(bits_cross)

    # Phase 2: Surrogate Search
    best_bits = None
    best_score = -1.0
    
    scored = []
    for bits in candidates:
        cell = bits_to_cell(bits)
        score = compute_gap(cell, n_g=3, n_k=6)
        scored.append((score, bits))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    
    for score, bits in scored[:2]:
        if time.time() > deadline - 30:
            break
        current_bits = bits[:]
        current_score = score
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(136):
                if time.time() > deadline - 30:
                    break
                new_bits = current_bits[:]
                new_bits[i] = 1 - new_bits[i]
                new_cell = bits_to_cell(new_bits)
                new_score = compute_gap(new_cell, n_g=3, n_k=6)
                if new_score > current_score:
                    current_bits = new_bits
                    current_score = new_score
                    improved = True
        if current_score > best_score:
            best_score = current_score
            best_bits = current_bits

    if best_bits is None:
        best_bits = scored[0][1]

    # Phase 3: Exact Evaluation
    best_cell = bits_to_cell(best_bits)
    if time.time() < deadline - 5:
        # We can try to refine with exact method if time permits, 
        # but surrogate is usually good enough for ranking.
        # Let's just return the best surrogate cell, evaluated exactly by the evaluator.
        pass
        
    return best_cell
