import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # 1. Precompute Plane Waves
    nw = 289
    indices = []
    n_max = 20
    for n in range(-n_max, n_max+1):
        for m in range(-n_max, n_max+1):
            indices.append((n*n + m*m, n, m))
    indices.sort(key=lambda x: x[0])
    gw = indices[:nw]
    g_vecs = np.array([(n, m) for _, n, m in gw])
    
    g_to_idx = {}
    for idx, (n, m) in enumerate(g_vecs):
        g_to_idx[(n, m)] = idx
        
    # Precompute diff indices
    diff_indices = np.zeros((nw, nw), dtype=int)
    for i in range(nw):
        for j in range(nw):
            diff = (g_vecs[i][0] - g_vecs[j][0], g_vecs[i][1] - g_vecs[j][1])
            if diff in g_to_idx:
                diff_indices[i, j] = g_to_idx[diff]
            else:
                diff_indices[i, j] = -1

    # 2. Precompute K-points
    k_points = []
    k_points.append(np.array([0.0, 0.0]))
    k_points.append(np.array([np.pi, 0.0]))
    k_points.append(np.array([np.pi, np.pi]))
    # Add grid points
    for i in range(8):
        for j in range(8):
            kx = (i / 7.0) * np.pi
            ky = (j / 7.0) * np.pi
            if 0 <= ky <= kx <= np.pi:
                k_points.append(np.array([kx, ky]))
    # Add random points to reach 30
    while len(k_points) < 30:
        rx = np.random.rand() * np.pi
        ry = np.random.rand() * np.pi
        if 0 <= ry <= rx <= np.pi:
            k_points.append(np.array([rx, ry]))
    k_points = np.array(k_points[:30])

    # 3. Evaluate Function
    def evaluate(mask_32x32):
        # Compute Fourier coefficients
        eps_coeffs = np.zeros(nw, dtype=complex)
        ii, jj = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        for idx in range(nw):
            n, m = g_vecs[idx]
            if n == 0 and m == 0:
                eps_coeffs[idx] = eps_low + (eps_high - eps_low) * np.mean(mask_32x32)
            else:
                phase = -2j * np.pi * (n * ii + m * jj) / size
                coeff = np.sum(mask_32x32 * np.exp(phase)) / (size * size)
                eps_coeffs[idx] = eps_low + (eps_high - eps_low) * coeff
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for k in k_points:
            g_plus_k = 2 * np.pi * g_vecs + k
            g_plus_k_mags_sq = np.sum(g_plus_k**2, axis=1)
            sqrt_mags = np.sqrt(g_plus_k_mags_sq)
            
            M_vals = np.where(diff_indices >= 0, eps_coeffs[diff_indices], 0)
            M = M_vals * np.outer(sqrt_mags, sqrt_mags)
            
            eigs = np.linalg.eigvalsh(M)
            eigs = np.sort(eigs)
            omegas = np.sqrt(eigs)
            
            if len(omegas) >= 5:
                w4 = omegas[3]
                w5 = omegas[4]
                if w5 < min_omega_5:
                    min_omega_5 = w5
                if w4 > max_omega_4:
                    max_omega_4 = w4
        
        gap = min_omega_5 - max_omega_4
        if min_omega_5 + max_omega_4 == 0:
            return 0.0
        score = 2 * gap / (min_omega_5 + max_omega_4)
        return score

    # 4. Reconstruction
    def reconstruct(fund):
        cell = np.zeros((size, size), dtype=int)
        for i in range(16):
            for j in range(16):
                val = fund[i, j]
                cell[i, 16+j] = val
                cell[i, 15-j] = val
                cell[31-i, 16+j] = val
                cell[31-i, 15-j] = val
        return cell

    # 5. Optimization
    np.random.seed(42)
    fund = np.random.randint(0, 2, (16, 16))
    # Enforce initial symmetry
    for i in range(16):
        for j in range(16):
            fund[15-j, 15-i] = fund[i, j]
            
    best_mask = reconstruct(fund)
    best_score = evaluate(best_mask)
    current_mask = best_mask.copy()
    current_fund = fund.copy()
    
    start_time = time.time()
    timeout = 110.0
    
    while time.time() - start_time < timeout:
        # Pick random pixel
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        # Flip
        current_fund[i, j] = 1 - current_fund[i, j]
        # Enforce symmetry
        i2, j2 = 15-j, 15-i
        if i != i2 or j != j2:
            current_fund[i2, j2] = current_fund[i, j]
            
        new_mask = reconstruct(current_fund)
        new_score = evaluate(new_mask)
        
        if new_score > best_score:
            best_score = new_score
            best_mask = new_mask.copy()
        else:
            # Revert
            current_fund[i, j] = 1 - current_fund[i, j]
            if i != i2 or j != j2:
                current_fund[i2, j2] = current_fund[i, j]
            current_mask = reconstruct(current_fund)
            
    return best_mask.tolist()
