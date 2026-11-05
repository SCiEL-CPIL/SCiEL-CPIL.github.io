import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=12, pw_radius=12):
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    k_points = [
        (0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0),
        (0.5, 0.1), (0.5, 0.2), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5),
        (0.3, 0.3), (0.4, 0.2), (0.2, 0.2)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        inv_eps_mat = inv_eps_full[
            Gx_diff.astype(int) + offset_d, 
            Gy_diff.astype(int) + offset_d
        ]
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)


def _make_symmetric(cell):
    """Ensure cell is symmetric under transpose, row-reverse, col-reverse."""
    N = len(cell)
    c = np.array(cell, dtype=float)
    # Average over all 8 symmetry operations of the square group
    c_t = c.T
    c_r = c[::-1, :]
    c_c = c[:, ::-1]
    c_rt = c[::-1, ::-1]
    c_tr = c.T[::-1, :]  # transpose then row-reverse
    c_tc = c.T[:, ::-1]  # transpose then col-reverse
    c_rtc = c[::-1, ::-1].T  # 180 then transpose
    c_averaged = (c + c_t + c_r + c_c + c_rt + c_tr + c_tc + c_rtc) / 8.0
    c_sym = (c_averaged > 0.5).astype(int)
    return c_sym.tolist()


def _compute_eigenvectors(cell, k_point, pw_radius=10):
    """Compute eigenvalues and eigenvectors for a given k-point."""
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    size_d = 4 * pw_radius + 1
    offset_d = 2 * pw_radius
    inv_eps_full = np.zeros((size_d, size_d), dtype=complex)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j])
            dqy = int(Gy_diff[i, j])
            inv_eps_full[dqx + offset_d, dqy + offset_d] = fft_inv[dqx % N, dqy % N]
    
    kvec = np.array(k_point, dtype=float)
    kG = kvec + G_arr
    dots = kG @ kG.T
    
    inv_eps_mat = inv_eps_full[
        Gx_diff.astype(int) + offset_d, 
        Gy_diff.astype(int) + offset_d
    ]
    
    M = dots * inv_eps_mat
    M = (M + M.conj().T) / 2
    
    try:
        eigvals, eigvecs = eigh(M)
        eigvals = np.sqrt(np.maximum(eigvals, 0))
        return eigvals, eigvecs
    except:
        return None, None


def _field_energy_density(eigvec, G_arr, cell_size=32):
    """Compute spatial energy density from plane-wave coefficients."""
    n_pw = len(G_arr)
    # Inverse Fourier transform to get spatial profile
    # The field in real space is sum over G of c_G * exp(i(G+k)·r)
    # For energy density, we need |E(r)|^2
    # Approximate by taking the magnitude of the inverse FFT of the coefficients
    
    # Create a grid for the inverse transform
    # We'll use a discrete approximation
    N = cell_size
    field = np.zeros((N, N), dtype=complex)
    
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            for idx in range(n_pw):
                Gx, Gy = G_arr[idx]
                phase = 2 * np.pi * (Gx * x + Gy * y)
                field[i, j] += eigvec[idx] * np.exp(1j * phase)
    
    return np.abs(field) ** 2


def _spectral_inverse_design(cell, max_iter=50, pw_radius=10, learning_rate=0.5):
    """
    Spectral-domain inverse design:
    1. Compute band edges at critical k-points
    2. Extract field energy densities for bands 4 and 5
    3. Use the difference to guide pixel updates
    """
    N = len(cell)
    current = np.array(cell, dtype=float)
    
    # Critical k-points in the irreducible BZ
    k_critical = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25)]
    
    best_cell = current.copy()
    best_score = _compute_gap_fast(current.tolist(), n_k=8, pw_radius=pw_radius)
    
    for iteration in range(max_iter):
        if time.time() > _deadline - 30:
            break
            
        # Compute energy densities at each critical k-point
        total_diff = np.zeros((N, N), dtype=float)
        
        for k in k_critical:
            eigvals, eigvecs = _compute_eigenvectors(current.tolist(), k, pw_radius)
            if eigvals is None or len(eigvals) < 5:
                continue
                
            # Band 4 (index 3) and Band 5 (index 4)
            if eigvecs is not None:
                # Get the eigenvectors for bands 4 and 5
                vec4 = eigvecs[:, 3]
                vec5 = eigvecs[:, 4]
                
                # Compute field energy densities
                # Use a faster approximation
                N_pw = len(vec4)
                G_list = []
                for Gx in range(-pw_radius, pw_radius + 1):
                    for Gy in range(-pw_radius, pw_radius + 1):
                        if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                            G_list.append((Gx, Gy))
                G_arr = np.array(G_list, dtype=float)
                
                # Fast field computation using FFT
                # Reshape coefficients to a 2D grid
                grid_size = 2 * pw_radius + 1
                field4 = np.zeros((grid_size, grid_size), dtype=complex)
                field5 = np.zeros((grid_size, grid_size), dtype=complex)
                
                offset = pw_radius
                for idx, (Gx, Gy) in enumerate(G_list):
                    field4[Gx + offset, Gy + offset] = vec4[idx]
                    field5[Gx + offset, Gy + offset] = vec5[idx]
                
                # Inverse FFT to get real-space fields
                # Pad to appropriate size
                field4_real = np.fft.ifft2(field4, s=(N, N)).real
                field5_real = np.fft.ifft2(field5, s=(N, N)).real
                
                energy4 = field4_real ** 2
                energy5 = field5_real ** 2
                
                # Normalize
                if np.sum(energy4) > 0:
                    energy4 /= np.sum(energy4)
                if np.sum(energy5) > 0:
                    energy5 /= np.sum(energy5)
                
                total_diff += (energy5 - energy4)
        
        # Normalize the difference
        total_diff /= len(k_critical)
        
        # Update pixels based on the energy difference
        # Where energy5 > energy4, we want to increase dielectric (1)
        # Where energy4 > energy5, we want to decrease dielectric (0)
        
        # Threshold-based update with symmetry preservation
        update_mask = total_diff > 0  # Increase dielectric here
        decrease_mask = total_diff < 0  # Decrease dielectric here
        
        # Apply updates in the fundamental domain and mirror
        # Fundamental domain: i <= j, i < N//2
        for i in range(N // 2):
            for j in range(i, N // 2):
                if update_mask[i, j] and current[i, j] == 0:
                    # Flip to 1 and mirror
                    current[i, j] = 1
                    current[i, N-1-j] = 1
                    current[N-1-i, j] = 1
                    current[N-1-i, N-1-j] = 1
                    current[j, i] = 1
                    current[j, N-1-i] = 1
                    current[N-1-j, i] = 1
                    current[N-1-j, N-1-i] = 1
                elif decrease_mask[i, j] and current[i, j] == 1:
                    # Flip to 0 and mirror
                    current[i, j] = 0
                    current[i, N-1-j] = 0
                    current[N-1-i, j] = 0
                    current[N-1-i, N-1-j] = 0
                    current[j, i] = 0
                    current[j, N-1-i] = 0
                    current[N-1-j, i] = 0
                    current[N-1-j, N-1-i] = 0
        
        # Evaluate the new cell
        new_score = _compute_gap_fast(current.tolist(), n_k=8, pw_radius=pw_radius)
        
        if new_score > best_score:
            best_score = new_score
            best_cell = current.copy()
        else:
            # Revert if no improvement
            current = best_cell.copy()
    
    return best_cell.astype(int).tolist(), best_score


def _gen_rod(r):
    """Circular dielectric rod of radius r (in units of cell size)."""
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < r*r:
                cell[i][j] = 1
    return cell


def _gen_random_symmetric(density):
    """Random symmetric pixel pattern respecting full square symmetry."""
    size = 32
    cell = np.zeros((size, size), dtype=int)
    for i in range(size//2):
        for j in range(i, size//2):
            if np.random.random() < density:
                cell[i, j] = 1
                cell[i, size-1-j] = 1
                cell[size-1-i, j] = 1
                cell[size-1-i, size-1-j] = 1
                cell[j, i] = 1
                cell[j, size-1-i] = 1
                cell[size-1-j, i] = 1
                cell[size-1-j, size-1-i] = 1
    return cell.tolist()


_deadline = 0

def design(problem):
    global _deadline
    start_time = time.time()
    _deadline = start_time + 195.0  # Safety margin (230s limit, leave 35s buffer)
    
    size = problem["pixels"]
    assert size == 32
    
    # Initialize with several diverse starting points
    initial_cells = [
        _gen_rod(0.15),
        _gen_rod(0.20),
        _gen_rod(0.25),
        _gen_random_symmetric(0.2),
        _gen_random_symmetric(0.3),
        _gen_random_symmetric(0.4),
    ]
    
    # Quick screen to pick the best starting point
    best_cell = None
    best_score = -1.0
    
    for cell in initial_cells:
        if time.time() > _deadline - 100:
            break
        try:
            score = _compute_gap_fast(cell, n_k=8, pw_radius=10)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    if best_cell is None:
        best_cell = _gen_rod(0.20)
        best_score = _compute_gap_fast(best_cell, n_k=8, pw_radius=10)
    
    # Run spectral inverse design from the best starting point
    if time.time() < _deadline - 60:
        optimized_cell, opt_score = _spectral_inverse_design(
            best_cell, 
            max_iter=40, 
            pw_radius=10,
            learning_rate=0.5
        )
        if opt_score > best_score:
            best_cell = optimized_cell
            best_score = opt_score
    
    # Try a second round with a different starting point if time allows
    if time.time() < _deadline - 40:
        alt_cell = _gen_random_symmetric(0.25)
        alt_score = _compute_gap_fast(alt_cell, n_k=8, pw_radius=10)
        if alt_score > best_score - 0.1:  # Only if close
            optimized_alt, opt_alt_score = _spectral_inverse_design(
                alt_cell, 
                max_iter=30, 
                pw_radius=10
            )
            if opt_alt_score > best_score:
                best_cell = optimized_alt
                best_score = opt_alt_score
    
    # Final verification
    if best_cell is not None:
        final_score = _compute_gap_fast(best_cell, n_k=12, pw_radius=12)
    
    return best_cell
