"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def _enforce_symmetry(cell):
    """Enforce 4-fold symmetry about the cell centre."""
    size = len(cell)
    c = np.array(cell, dtype=float)
    # Enforce: c[i,j] == c[j,i] (transpose) and c[i,j] == c[size-1-i,j] (vertical flip)
    c = 0.5 * (c + c.T)
    c = 0.5 * (c + c[::-1, :])
    c = (c > 0.5).astype(int)
    return c.tolist()


def _fourier_coeff(eps_inv, mx, my, size):
    """Compute Fourier coefficient eps_inv(G) for G = 2*pi*(mx,my)/a, a=1."""
    # eps_inv(G) = (1/size^2) * sum_{i,j} eps_inv[i,j] * exp(-i*2*pi*(mx*i/size + my*j/size))
    total = 0.0
    for i in range(size):
        for j in range(size):
            total += eps_inv[i, j] * np.exp(-1j * 2 * np.pi * (mx * i / size + my * j / size))
    return total / (size * size)


def _compute_bands(cell, n_pw=17, eps_high=12.0):
    """
    Compute TM band structure and return the gap-midgap ratio between bands 4 and 5.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps = eps * (eps_high - 1.0) + 1.0
    eps_inv = 1.0 / eps
    
    n_half = n_pw // 2
    mx_vals = list(range(-n_half, n_half + 1))
    my_vals = list(range(-n_half, n_half + 1))
    
    # Precompute all Fourier coefficients of eps_inv
    fourier_cache = {}
    for mx in mx_vals:
        for my in my_vals:
            fourier_cache[(mx, my)] = _fourier_coeff(eps_inv, mx, my, size)
    
    pw_indices = [(mx, my) for mx in mx_vals for my in my_vals]
    n_pw_total = len(pw_indices)
    
    # k-points in irreducible BZ: 0 <= ky <= kx <= pi
    k_points = []
    for kx in np.linspace(0, np.pi, 8):
        for ky in np.linspace(0, kx, 8):
            k_points.append((float(kx), float(ky)))
    # Ensure we have reasonable coverage
    k_points = list(set([(round(kx, 6), round(ky, 6)) for kx, ky in k_points]))
    if len(k_points) > 30:
        idx = np.linspace(0, len(k_points) - 1, 30, dtype=int)
        k_points = [k_points[i] for i in idx]
    
    band4_max = 0.0
    band5_min = float('inf')
    
    for kx, ky in k_points:
        k = np.array([kx, ky])
        
        # Build the matrix M[m,n] = eps_inv(G_m - G_n) * |k + G_n|^2
        M = np.zeros((n_pw_total, n_pw_total), dtype=complex)
        for m_idx, (mx_m, my_m) in enumerate(pw_indices):
            for n_idx, (mx_n, my_n) in enumerate(pw_indices):
                dmx = mx_m - mx_n
                dmy = my_m - my_n
                coeff = fourier_cache.get((dmx, dmy), 0.0)
                Gn = 2 * np.pi * np.array([mx_n, my_n])
                k_plus_Gn = k + Gn
                k_plus_Gn_sq = float(np.dot(k_plus_Gn, k_plus_Gn))
                M[m_idx, n_idx] = coeff * k_plus_Gn_sq
        
        M_real = np.real(M)
        # Ensure symmetry
        M_real = 0.5 * (M_real + M_real.T)
        
        try:
            eigvals = eigh(M_real, subset_by_index=[3, 4])
            if isinstance(eigvals, tuple):
                eigvals = eigvals[0]
            omega4 = np.sqrt(max(eigvals[0], 0))
            omega5 = np.sqrt(max(eigvals[1], 0))
        except Exception:
            eigvals, _ = eigh(M_real)
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
        
        band4_max = max(band4_max, omega4)
        band5_min = min(band5_min, omega5)
    
    if band5_min == float('inf') or band4_max == 0:
        return -1.0
    
    return 2 * (band5_min - band4_max) / (band5_min + band4_max)


def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell for maximum TM band gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial: centered circle radius 0.2
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.04:
                cell[i, j] = 1
    cell = np.array(_enforce_symmetry(cell.tolist()), dtype=int)
    
    best_score = _compute_bands(cell, n_pw=17, eps_high=eps_high)
    best_cell = cell.copy()
    
    np.random.seed(42)
    
    # Local search on boundary pixels
    def get_boundary(c):
        b = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < size and 0 <= nj < size:
                        if c[i,j] != c[ni,nj]:
                            b[i,j] = True
                            break
        return b
    
    boundary = get_boundary(cell)
    bidx = np.argwhere(boundary)
    
    for it in range(800):
        if time.time() > deadline:
            break
        if len(bidx) == 0:
            break
        idx = np.random.choice(len(bidx))
        i, j = bidx[idx]
        cell[i, j] = 1 - cell[i, j]
        c_sym = np.array(_enforce_symmetry(cell.tolist()), dtype=int)
        s = _compute_bands(c_sym, n_pw=17, eps_high=eps_high)
        if s > best_score:
            best_score = s
            best_cell = c_sym.copy()
            cell = c_sym.copy()
            boundary = get_boundary(cell)
            bidx = np.argwhere(boundary)
        else:
            cell = best_cell.copy()
    
    # Try a few global shapes
    for radius in [0.18, 0.22, 0.28, 0.32]:
        if time.time() > deadline:
            break
        c2 = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < radius*radius:
                    c2[i,j] = 1
        c2 = np.array(_enforce_symmetry(c2.tolist()), dtype=int)
        s2 = _compute_bands(c2, n_pw=17, eps_high=eps_high)
        if s2 > best_score:
            best_score = s2
            best_cell = c2
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
