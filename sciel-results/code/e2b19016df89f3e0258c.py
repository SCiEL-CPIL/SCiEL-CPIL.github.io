"""Photonic crystal design: maximize TM band-4/5 gap via symmetric PWE optimization."""
import time
import numpy as np
from scipy.linalg import eigh

_DEADLINE = 230.0
_START = time.time()

def _remaining():
    return _DEADLINE - time.time()

def _symmetric_cell(base):
    """Apply 4-fold rotation + inversion symmetry to a 32x32 base grid."""
    n = 32
    c = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            v = base[i, j]
            c[i, j] = v
            c[j, i] = max(c[j, i], v)
            c[n - 1 - i, n - 1 - j] = max(c[n - 1 - i, n - 1 - j], v)
            c[n - 1 - j, n - 1 - i] = max(c[n - 1 - j, n - 1 - i], v)
    return c

def _compute_gap_fast(cell, eps_low=1.0, eps_high=12.0):
    """Fast PWE band-structure computation for TM mode. Returns gap-midgap ratio."""
    n = 32
    eps = np.where(cell == 1, eps_high, eps_low)
    
    # Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    N = 17  # 17x17 = 289 plane waves
    G = np.zeros((N, N, 2), dtype=int)
    idx = 0
    for gx in range(-8, 9):
        for gy in range(-8, 9):
            if gx * gx + gy * gy <= 64:
                G[idx, 0] = gx
                G[idx, 1] = gy
                idx += 1
    nPW = idx
    
    # Compute Fourier coefficients of 1/eps
    # 1/eps(G) = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i 2pi (Gx*i/N + Gy*j/N))
    inv_eps_2d = inv_eps.reshape(n, n)
    # Use FFT for efficiency
    inv_eps_padded = np.zeros((64, 64), dtype=complex)
    inv_eps_padded[:n, :n] = inv_eps_2d
    fft_2d = np.fft.fft2(inv_eps_padded)
    # Rescale: the DFT gives sum over 64x64, but we need sum over 32x32
    # Actually fft2 computes sum_{i=0}^{63} sum_{j=0}^{63} f[i,j] exp(-2pi i (kx*i/64 + ky*j/64))
    # We need: (1/32^2) sum_{i=0}^{31} sum_{j=0}^{31} f[i,j] exp(-2pi i (kx*i/32 + ky*j/32))
    # Since f is zero outside 32x32, the 64x64 FFT at (kx,ky) where kx,ky in 0..63
    # gives sum_{i=0}^{31} sum_{j=0}^{31} f[i,j] exp(-2pi i (kx*i/64 + ky*j/64))
    # For kx,ky in 0..31: this equals what we want with Gx=kx, Gy=ky (scaled by 1/32^2)
    # For kx in 32..63: corresponds to negative Gx
    fft_coeffs = fft_2d / (n * n)
    
    def get_coeff(gx, gy):
        """Get Fourier coefficient for G = (gx, gy)."""
        kx = gx % 64
        ky = gy % 64
        return fft_coeffs[kx, ky]
    
    # Build PWE matrix for a given wavevector k
    def solve_band(kx, ky):
        kx2 = kx * kx + ky * ky
        A = np.zeros((nPW, nPW), dtype=float)
        for i in range(nPW):
            for j in range(nPW):
                dgx = G[i, 0] - G[j, 0]
                dgy = G[i, 1] - G[j, 1]
                if dgx == 0 and dgy == 0:
                    A[i, j] = (kx + G[i, 0]) ** 2 + (ky + G[i, 1]) ** 2
                else:
                    A[i, j] = get_coeff(dgx, dgy).real
                    # The imaginary part doesn't contribute to the symmetric matrix
                    # Actually for TM: the matrix element is <G+k|1/eps|G'+k>
                    # which is the Fourier coefficient of 1/eps at G-G'
                    # This should be real for a real eps field
                    A[i, j] = float(np.real(get_coeff(dgx, dgy)))
        
        # Solve eigenvalue problem
        try:
            eigvals, _ = eigh(A)
            return eigvals
        except:
            return None
    
    # Irreducible Brillouin zone: kx >= ky >= 0, kx + ky <= 1 (for square lattice)
    # Sample 30 wavevectors
    kvecs = []
    # Along Gamma-X: (t, 0) for t in [0, 1]
    for t in np.linspace(0, 1, 6):
        kvecs.append((t, 0.0))
    # Along Gamma-M: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        kvecs.append((t, t))
    # Along X-M: (1-t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        kvecs.append((1.0 - t, t))
    # Interior points
    for t in np.linspace(0.1, 0.4, 4):
        for s in np.linspace(0.05, t - 0.05, 3):
            if t + s <= 1.0:
                kvecs.append((t, s))
    
    # Keep only 30
    kvecs = kvecs[:30]
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for kx, ky in kvecs:
        eigvals = solve_band(kx, ky)
        if eigvals is None:
            continue
        # TM bands: omega^2 = eigenvalues, so omega = sqrt(eigenvalue)
        # Band 4 = 4th smallest, band 5 = 5th smallest
        if len(eigvals) >= 5:
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            max_omega4 = max(max_omega4, w4)
            min_omega5 = min(min_omega5, w5)
    
    if min_omega5 == float('inf'):
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap < 1e-10:
        return -1.0
    
    return 2.0 * gap / midgap

def _generate_seed(size=32, radius_frac=0.3):
    """Generate a symmetric dielectric rod seed."""
    base = np.zeros((size, size), dtype=int)
    cx = cy = (size - 1) / 2.0
    r = radius_frac * size
    for i in range(size):
        for j in range(size):
            dx = i - cx
            dy = j - cy
            if dx * dx + dy * dy <= r * r:
                base[i, j] = 1
    return _symmetric_cell(base)

def _flip_pixel(base, i, j):
    """Flip a pixel in the base grid (before symmetry)."""
    base[i, j] = 1 - base[i, j]

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Start with a dielectric rod seed
    best_base = _generate_seed(size, 0.25)
    best_score = _compute_gap_fast(best_base, eps_low, eps_high)
    
    if _remaining() < 30:
        return best_base.tolist()
    
    # Local search: flip individual pixels in the base (top-left quadrant region)
    # The base is 32x32 but due to symmetry we only need to consider a fundamental region
    # For 4-fold + inversion symmetry, the fundamental domain is roughly the top-left 8x8 region
    # but let's just search over all positions in the first quadrant
    
    # Define search positions (unique pixels under symmetry)
    # A pixel (i,j) is equivalent to (j,i), (31-i,31-j), (31-j,31-i)
    # So we only need to consider i <= j and i + j <= 31 (roughly)
    search_positions = []
    for i in range(size):
        for j in range(i, size):  # i <= j for transpose symmetry
            # Check inversion: (31-i, 31-j) should map to something with i' >= j'
            # We'll just include all (i,j) with i <= j and let symmetry handle the rest
            # But to avoid redundancy, we need i+j <= 31 for the inversion to not create a new orbit
            # Actually, let's just use i <= j as the canonical representative
            search_positions.append((i, j))
    
    # Shuffle for diversity
    np.random.seed(42)
    np.random.shuffle(search_positions)
    
    # Hill-climbing with occasional random restarts
    current_base = best_base.copy()
    current_score = best_score
    no_improve = 0
    max_no_improve = 50
    
    step = 0
    while _remaining() > 40 and no_improve < max_no_improve:
        step += 1
        # Try a random pixel flip
        idx = step % len(search_positions)
        i, j = search_positions[idx]
        
        # Flip
        old_val = current_base[i, j]
        current_base[i, j] = 1 - old_val
        
        # Evaluate
        full_cell = _symmetric_cell(current_base)
        score = _compute_gap_fast(full_cell, eps_low, eps_high)
        
        if score > current_score:
            current_score = score
            no_improve = 0
            if score > best_score:
                best_score = score
                best_base = current_base.copy()
        else:
            # Revert
            current_base[i, j] = old_val
            no_improve += 1
        
        # Occasionally try a random restart
        if no_improve % 30 == 0 and _remaining() > 60:
            # Try a small random perturbation of the best
            trial = best_base.copy()
            for _ in range(3):
                ri = np.random.randint(0, size)
                rj = np.random.randint(0, size)
                trial[ri, rj] = 1 - trial[ri, rj]
            trial_cell = _symmetric_cell(trial)
            trial_score = _compute_gap_fast(trial_cell, eps_low, eps_high)
            if trial_score > current_score:
                current_base = trial
                current_score = trial_score
                no_improve = 0
            else:
                no_improve += 1
    
    # Final verification
    final_cell = _symmetric_cell(best_base)
    final_score = _compute_gap_fast(final_cell, eps_low, eps_high)
    
    if final_score > best_score:
        best_base = final_cell.copy()
    
    # Ensure the result is a proper nested list of 0/1
    result = _symmetric_cell(best_base)
    return result.tolist()
