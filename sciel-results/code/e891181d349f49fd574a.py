"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.linalg import eigh

def design(problem):
    """Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = 200.0  # seconds, with safety margin
    
    # We'll use a parametric approach with concentric structures
    # The cell must be symmetric: cell == cell[::-1] row-wise, column-wise, and transpose
    # This means we only need to define the top-left 16x16 quadrant
    
    # Strategy: Try several parametric families and pick the best
    # Family 1: Concentric squares (air core + dielectric shell)
    # Family 2: Cross-shaped dielectric bars
    # Family 3: Combination of rod + cross
    
    best_cell = None
    best_score = -1.0
    
    def build_cell_from_quadrant(q):
        """Build full 32x32 cell from 16x16 top-left quadrant, enforcing C4v symmetry."""
        cell = [[0]*size for _ in range(size)]
        for i in range(size // 2):
            for j in range(size // 2):
                val = q[i][j]
                # Mirror to all four quadrants
                cell[i][j] = val
                cell[size-1-i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][size-1-j] = val
        return cell
    
    def compute_gap_ratio(cell, n_pw=17, n_k=20):
        """Compute the gap-midgap ratio using plane-wave method.
        
        For TM polarization in 2D: (1/eps) * (-k^2) * E = omega^2 * E
        where k is the wavevector.
        
        We solve the eigenvalue problem:
        sum_N' (1/eps)_{k-k'} * |k'|^2 * E_{k'} = omega^2 * E_k
        
        where (1/eps)_{k-k'} are the Fourier coefficients of 1/eps(x,y).
        """
        # Build 1/eps array
        eps_inv = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                eps_val = eps_high if cell[i][j] == 1 else eps_low
                eps_inv[i, j] = 1.0 / eps_val
        
        # Compute Fourier coefficients of 1/eps
        # Use FFT for efficiency
        eps_inv_fft = np.fft.fft2(eps_inv)
        eps_inv_fft = eps_inv_fft / (size * size)  # Normalize
        
        # Plane wave indices
        half_n = n_pw // 2
        pw_indices = list(range(-half_n, half_n + 1))
        n_pw_actual = len(pw_indices)
        
        # Build the matrix for a given k-point
        # For TM: A_{mn} = (1/eps)_{m-n} * |k_m|^2
        # where m,n index the plane waves, and (1/eps)_{m-n} is the Fourier coeff
        
        def get_fourier_coeff(dm, dn):
            """Get Fourier coefficient (1/eps)_{dm, dn}"""
            # dm, dn are integer differences in G-vector indices
            # G = (2*pi/size) * (dm, dn) in our indexing
            # The Fourier coefficient is eps_inv_fft[dm mod size, dn mod size]
            idx_m = dm % size
            idx_n = dn % size
            return eps_inv_fft[idx_m, idx_n]
        
        # K-points in the irreducible BZ
        # Square lattice, irreducible BZ: 0 <= kx <= ky <= pi/a
        # In units of 2*pi/a (our G-vectors are 2*pi*n/size, so k in units of 2*pi/size)
        # Actually, let's use k in units of 2*pi/a where a=1 (lattice constant)
        # Then G = 2*pi*(m, n) for integers m, n
        # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= (pi/a)^2... 
        # Actually for square lattice, the BZ is -pi/a to pi/a in both directions
        # Irreducible: 0 <= kx <= min(ky, pi/a - ky) for 0 <= ky <= pi/a
        # Simpler: sample points in the triangle 0 <= kx <= ky <= pi/a
        
        k_points = []
        # Gamma point
        k_points.append((0.0, 0.0))
        # X point (pi/a, 0) -> in our units where G = 2*pi*n/size... 
        # Let me redefine: use k in units of 2*pi/a
        # Then G = (m, n) for integers m, n
        # BZ: -0.5 <= kx, ky <= 0.5
        # Irreducible: 0 <= kx <= ky <= 0.5
        
        # Sample points
        n_sample = 5
        for i in range(n_sample + 1):
            for j in range(n_sample + 1):
                ky = 0.5 * j / n_sample
                kx = 0.5 * i / n_sample
                if kx <= ky:
                    k_points.append((kx, ky))
        
        # Also add some interior points
        for i in range(n_sample + 1):
            for j in range(n_sample + 1):
                ky = 0.5 * j / n_sample
                kx_max = min(ky, 0.5)
                if kx_max > 0:
                    kx = 0.25 * i / n_sample
                    if kx <= ky:
                        k_points.append((kx, ky))
        
        # Remove duplicates and limit count
        k_points = list(set(k_points))
        if len(k_points) > n_k:
            # Keep a representative subset
            k_points = k_points[:n_k]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # Build the eigenvalue matrix
            # Plane waves: G_m = (m_x, m_y) for m_x, m_y in pw_indices
            # k + G_m has magnitude squared = (kx + m_x)^2 + (ky + m_y)^2
            
            # For TM: the operator is (1/eps) * |k+G|^2
            # Matrix element: <G_m| (1/eps)*|k+G|^2 |G_n> = (1/eps)_{G_m - G_n} * |k + G_n|^2
            # Wait, let me be more careful.
            # The eigenvalue equation: (1/eps(x)) * |k + G|^2 * E(k+G) = omega^2 * E(k+G)
            # Expanding 1/eps in Fourier: (1/eps)(x) = sum_q (1/eps)_q * exp(i q.x)
            # Then: sum_q sum_G' (1/eps)_q * |k + G'|^2 * E(k+G') * exp(i q.x) = omega^2 * E(k+G)
            # For this to hold for all x, we need:
            # sum_G' (1/eps)_{G - G'} * |k + G'|^2 * E(k+G') = omega^2 * E(k+G)
            # where G - G' is the Fourier index.
            
            N = n_pw_actual
            A = np.zeros((N, N), dtype=complex)
            
            for i, (mx, my) in enumerate([(pw_indices[m], pw_indices[n]) for m in range(N) for n in range(N)][:N]):
                pass
            
            # Let me redo this more carefully
            pw_list = [(pw_indices[m], pw_indices[n]) for m in range(N) for n in range(N)]
            # Wait, that's N^2 points, not N. Let me use a linear indexing.
            
            # Actually, let's use a simpler set of plane waves
            # Use all (m, n) where |m|, |n| <= half_n
            pw_list = [(m, n) for m in pw_indices for n in pw_indices]
            N = len(pw_list)
            
            A = np.zeros((N, N), dtype=complex)
            
            for i in range(N):
                Gi_x, Gi_y = pw_list[i]
                for j in range(N):
                    Gj_x, Gj_y = pw_list[j]
                    # (1/eps)_{Gi - Gj}
                    dm = Gi_x - Gj_x
                    dn = Gi_y - Gj_y
                    coeff = get_fourier_coeff(dm, dn)
                    # |k + Gj|^2
                    kmg2 = (kx + Gj_x)**2 + (ky + Gj_y)**2
                    A[i, j] = coeff * kmg2
            
            # Solve eigenvalue problem
            # A is Hermitian (should be, since (1/eps) is real and symmetric)
            # Make sure it's Hermitian
            A = (A + A.conj().T) / 2.0
            
            try:
                eigvals = eigh(A, eigvals_only=True)
                eigvals = np.real(eigvals)
                # omega^2 = eigvals, so omega = sqrt(eigvals)
                # We want bands 4 and 5 (1-indexed), so indices 3 and 4 (0-indexed)
                if len(eigvals) >= 5:
                    omega4 = math.sqrt(max(eigvals[3], 0))
                    omega5 = math.sqrt(max(eigvals[4], 0))
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except Exception:
                continue
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap  # This is half the ratio; the task says 2*gap/midgap
    
    # Define parametric families
    
    def family_concentric_squares(r1, r2, r3, r4):
        """Concentric square rings: dielectric in [r1,r2] and [r3,r4] from center."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                # Distance from center of the 32x32 cell
                # Cell center is at (15.5, 15.5) in pixel coords
                # In the quadrant (top-left), pixel (i,j) corresponds to cell pixel (i,j)
                # Distance from center: dx = 15.5 - j, dy = 15.5 - i (in pixel units)
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0  # Normalize to [0, ~0.71]
                if (r1 <= r < r2) or (r3 <= r < r4):
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q
    
    def family_cross(w1, w2, offset):
        """Cross-shaped dielectric: horizontal and vertical bars."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                # Horizontal bar: |dy| < w1/2 and |dx| < 16
                # Vertical bar: |dx| < w2/2 and |dy| < 16
                if abs(dy) < w1/2.0 or abs(dx) < w2/2.0:
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q
    
    def family_rod_cross(r_rod, w_cross):
        """Dielectric rod at center plus cross-shaped bars."""
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0
                # Rod
                if r < r_rod:
                    q[i][j] = 1
                # Cross
                elif abs(dy) < w_cross/2.0 and abs(dx) < 12:
                    q[i][j] = 1
                elif abs(dx) < w_cross/2.0 and abs(dy) < 12:
                    q[i][j] = 1
                else:
                    q[i][j] = 0
        return q
    
    # Try a grid of parameters for each family
    families = []
    
    # Family 1: Concentric squares
    for r1 in [0.0, 0.1, 0.2, 0.3]:
        for r2 in [0.3, 0.4, 0.5, 0.6]:
            for r3 in [0.5, 0.6, 0.7]:
                for r4 in [0.7, 0.8, 0.9]:
                    if r1 < r2 < r3 < r4:
                        families.append(('concentric', (r1, r2, r3, r4)))
    
    # Family 2: Cross
    for w1 in [2, 4, 6, 8, 10]:
        for w2 in [2, 4, 6, 8, 10]:
            families.append(('cross', (w1, w2, 0)))
    
    # Family 3: Rod + cross
    for r_rod in [0.1, 0.2, 0.3, 0.4]:
        for w_cross in [2, 4, 6, 8]:
            families.append(('rod_cross', (r_rod, w_cross)))
    
    # Evaluate each candidate
    for family_name, params in families:
        if time.time() - start_time > deadline - 30:
            break
        
        if family_name == 'concentric':
            r1, r2, r3, r4 = params
            q = family_concentric_squares(r1, r2, r3, r4)
        elif family_name == 'cross':
            w1, w2, offset = params
            q = family_cross(w1, w2, offset)
        elif family_name == 'rod_cross':
            r_rod, w_cross = params
            q = family_rod_cross(r_rod, w_cross)
        else:
            continue
        
        cell = build_cell_from_quadrant(q)
        score = compute_gap_ratio(cell, n_pw=11, n_k=15)
        
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # If no good cell found, fall back to a simple design
    if best_cell is None:
        # Default: dielectric rod
        q = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(16):
                dx = 15.5 - j
                dy = 15.5 - i
                r = math.sqrt(dx*dx + dy*dy) / 16.0
                q[i][j] = 1 if r < 0.3 else 0
        best_cell = build_cell_from_quadrant(q)
    
    return best_cell
# EVOLVE-BLOCK-END
