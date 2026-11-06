"""Design a D4-symmetric 32x32 photonic crystal unit cell for a large TM band gap."""
import math
import time
import numpy as np
from scipy.special import j0, j1

# EVOLVE-BLOCK-START

def _compute_gap_fast(cell, size=32, n_pw=17, n_k=8, eps_high=12.0):
    """
    Fast approximate band-structure calculation using a small plane-wave basis.
    Returns the gap-midgap ratio between bands 4 and 5 (1-indexed).
    """
    N = n_pw
    # Build reciprocal lattice vectors: G = (m, n) with m, n in range(-N//2, N//2+1)
    # For a square lattice with lattice constant a=1, G = 2*pi*(m, n)
    # We use a truncated set.
    
    half = N // 2
    G_vecs = []
    for m in range(-half, half + 1):
        for n in range(-half, half + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs, dtype=float) * 2.0 * np.pi
    nG = len(G_vecs)
    
    # Compute Fourier coefficients of 1/epsilon for each G
    # 1/eps = 1 for air (pixel=0), 1/12 for dielectric (pixel=1)
    # We need the discrete Fourier coefficients of the 1/eps map
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i, j] = 1.0 if cell[i][j] == 0 else 1.0 / eps_high
    
    # Compute Fourier coefficients using FFT
    # The Fourier coefficient for G=(m,n) is (1/N^2) * sum_{i,j} f(i,j) * exp(-i*2*pi*(m*i/N + n*j/N))
    # But our G vectors are in units of 2*pi, and the pixel grid is size x size
    # So the DFT index for G=(m,n) corresponds to DFT frequency (m mod N, n mod N)
    
    # Use FFT to get all coefficients at once
    fft_inv_eps = np.fft.fft2(inv_eps)
    # fft_inv_eps[m, n] = sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(m*i/N + n*j/N))
    # So the coefficient for G=(m,n) is fft_inv_eps[m % size, n % size] / size^2
    
    # But wait - our G vectors are (m, n) where m, n are integers, and the lattice constant is 1.
    # The pixel grid represents one unit cell of size 1x1, so each pixel is 1/size x 1/size.
    # The Fourier coefficient of the periodic function f(x,y) with period 1 is:
    # f_mn = integral_0^1 integral_0^1 f(x,y) exp(-i 2 pi (m x + n y)) dx dy
    # Approximated by: (1/size^2) * sum_{i,j} f(i/size, j/size) exp(-i 2 pi (m i/size + n j/size))
    # = fft_inv_eps[m % size, n % size] / size^2
    
    # So for G=(m,n), the coefficient is:
    # f_mn = fft_inv_eps[m % size, n % size] / size^2
    
    # For TM polarization, the eigenvalue problem is:
    # det[(k - G)^2 * delta - k^2 * (1/eps)_GG'] = 0
    # where (1/eps)_GG' = f_{G-G'}
    
    # We'll solve for k^2 at several k-points in the irreducible BZ.
    
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx + ky <= 1 (in units of pi/a)
    # Actually, the BZ is [-pi, pi] x [-pi, pi] (in units of 1/a, with a=1)
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= pi... no.
    # For square lattice, the irreducible BZ is: 0 <= kx <= ky, kx + ky <= pi
    # Wait, the BZ boundary is at kx = ±pi, ky = ±pi.
    # The irreducible wedge under D4 symmetry is: 0 <= kx <= ky, kx + ky <= pi
    # No, that's not right either. Let me think again.
    
    # The square BZ is [-pi, pi] x [-pi, pi].
    # D4 symmetry includes reflections and 90-degree rotations.
    # The irreducible wedge is: 0 <= kx <= ky, kx + ky <= pi
    # Actually, the standard irreducible wedge for square lattice is:
    # 0 <= kx <= ky, kx + ky <= pi
    # This covers 1/8 of the BZ.
    
    # Let's use k-points:
    # Gamma: (0, 0)
    # M: (pi, 0)
    # X: (pi/2, pi/2) -- wait, no. For square lattice with a=1, the BZ is [-pi, pi]^2.
    # High-symmetry points:
    # Gamma = (0, 0)
    # M = (pi, 0) or (0, pi)
    # X = (pi, pi) -- no, that's a corner.
    
    # Actually, for a square lattice with lattice constant a, the reciprocal lattice vectors are
    # b1 = (2pi/a, 0), b2 = (0, 2pi/a). The BZ is [-pi/a, pi/a] x [-pi/a, pi/a].
    # High-symmetry points:
    # Gamma = (0, 0)
    # M = (pi/a, 0)
    # X = (pi/a, pi/a)
    
    # With a=1:
    # Gamma = (0, 0)
    # M = (pi, 0)
    # X = (pi, pi)
    
    # Irreducible wedge: 0 <= kx <= ky, kx + ky <= pi
    # Wait, that doesn't include X = (pi, pi). Let me reconsider.
    
    # The irreducible wedge for the square BZ under D4 is:
    # 0 <= kx <= ky, kx + ky <= pi
    # This is a triangle with vertices (0,0), (pi/2, pi/2), (0, pi).
    # But (pi, pi) is not in this wedge. That's because (pi, pi) is equivalent to (0,0) under
    # a reciprocal lattice vector? No, (pi, pi) - (2pi, 0) = (-pi, pi), which is in the BZ.
    # Hmm, I think I'm confusing myself.
    
    # Let me just use a set of k-points that sample the irreducible BZ.
    # The irreducible BZ is the region: 0 <= kx <= ky, kx + ky <= pi
    # Points:
    # (0, 0) - Gamma
    # (pi/2, pi/2) - center of the wedge boundary
    # (0, pi) - M point (equivalent to (pi, 0))
    # (pi/4, pi/4) - interior
    # (pi/6, pi/3) - interior
    # (pi/3, 2*pi/3) - near the ky = kx + pi/3 boundary... 
    
    # Let me just pick a few representative k-points:
    k_points = [
        (0.0, 0.0),           # Gamma
        (0.5 * np.pi, 0.5 * np.pi),  # (pi/2, pi/2)
        (0.0, np.pi),         # M
        (0.25 * np.pi, 0.25 * np.pi),
        (0.2 * np.pi, 0.5 * np.pi),
        (0.3 * np.pi, 0.6 * np.pi),
        (0.1 * np.pi, 0.8 * np.pi),
        (0.4 * np.pi, 0.7 * np.pi),
    ]
    
    max_gap_ratio = -np.inf
    
    for kx, ky in k_points:
        k_vec = np.array([kx, ky])
        
        # Build the eigenvalue matrix
        # For TM: (k - G)^2 * delta_ij - k^2 * (1/eps)_{G_i - G_j} = 0
        # This is a generalized eigenvalue problem, but we can convert to standard form.
        
        # Actually, for TM polarization, the equation is:
        # sum_{G'} (k - G)^2 delta_{GG'} E_{G'} = k^2 sum_{G'} (1/eps)_{G-G'} E_{G'}
        # This is: A E = k^2 B E, where A is diagonal with A_GG = (k-G)^2
        # and B_GG' = (1/eps)_{G-G'}
        
        # To solve, we can use the Cholesky decomposition of B if it's positive definite.
        # Or we can just solve the generalized eigenvalue problem directly.
        
        # Build A (diagonal) and B
        A_diag = np.zeros(nG)
        for idx, (gm, gn) in enumerate(G_vecs / (2.0 * np.pi)):
            kx_g = kx - gm * 2.0 * np.pi
            ky_g = ky - gn * 2.0 * np.pi
            A_diag[idx] = kx_g ** 2 + ky_g ** 2
        
        # Build B matrix
        B = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            for j in range(nG):
                dGx = G_vecs[i, 0] - G_vecs[j, 0]
                dGy = G_vecs[i, 1] - G_vecs[j, 1]
                # Get the Fourier coefficient for (dGx/(2pi), dGy/(2pi))
                mx = int(round(dGx / (2.0 * np.pi)))
                my = int(round(dGy / (2.0 * np.pi)))
                B[i, j] = fft_inv_eps[mx % size, my % size] / (size * size)
        
        # Solve the generalized eigenvalue problem A E = k^2 B E
        # Convert to standard form: B^{-1/2} A B^{-1/2} E' = k^2 E'
        # First, check if B is positive definite
        try:
            eigvals, eigvecs = np.linalg.eigh(B)
            if np.min(eigvals) <= 0:
                continue  # B is not positive definite, skip this k-point
            B_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
            C = B_inv_sqrt @ np.diag(A_diag) @ B_inv_sqrt
            C = (C + C.conj().T) / 2.0  # Ensure Hermitian
            k2_vals = np.linalg.eigvalsh(C)
            k2_vals = np.sort(k2_vals)
            # Take the 5th and 4th eigenvalues (0-indexed: 3 and 4)
            if len(k2_vals) >= 5:
                k4 = np.sqrt(max(k2_vals[3], 0))
                k5 = np.sqrt(max(k2_vals[4], 0))
                if k4 + k5 > 0:
                    gap_ratio = 2.0 * (k5 - k4) / (k5 + k4)
                    if gap_ratio > max_gap_ratio:
                        max_gap_ratio = gap_ratio
        except:
            continue
    
    return max_gap_ratio


def _generate_cell(params, size=32):
    """
    Generate a D4-symmetric binary cell from Bessel function parameters.
    params: dict with keys 'w0', 'kr', 'w4', 'k4', 'w8', 'k8', 'threshold'
    """
    cell = np.zeros((size, size), dtype=int)
    w0 = params.get('w0', 1.0)
    kr = params.get('kr', 1.0)
    w4 = params.get('w4', 0.5)
    k4 = params.get('k4', 2.0)
    w8 = params.get('w8', 0.2)
    k8 = params.get('k8', 4.0)
    threshold = params.get('threshold', 0.0)
    
    # Center of the cell
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    
    for i in range(size):
        for j in range(size):
            dx = j - cx
            dy = i - cy
            r = math.sqrt(dx * dx + dy * dy)
            theta = math.atan2(dy, dx)
            
            # Superposition of Bessel functions
            val = w0 * j0(kr * r)
            val += w4 * j0(k4 * r) * math.cos(4 * theta)
            val += w8 * j0(k8 * r) * math.cos(8 * theta)
            
            cell[i, j] = 1 if val > threshold else 0
    
    # Ensure D4 symmetry by averaging over the symmetry group
    # The cell should be invariant under:
    # 1. Identity
    # 2. 90-degree rotation
    # 3. 180-degree rotation
    # 4. 270-degree rotation
    # 5. Reflection about x-axis
    # 6. Reflection about y-axis
    # 7. Reflection about x=y
    # 8. Reflection about x=-y
    
    # For a 32x32 cell, we can enforce symmetry by setting each orbit to the majority value
    # Or we can just take the minimum (or maximum) over the orbit to ensure invariance
    
    # Let's enforce by taking the AND (minimum) over the orbit for dielectric (1)
    # This ensures that if any pixel in the orbit is air, the whole orbit is air
    # Actually, that might make the cell too small. Let's use the average and threshold.
    
    # Better approach: compute the value at each pixel, then for each D4 orbit,
    # set all pixels to the majority value.
    
    # For simplicity, let's just enforce the symmetry by construction.
    # The Bessel function approach with cos(4m*theta) should already be D4-symmetric
    # if we're careful about the center.
    
    # Actually, cos(4*theta) is invariant under 90-degree rotation, so the pattern
    # should be D4-symmetric. But due to pixelation, we need to enforce it.
    
    # Let's enforce by averaging over the 8 symmetry operations
    sym_cell = np.zeros((size, size), dtype=float)
    count = np.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            # 8 symmetry operations
            coords = [
                (i, j),
                (j, size - 1 - i),
                (size - 1 - i, size - 1 - j),
                (size - 1 - j, i),
                (i, size - 1 - j),
                (size - 1 - j, size - 1 - i),
                (size - 1 - i, j),
                (j, i),
            ]
            val = 0
            for ci, cj in coords:
                val += cell[ci, cj]
            sym_cell[i, j] = val / 8.0
            count[i, j] = 8
    
    cell = (sym_cell > 0.5).astype(int)
    
    return cell.tolist()


def design(problem):
    """
    Design a 32x32 photonic crystal unit cell for a large TM band gap.
    Uses a Bessel-function parameterized family and local search.
    """
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    # Initial parameter set
    # Target: large gap between bands 4 and 5
    # The seed program uses a dielectric rod of radius 0.2, which gives a score of 1.0
    # We want to do better.
    
    # Start with parameters that approximate a dielectric disk
    best_params = {
        'w0': 1.0,
        'kr': 4.0,
        'w4': 0.0,
        'k4': 2.0,
        'w8': 0.0,
        'k8': 4.0,
        'threshold': 0.0,
    }
    
    # Generate initial cell
    best_cell = _generate_cell(best_params, size)
    best_score = _compute_gap_fast(best_cell, size, eps_high=eps_high)
    
    # Local search
    param_names = ['w0', 'kr', 'w4', 'k4', 'w8', 'k8', 'threshold']
    step_sizes = {'w0': 0.1, 'kr': 0.2, 'w4': 0.05, 'k4': 0.2, 'w8': 0.02, 'k8': 0.2, 'threshold': 0.05}
    
    # Bounds
    bounds = {
        'w0': (0.5, 2.0),
        'kr': (1.0, 8.0),
        'w4': (-0.5, 0.5),
        'k4': (1.0, 6.0),
        'w8': (-0.3, 0.3),
        'k8': (2.0, 8.0),
        'threshold': (-0.5, 0.5),
    }
    
    # Coordinate descent with random restarts
    n_restarts = 5
    max_iter = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            params = dict(best_params)
        else:
            # Random restart
            params = {}
            for name in param_names:
                lo, hi = bounds[name]
                params[name] = lo + (hi - lo) * (0.2 + 0.6 * (restart / n_restarts) * (0.5 + 0.5 * math.sin(restart * 7.0)))
        
        for iteration in range(max_iter):
            if time.time() > deadline - 30.0:
                break
            
            improved = False
            for name in param_names:
                if time.time() > deadline - 30.0:
                    break
                
                step = step_sizes[name]
                for direction in [-1, 1]:
                    new_params = dict(params)
                    new_val = params[name] + direction * step
                    lo, hi = bounds[name]
                    if new_val < lo or new_val > hi:
                        continue
                    new_params[name] = new_val
                    
                    new_cell = _generate_cell(new_params, size)
                    new_score = _compute_gap_fast(new_cell, size, eps_high=eps_high)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_params = dict(new_params)
                        best_cell = new_cell
                        params = dict(new_params)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try smaller steps
                step_scale = 0.5
                for name in param_names:
                    if time.time() > deadline - 30.0:
                        break
                    step = step_sizes[name] * step_scale
                    for direction in [-1, 1]:
                        new_params = dict(params)
                        new_val = params[name] + direction * step
                        lo, hi = bounds[name]
                        if new_val < lo or new_val > hi:
                            continue
                        new_params[name] = new_val
                        
                        new_cell = _generate_cell(new_params, size)
                        new_score = _compute_gap_fast(new_cell, size, eps_high=eps_high)
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_params = dict(new_params)
                            best_cell = new_cell
                            params = dict(new_params)
                            improved = True
                            break
                    if improved:
                        break
    
    # Return the best cell found
    if time.time() > deadline - 10.0:
        # Return a simple dielectric disk as fallback
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = (j + 0.5) / size - 0.5
                y = (i + 0.5) / size - 0.5
                row.append(1 if x * x + y * y < 0.2 ** 2 else 0)
            cell.append(row)
        return cell
    
    return best_cell

# EVOLVE-BLOCK-END
