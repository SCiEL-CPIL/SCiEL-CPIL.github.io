"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import eigh

def design(problem):
    """
    Design a 32x32 symmetric unit cell to maximize the TM band gap between bands 4 and 5.
    Uses a local search over superellipse parameters (exponent, size) with a fast
    plane-wave band solver. Enforces 4-fold symmetry and inversion symmetry.
    """
    size = problem["pixels"]  # 32
    eps_low = 1
    eps_high = 12
    lower_band = 4
    polarization = "TM"

    deadline = time.time() + 110  # Safety margin: evaluator kills at 230s, keep 120s for overhead

    def make_cell(a, b, n):
        """Create a symmetric cell with superellipse shape. a, b are half-widths, n is exponent."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Superellipse: |x/a|^n + |y/b|^n <= 1
                if a > 0 and b > 0:
                    val = abs(x / a) ** n + abs(y / b) ** n
                    cell[i, j] = 1 if val <= 1.0 else 0
        return cell

    def compute_gap(cell, num_pw=17):
        """
        Compute the TM band gap ratio between bands 4 and 5 using plane-wave method.
        num_pw: number of plane waves in each direction (total = num_pw^2)
        """
        cell = np.asarray(cell)
        # Compute inverse epsilon
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps

        # Fourier coefficients of inv_eps
        # We need the first few Fourier components
        # Use direct computation for small number of G vectors
        N = size
        G_max = (num_pw - 1) // 2  # max G index
        
        # Precompute Fourier coefficients for all needed G vectors
        # inv_eps(G) = (1/N^2) sum_{i,j} inv_eps(i,j) exp(-i G . r_ij)
        # For real symmetric function, this is real
        
        # Generate G vectors
        G_vectors = []
        for gx in range(-G_max, G_max + 1):
            for gy in range(-G_max, G_max + 1):
                G_vectors.append((gx, gy))
        
        num_G = len(G_vectors)
        
        # Compute Fourier coefficients
        # r_ij = (i/N, j/N)
        # G . r = 2*pi*(gx*i/N + gy*j/N)
        
        fourier_coeffs = np.zeros(num_G, dtype=complex)
        for idx, (gx, gy) in enumerate(G_vectors):
            if gx == 0 and gy == 0:
                fourier_coeffs[idx] = np.mean(inv_eps)
            else:
                # Compute sum over all pixels
                phase = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        phase[i, j] = -2 * np.pi * (gx * i / N + gy * j / N)
                fourier_coeffs[idx] = np.sum(inv_eps * np.exp(1j * phase)) / (N * N)
        
        # Build the eigenvalue matrix for TM polarization
        # For TM: (k^2) * A = omega^2 * B * A
        # where A_g = sum_h epsilon^{-1}(g-h) * k_h^2 * A_h
        # This is a generalized eigenvalue problem
        
        # Actually, for TM polarization, the equation is:
        # omega^2 = (c^2 * k^2) / <epsilon^{-1}> type structure
        # More precisely, we solve: sum_h epsilon^{-1}(g-h) (k+h)^2 A_h = omega^2 A_g
        
        # Let's use the standard formulation:
        # For TM: det[(k+G)^2 delta_{G,G'} - omega^2 epsilon^{-1}(G-G')] = 0
        
        # We'll evaluate at several k-points in the irreducible BZ
        # Irreducible BZ for square lattice: 0 <= ky <= kx, kx <= pi/a, ky <= pi/a
        # Key points: Gamma (0,0), X (pi/a, 0), M (pi/a, pi/a), and along X-M line
        
        # For simplicity, evaluate at Gamma, X, M, and midpoint of X-M
        # a = 1 (lattice constant)
        
        k_points = [
            (0.0, 0.0),           # Gamma
            (np.pi, 0.0),         # X
            (np.pi, np.pi),       # M
            (np.pi, np.pi/2),     # midpoint of X-M
            (np.pi/2, 0.0),       # midpoint of Gamma-X
            (np.pi/2, np.pi/2),   # center
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix
            A = np.zeros((num_G, num_G), dtype=complex)
            for gi in range(num_G):
                G_i = np.array(G_vectors[gi])
                kpi = k + G_i
                for gj in range(num_G):
                    G_j = np.array(G_vectors[gj])
                    G_diff = G_i - G_j
                    # Find index of G_diff in G_vectors
                    # Since G_diff might not be in our list, we need to handle this
                    # Actually, for the plane wave method, we need epsilon^{-1}(G_i - G_j)
                    # which is the Fourier coefficient at G_i - G_j
                    # If G_i - G_j is outside our range, we approximate as 0 (or use the fact that
                    # for a periodic function, high-order coefficients are small)
                    
                    # Find if G_diff is in our G_vectors list
                    idx_diff = -1
                    for idx, (gx, gy) in enumerate(G_vectors):
                        if gx == G_diff[0] and gy == G_diff[1]:
                            idx_diff = idx
                            break
                    
                    if idx_diff >= 0:
                        eps_inv = fourier_coeffs[idx_diff]
                    else:
                        # Approximate as zero for high-order terms
                        eps_inv = 0.0
                    
                    A[gi, gj] = (kpi[0] ** 2 + kpi[1] ** 2) * eps_inv if gi == gj else 0.0
            
            # Wait, this is not quite right. Let me reformulate.
            # The TM band equation is:
            # sum_{G'} epsilon^{-1}(G - G') (k + G')^2 A_{G'} = omega^2 A_G
            # So the matrix M has elements M_{G,G'} = epsilon^{-1}(G - G') (k + G')^2
            # And we solve M @ A = omega^2 @ A
            
            A = np.zeros((num_G, num_G), dtype=complex)
            for gi in range(num_G):
                G_i = np.array(G_vectors[gi])
                for gj in range(num_G):
                    G_j = np.array(G_vectors[gj])
                    G_diff = G_i - G_j
                    # Find epsilon^{-1}(G_diff)
                    idx_diff = -1
                    for idx, (gx, gy) in enumerate(G_vectors):
                        if gx == G_diff[0] and gy == G_diff[1]:
                            idx_diff = idx
                            break
                    
                    if idx_diff >= 0:
                        eps_inv = fourier_coeffs[idx_diff]
                    else:
                        eps_inv = 0.0
                    
                    k_j = k + G_j
                    A[gi, gj] = eps_inv * (k_j[0] ** 2 + k_j[1] ** 2)
            
            # Solve eigenvalue problem
            try:
                # The matrix should be Hermitian if the cell has inversion symmetry
                # Make it Hermitian by symmetrizing
                A = (A + A.conj().T) / 2
                eigvals = eigh(A, eigvals_only=True)
                eigvals = np.sort(np.real(eigvals))
                
                # omega^2 = eigvals, so omega = sqrt(eigvals)
                # We want bands 4 and 5 (1-indexed), so indices 3 and 4 (0-indexed)
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                continue
        
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
        
        gap_ratio = 2 * (min_omega_5 - max_omega_4) / (min_omega_5 + max_omega_4)
        return gap_ratio

    # Initial parameters
    best_cell = make_cell(0.2, 0.2, 2.0)  # Circle of radius 0.2
    best_score = compute_gap(best_cell, num_pw=11)
    
    # Local search over parameters
    params = [0.2, 0.2, 2.0]  # a, b, n
    step_sizes = [0.01, 0.01, 0.1]
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        improved = False
        for p in range(3):
            for direction in [-1, 1]:
                new_params = params.copy()
                new_params[p] += direction * step_sizes[p]
                
                # Keep parameters in reasonable range
                if p == 0 and (new_params[p] < 0.05 or new_params[p] > 0.45):
                    continue
                if p == 1 and (new_params[p] < 0.05 or new_params[p] > 0.45):
                    continue
                if p == 2 and (new_params[p] < 0.5 or new_params[p] > 8.0):
                    continue
                
                new_cell = make_cell(new_params[0], new_params[1], new_params[2])
                new_score = compute_gap(new_cell, num_pw=11)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell
                    params = new_params
                    improved = True
        
        # Reduce step size if no improvement
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if step_sizes[0] < 0.001:
                break
    
    # Convert numpy array to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
# EVOLVE-BLOCK-END
