"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Designs a 32x32 unit cell using continuous permittivity field optimization.
    Uses low-order Fourier series representation and gradient-based optimization
    to maximize the TM band gap between bands 4 and 5.
    """
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Create pixel coordinate grid
    coords = np.linspace(0, 1, size, endpoint=False) + 0.5/size
    x_grid, y_grid = np.meshgrid(coords, coords)
    
    # Define Fourier basis functions (low-order harmonics)
    # Use up to 3rd order in each direction
    max_order = 3
    
    def fourier_basis(x, y):
        """Generate Fourier basis values at given coordinates."""
        basis_vals = []
        for nx in range(-max_order, max_order + 1):
            for ny in range(-max_order, max_order + 1):
                if nx == 0 and ny == 0:
                    basis_vals.append(np.ones_like(x))
                else:
                    basis_vals.append(np.cos(2 * np.pi * nx * x) * np.cos(2 * np.pi * ny * y))
        return np.array(basis_vals)
    
    # Precompute basis matrix
    basis_matrix = fourier_basis(x_grid, y_grid)  # Shape: (num_basis, size, size)
    num_basis = basis_matrix.shape[0]
    
    # Initialize coefficients with a symmetric pattern
    # Start with a reasonable initial guess
    np.random.seed(42)
    coeffs = np.zeros(num_basis)
    # Add some initial structure
    for idx in range(num_basis):
        # Map index back to (nx, ny)
        nx_idx = idx // (2 * max_order + 1)
        ny_idx = idx % (2 * max_order + 1)
        nx = nx_idx - max_order
        ny = ny_idx - max_order
        if abs(nx) <= 1 and abs(ny) <= 1:
            coeffs[idx] = 0.1 * (1 - abs(nx) - abs(ny))
    
    # Ensure symmetry: only use even harmonics for square symmetry
    # Filter to keep only terms where nx and ny have same parity (both even or both odd)
    symmetric_indices = []
    for idx in range(num_basis):
        nx_idx = idx // (2 * max_order + 1)
        ny_idx = idx % (2 * max_order + 1)
        nx = nx_idx - max_order
        ny = ny_idx - max_order
        # For square symmetry about center, we need cos(nx*x)*cos(ny*y) 
        # which is symmetric if we use only cosine terms (already doing that)
        symmetric_indices.append(idx)
    
    # Objective function: negative of the gap-midgap ratio
    def objective(coeffs):
        """Compute negative gap-midgap ratio to minimize."""
        # Compute permittivity field from coefficients
        eps_field = np.zeros((size, size))
        for idx in range(num_basis):
            eps_field += coeffs[idx] * basis_matrix[idx]
        
        # Clip to valid range
        eps_field = np.clip(eps_field, eps_low, eps_high)
        
        # Convert to binary-ish for the pixel representation
        # Use thresholding
        threshold = (eps_low + eps_high) / 2
        pixel_cell = (eps_field > threshold).astype(int)
        
        # Ensure symmetry
        pixel_cell = np.maximum(pixel_cell, pixel_cell[::-1, :])
        pixel_cell = np.maximum(pixel_cell, pixel_cell[:, ::-1])
        pixel_cell = np.maximum(pixel_cell, pixel_cell.T)
        
        # Compute band structure using plane-wave method
        gap_ratio = compute_gap_ratio(pixel_cell, size, eps_high)
        return -gap_ratio
    
    def compute_gap_ratio(cell, size, eps_high):
        """Compute the gap-midgap ratio using plane-wave method."""
        # Simplified plane-wave calculation
        # Use a subset of plane waves for speed
        num_pw = 15  # 15x15 = 225 plane waves
        pw_indices = range(-num_pw, num_pw + 1)
        
        # Fourier coefficients of permittivity
        eps_fourier = np.zeros((len(pw_indices), len(pw_indices)), dtype=complex)
        for i, nx in enumerate(pw_indices):
            for j, ny in enumerate(pw_indices):
                # Compute Fourier coefficient of the pixel map
                coeff = 0.0
                for x_idx in range(size):
                    for y_idx in range(size):
                        coeff += cell[x_idx, y_idx] * np.exp(-2j * np.pi * (nx * x_idx / size + ny * y_idx / size))
                eps_fourier[i, j] = coeff / (size * size)
        
        # Add background
        eps_mean = np.mean(cell)
        eps_fourier[np.where(np.array(list(zip(pw_indices, pw_indices))) == (0, 0))] = eps_mean
        
        # Evaluate at key points in Brillouin zone
        k_points = [
            (0, 0),          # Gamma
            (0.5, 0),        # X
            (0.5, 0.5),      # M
            (0.25, 0.25),    # Interior
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Build Hamiltonian matrix
            n = len(pw_indices)
            H = np.zeros((n, n), dtype=complex)
            
            for i, nxi in enumerate(pw_indices):
                for j, nyj in enumerate(pw_indices):
                    # (k + G_i) dot (k + G_j)
                    kx_i = kx + nxi
                    ky_i = ky + nyj
                    kx_j = kx + pw_indices[j]
                    ky_j = ky + pw_indices[i]
                    
                    # Inverse permittivity Fourier coefficient
                    inv_eps_coeff = 0.0
                    for m, nmx in enumerate(pw_indices):
                        for l, nmy in enumerate(pw_indices):
                            if (nxi - nmx) == pw_indices[j] and (nyj - nmy) == pw_indices[i]:
                                inv_eps_coeff = eps_fourier[m, l]
                                break
                        else:
                            continue
                        break
                    
                    # Simplified: use direct inverse
                    inv_eps = 1.0 / (eps_mean + 0.5 * (eps_fourier[i, j] if i != j else 0))
                    
                    H[i, j] = (kx_i * kx_j + ky_i * ky_j) * inv_eps
            
            # Solve eigenvalue problem
            eigenvalues = np.linalg.eigvalsh(H)
            eigenvalues = np.sort(eigenvalues)
            
            if len(eigenvalues) >= 5:
                omega4 = np.sqrt(max(eigenvalues[3], 0))
                omega5 = np.sqrt(max(eigenvalues[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 <= max_omega4:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        return 2 * gap / midgap if midgap > 0 else 0.0
    
    # Optimize coefficients
    # Use gradient-free optimization for robustness
    result = minimize(objective, coeffs, method='Nelder-Mead', 
                     options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.01})
    
    optimal_coeffs = result.x
    
    # Generate final cell
    eps_field = np.zeros((size, size))
    for idx in range(num_basis):
        eps_field += optimal_coeffs[idx] * basis_matrix[idx]
    
    eps_field = np.clip(eps_field, eps_low, eps_high)
    threshold = (eps_low + eps_high) / 2
    cell = (eps_field > threshold).astype(int)
    
    # Enforce symmetry
    cell = np.maximum(cell, cell[::-1, :])
    cell = np.maximum(cell, cell[:, ::-1])
    cell = np.maximum(cell, cell.T)
    
    return cell.tolist()

# EVOLVE-BLOCK-END
