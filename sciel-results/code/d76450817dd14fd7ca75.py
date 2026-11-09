import numpy as np
import scipy.optimize as optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit

    N = 8

    # Precompute quadrature points outside the function
    c_vals, w_vals = np.polynomial.legendre.leggauss(100)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals

    def compute_volume(X):
        """
        Compute the union volume of the 64 tubes.
        X is an array of shape (8,) representing the base x-coordinates for each i.
        The y-coordinates are the same as x-coordinates (symmetry).
        """

        total_vol = 0.0
        for idx in range(len(c_vals)):
            c = c_vals[idx]
            w = w_vals[idx]
            # Intervals for each i
            starts = X + c * np.arange(N) / N
            ends = starts + 1.0 / N

            # Sort by start
            order = np.argsort(starts)
            s = starts[order]
            e = ends[order]

            # Merge intervals
            cur_s = s[0]
            cur_e = e[0]
            length = 0.0
            for k in range(1, N):
                if s[k] <= cur_e:
                    cur_e = max(cur_e, e[k])
                else:
                    length += cur_e - cur_s
                    cur_s = s[k]
                    cur_e = e[k]
            length += cur_e - cur_s

            total_vol += length * length * w

        return total_vol

    def fft_to_coords(coeffs):
        """Convert Fourier coefficients to spatial coordinates."""
        # coeffs shape: (n_modes,) where modes are [0, 1, ..., n_modes-1]
        # We use real FFT: coeffs[0] is DC, coeffs[1:] are cosine/sine pairs
        n_modes = len(coeffs)
        # Create frequency indices
        freqs = np.arange(n_modes)
        # Build the signal from coefficients
        t = np.arange(N)
        x = np.zeros(N)
        for m in range(n_modes):
            if m == 0:
                x += coeffs[0]
            else:
                # For real signals, we need both cos and sin
                # coeffs[m] is amplitude, but we need phase too
                # Let's use a different approach: store complex coefficients
                pass
        
        # Simpler: use direct DFT
        # coeffs are real Fourier coefficients: x_k = sum_m a_m * cos(2*pi*m*k/N) + b_m * sin(2*pi*m*k/N)
        # But we only have one coefficient per mode. Let's use:
        # x_k = sum_m c_m * exp(2*pi*i*m*k/N) + c.m * exp(-2*pi*i*m*k/N) for m>=1
        # This gives: x_k = c_0 + sum_{m=1}^{M} 2*Re(c_m * exp(2*pi*i*m*k/N))
        
        # Actually, let's just use the coefficients directly as amplitudes for cosine terms
        # x_k = sum_{m=0}^{M} a_m * cos(2*pi*m*k/N)
        for m in range(n_modes):
            x += coeffs[m] * np.cos(2 * np.pi * m * t / N)
        
        return x

    def coords_to_fft(x):
        """Convert spatial coordinates to Fourier coefficients."""
        n_modes = 4  # Use modes 0,1,2,3
        t = np.arange(N)
        coeffs = np.zeros(n_modes)
        for m in range(n_modes):
            coeffs[m] = np.mean(x * np.cos(2 * np.pi * m * t / N))
        return coeffs

    def obj_fft(coeffs):
        """Objective in Fourier coefficient space."""
        x = fft_to_coords(coeffs)
        # Center the coordinates (remove DC component effect for optimization)
        x_centered = x - np.mean(x)
        # Add back a small offset to avoid numerical issues
        return compute_volume(x_centered)

    # Initial guess: linear shearing
    X0 = -np.arange(N) / N
    X0_centered = X0 - np.mean(X0)
    
    # Get initial Fourier coefficients
    n_modes = 4
    coeffs_init = coords_to_fft(X0_centered)
    
    best_val = obj_fft(coeffs_init)
    best_coeffs = coeffs_init.copy()
    
    # Spectral grid search over Fourier coefficients
    # We'll search in a bounded box around the initial guess
    rng = np.random.default_rng(42)
    
    # First, do a coarse grid search in Fourier space
    # For each mode, search in a range
    search_ranges = []
    for m in range(n_modes):
        if m == 0:
            search_ranges.append((-0.1, 0.1))  # DC component
        else:
            search_ranges.append((-0.3, 0.3))  # Higher modes
    
    # Coarse grid: 5 points per mode
    n_points = 5
    grid_vals = [np.linspace(search_ranges[m][0], search_ranges[m][1], n_points) for m in range(n_modes)]
    
    best_grid_val = best_val
    best_grid_coeffs = best_coeffs.copy()
    
    # Iterate over grid
    for m in range(n_modes):
        for v in grid_vals[m]:
            test_coeffs = best_coeffs.copy()
            test_coeffs[m] = v
            val = obj_fft(test_coeffs)
            if val < best_grid_val:
                best_grid_val = val
                best_grid_coeffs = test_coeffs.copy()
    
    # Refine around the best grid point with local optimization
    if time.time() - start_time < timeout - 100:
        try:
            res = optimize.minimize(obj_fft, best_grid_coeffs, method='Powell',
                                    options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-13})
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Try additional initializations in Fourier space
    additional_inits = [
        np.zeros(n_modes),  # all zero
        np.array([0, 0.2, 0, -0.1]),  # some random patterns
        np.array([0, -0.2, 0.1, 0]),
        np.array([0, 0.1, -0.1, 0.05]),
    ]
    
    for init in additional_inits:
        if time.time() - start_time > timeout - 50:
            break
        try:
            res = optimize.minimize(obj_fft, init, method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12})
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() - start_time < timeout - 30:
        try:
            res = optimize.minimize(obj_fft, best_grid_coeffs, method='Powell',
                                    options={'maxiter': 5000, 'xtol': 1e-11, 'ftol': 1e-14})
            if res.fun < best_grid_val:
                best_grid_val = res.fun
                best_grid_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Convert best Fourier coefficients back to spatial coordinates
    X_best = fft_to_coords(best_grid_coeffs)
    X_best = X_best - np.mean(X_best)  # Center
    
    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = X_best[j]

    return positions
