import numpy as np
import time
import scipy.optimize as optimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit
    
    N = 8

    # Precompute quadrature
    c_vals, w_vals = np.polynomial.legendre.leggauss(120)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals
    Nc = len(c_vals)

    def compute_volume(X, Y):
        """
        Compute the union volume of the 64 tubes with separable product structure.
        X: shape (8,) base x-coords for i-index
        Y: shape (8,) base y-coords for j-index
        At height c, x-intervals for i: [X[i] + c*i/N, X[i] + c*i/N + 1/N]
        y-intervals for j: [Y[j] + c*j/N, Y[j] + c*j/N + 1/N]
        Union area = (1D x-union length) * (1D y-union length) due to product structure.
        """
        # Vectorize over c
        # starts_x shape: (Nc, N)
        idx = np.arange(N)
        starts_x = X[None, :] + c_vals[:, None] * idx[None, :] / N
        ends_x = starts_x + 1.0 / N
        
        starts_y = Y[None, :] + c_vals[:, None] * idx[None, :] / N
        ends_y = starts_y + 1.0 / N
        
        # Sort and merge for each c
        order_x = np.argsort(starts_x, axis=1)
        s_x = np.take_along_axis(starts_x, order_x, axis=1)
        e_x = np.take_along_axis(ends_x, order_x, axis=1)
        
        order_y = np.argsort(starts_y, axis=1)
        s_y = np.take_along_axis(starts_y, order_y, axis=1)
        e_y = np.take_along_axis(ends_y, order_y, axis=1)
        
        # Merge intervals (vectorized over c)
        len_x = np.zeros(Nc)
        len_y = np.zeros(Nc)
        
        for dim, s, e in [(0, s_x, e_x), (1, s_y, e_y)]:
            cur_s = s[:, 0].copy()
            cur_e = e[:, 0].copy()
            length = np.zeros(Nc)
            for k in range(1, N):
                overlap = s[:, k] <= cur_e
                cur_e = np.maximum(cur_e, e[:, k])
                # When no overlap, add current segment
                length += np.where(overlap, 0.0, cur_e - cur_s)
                # Update for next iteration
                new_s = np.where(overlap, cur_s, s[:, k])
                new_e = np.where(overlap, cur_e, e[:, k])
                cur_s = new_s
                cur_e = new_e
            length += cur_e - cur_s
            if dim == 0:
                len_x = length
            else:
                len_y = length
        
        total_vol = np.sum(len_x * len_y * w_vals)
        return total_vol

    def compute_volume_single(X_full):
        """For symmetric case (Y=X), used in initial optimization."""
        return compute_volume(X_full, X_full)

    # Parametric representation using Fourier modes for smooth coherence
    # X[i] = a0 + a1*i + a2*cos(2*pi*i/N) + a3*sin(2*pi*i/N) + a4*cos(4*pi*i/N) + a5*sin(4*pi*i/N)
    # This gives 6 params per dimension instead of 7 free params
    
    idx = np.arange(N)
    fourier_basis = np.column_stack([
        np.ones(N),
        idx.astype(float),
        np.cos(2*np.pi*idx/N),
        np.sin(2*np.pi*idx/N),
        np.cos(4*np.pi*idx/N),
        np.sin(4*np.pi*idx/N)
    ])
    
    def params_to_positions(params):
        """Convert 6 Fourier params to 8 positions."""
        return params @ fourier_basis.T
    
    def obj_sym_fourier(params):
        X_full = params_to_positions(params)
        return compute_volume_single(X_full)
    
    # Initial guess: linear shearing
    X0 = -np.arange(N) / N
    
    # Fit initial Fourier coefficients
    def fit_fourier(positions):
        """Least squares fit of positions to Fourier basis."""
        coeffs, _, _, _ = np.linalg.lstsq(fourier_basis, positions, rcond=None)
        return coeffs
    
    X_init_coeffs = fit_fourier(X0)
    
    # Phase 1: Optimize symmetric case with Fourier params (6 params)
    rng = np.random.default_rng(42)
    best_val = compute_volume_single(X0)
    X_best_coeffs = X_init_coeffs.copy()
    
    # Multiple initializations in Fourier space
    inits_sym = [
        X_init_coeffs.copy(),
        np.array([0.0, -0.125, 0.0, 0.0, 0.0, 0.0]),  # linear
        np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),  # cosine bump
        np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0]),  # sine bump
        rng.normal(0, 0.1, 6),
        rng.normal(0, 0.1, 6),
        np.array([0.0, -0.0625, 0.1, 0.0, 0.05, 0.0]),
        np.array([0.0, -0.1, 0.0, 0.0, 0.08, 0.0]),
    ]
    
    for init in inits_sym:
        if time.time() - start_time > timeout - 180:
            break
        try:
            res = optimize.minimize(obj_sym_fourier, init, method='Powell',
                                    options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-13})
            if time.time() - start_time > timeout - 180:
                break
            if res.fun < best_val:
                best_val = res.fun
                X_best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Get best symmetric positions
    X_best_sym = params_to_positions(X_best_coeffs)
    
    # Phase 2: Asymmetric optimization with Fourier params (12 params)
    def obj_asym(params):
        X_opt = params[:6]
        Y_opt = params[6:]
        X_full = params_to_positions(X_opt)
        Y_full = params_to_positions(Y_opt)
        return compute_volume(X_full, Y_full)
    
    # Initialize asymmetric from best symmetric
    asym_init = np.concatenate([X_best_coeffs, X_best_coeffs])
    asym_best = asym_init.copy()
    asym_best_val = obj_asym(asym_best)
    
    # Try asymmetric initializations
    if time.time() - start_time < timeout - 90:
        asym_inits = [
            asym_init.copy(),
            np.concatenate([X_best_coeffs, np.zeros(6)]),
            np.concatenate([np.zeros(6), X_best_coeffs]),
            np.concatenate([X_best_coeffs, rng.normal(0, 0.05, 6)]),
            np.concatenate([rng.normal(0, 0.05, 6), X_best_coeffs]),
            rng.normal(0, 0.08, 12),
            rng.normal(0, 0.08, 12),
        ]
        
        for ainit in asym_inits:
            if time.time() - start_time > timeout - 30:
                break
            try:
                res_a = optimize.minimize(obj_asym, ainit, method='Powell',
                                          options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-13})
                if time.time() - start_time > timeout - 30:
                    break
                if res_a.fun < asym_best_val:
                    asym_best_val = res_a.fun
                    asym_best = res_a.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() - start_time < timeout - 15:
        try:
            res_final = optimize.minimize(obj_asym, asym_best, method='Powell',
                                          options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-15})
            if res_final.fun < asym_best_val:
                asym_best_val = res_final.fun
                asym_best = res_final.x.copy()
        except Exception:
            pass
    
    # Extract X and Y from best asymmetric solution
    X_final = params_to_positions(asym_best[:6])
    Y_final = params_to_positions(asym_best[6:])

    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_final[i]
        positions[k, 1] = Y_final[j]

    return positions
