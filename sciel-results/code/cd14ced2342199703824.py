import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # If n is very large, subsample for kernel regression to ensure speed
    max_samples_for_smoothing = 8000
    if n > max_samples_for_smoothing:
        idx = np.random.choice(n, max_samples_for_smoothing, replace=False)
        x_smooth = x[idx]
        t_smooth = t[idx]
    else:
        x_smooth = x
        t_smooth = t
    
    # Estimate bandwidth for Nadaraya-Watson
    std_x = np.std(x_smooth)
    if std_x < 1e-10:
        return np.zeros(6)
    n_smooth = len(x_smooth)
    
    # Search over several bandwidths to find the best fit
    # Try multiple bandwidth scales
    base_h = 1.06 * std_x * (n_smooth ** (-0.2))
    bandwidths = [base_h * f for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
    
    best_score = np.inf
    best_coeff = np.zeros(6)
    
    for h in bandwidths:
        if time.time() > deadline - 30:
            break
        
        # Compute denoised targets using Gaussian Kernel Regression
        t_denoised = np.zeros_like(t_smooth)
        
        chunk_size = 1000
        for i in range(0, n_smooth, chunk_size):
            end = min(i + chunk_size, n_smooth)
            x_chunk = x_smooth[i:end]
            diff = x_chunk[:, np.newaxis] - x_smooth[np.newaxis, :]
            weights = np.exp(-0.5 * (diff / h) ** 2)
            w_sum = np.sum(weights, axis=1)
            wt_sum = np.sum(weights * t_smooth[np.newaxis, :], axis=1)
            t_denoised[i:end] = wt_sum / np.where(w_sum > 1e-10, w_sum, 1.0)
            
            if time.time() > deadline - 20:
                break
        
        # Fit degree 5 polynomial using Legendre basis for stability
        mu = np.mean(x_smooth)
        sigma = np.std(x_smooth)
        if sigma < 1e-10:
            sigma = 1.0
        
        x_scaled = (x_smooth - mu) / sigma
        
        # Build Legendre polynomial basis
        # Use np.polynomial.legendre.legvander
        V = np.polynomial.legendre.legvander(x_scaled, 5)
        
        # Solve least squares
        try:
            coeff_leg, residuals, rank, sv = np.linalg.lstsq(V, t_denoised, rcond=None)
        except Exception:
            continue
        
        # Convert Legendre coefficients to monomial coefficients
        # First convert from scaled x to original x
        # P(x) = sum c_i * L_i((x-mu)/sigma)
        # We need to express this in powers of x
        
        # Method: evaluate the Legendre polynomial at 6 points, then fit monomial
        # This is more robust than trying to analytically convert
        
        # Generate 6 well-distributed points in the range of x
        x_min = np.min(x_smooth)
        x_max = np.max(x_smooth)
        x_eval = np.linspace(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min), 6)
        
        # Evaluate the Legendre-based polynomial at these points
        x_eval_scaled = (x_eval - mu) / sigma
        V_eval = np.polynomial.legendre.legvander(x_eval_scaled, 5)
        y_eval = V_eval @ coeff_leg
        
        # Now fit a degree-5 monomial polynomial to (x_eval, y_eval)
        try:
            coeff_monomial = np.polynomial.polynomial.polyfit(x_eval, y_eval, 5)
        except Exception:
            continue
        
        # Score: use simple MSE on training data as proxy
        x_test = x_smooth[:min(1000, n_smooth)]
        t_test = t_smooth[:min(1000, n_smooth)]
        pred = np.polynomial.polynomial.polyval(x_test, coeff_monomial)
        mse = np.mean((pred - t_test) ** 2)
        
        if mse < best_score:
            best_score = mse
            best_coeff = coeff_monomial
    
    # Robust iterative reweighting: refine the best polynomial using Huber weights
    if best_score != np.inf and time.time() < deadline - 10:
        try:
            # Compute residuals for all training points using the current best polynomial
            pred_all = np.polynomial.polynomial.polyval(x_smooth, best_coeff)
            residuals = t_smooth - pred_all
            
            # Compute Huber weights: w = 1 if |r| <= delta, else delta/|r|
            # Use median absolute deviation to estimate scale
            mad = np.median(np.abs(residuals - np.median(residuals)))
            if mad < 1e-10:
                mad = np.std(residuals)
            if mad < 1e-10:
                mad = 1.0
            
            # Huber delta: typically 1.5 * sigma, where sigma ~ 1.4826 * MAD
            huber_delta = 1.5 * 1.4826 * mad
            
            # Compute weights
            abs_res = np.abs(residuals)
            weights_huber = np.where(abs_res <= huber_delta, 1.0, huber_delta / abs_res)
            
            # Re-fit with weighted least squares using Legendre basis
            x_scaled_all = (x_smooth - mu) / sigma
            V_all = np.polynomial.legendre.legvander(x_scaled_all, 5)
            
            # Weighted least squares: minimize sum(w_i * (V_i @ coeff - t_i)^2)
            # This is equivalent to solving (V^T W V) coeff = V^T W t
            W = np.diag(weights_huber)
            try:
                A = V_all.T @ W @ V_all
                b = V_all.T @ (weights_huber * t_smooth)
                coeff_leg_refined = np.linalg.solve(A + 1e-10 * np.eye(6), b)
            except np.linalg.LinAlgError:
                coeff_leg_refined = np.linalg.lstsq(V_all, t_smooth, rcond=None)[0]
            
            # Convert back to monomial coefficients
            x_eval_scaled2 = (x_eval - mu) / sigma
            V_eval2 = np.polynomial.legendre.legvander(x_eval_scaled2, 5)
            y_eval2 = V_eval2 @ coeff_leg_refined
            coeff_monomial_refined = np.polynomial.polynomial.polyfit(x_eval, y_eval2, 5)
            
            # Check if refined version is better
            pred_refined = np.polynomial.polynomial.polyval(x_test, coeff_monomial_refined)
            mse_refined = np.mean((pred_refined - t_test) ** 2)
            
            if mse_refined < best_score:
                best_coeff = coeff_monomial_refined
                best_score = mse_refined
                
        except Exception:
            pass
    
    if best_score == np.inf:
        # Fallback: simple polynomial fit
        try:
            mu = np.mean(x)
            sigma = np.std(x)
            if sigma < 1e-10:
                sigma = 1.0
            x_scaled = (x - mu) / sigma
            V = np.polynomial.legendre.legvander(x_scaled, 5)
            coeff_leg, _, _, _ = np.linalg.lstsq(V, t, rcond=None)
            
            x_min = np.min(x)
            x_max = np.max(x)
            x_eval = np.linspace(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min), 6)
            x_eval_scaled = (x_eval - mu) / sigma
            V_eval = np.polynomial.legendre.legvander(x_eval_scaled, 5)
            y_eval = V_eval @ coeff_leg
            best_coeff = np.polynomial.polynomial.polyfit(x_eval, y_eval, 5)
        except Exception:
            best_coeff = np.zeros(6)
    
    return best_coeff
