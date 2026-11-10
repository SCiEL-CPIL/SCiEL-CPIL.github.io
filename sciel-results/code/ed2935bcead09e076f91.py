import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    
    Strategy: 
    Instead of a single global polynomial which extrapolates poorly, we use a local 
    weighted polynomial regression approach. We select a set of "knots" (quantiles of x) 
    and fit a low-degree polynomial in each local neighborhood. To satisfy the output 
    constraint of returning a single global polynomial, we use a clever trick:
    
    Actually, the contract says "return 6 finite real polynomial coefficients in ASCENDING 
    powers of local slow state x". It does NOT say we can return a piecewise function. 
    It says "construct(...) -> 6 finite real polynomial coefficients". 
    This implies the model MUST be a single global polynomial sum(c_i*x**i).
    
    Wait, let me re-read the contract carefully.
    "Fit a shared degree-five polynomial approximation ... approximating the unresolved 
    tendency as sum(c_i*x**i). ... Coefficients must be a single shared polynomial for 
    every cell and forcing."
    
    Okay, so I CANNOT return a piecewise function. I must return 6 coefficients for a 
    single global polynomial.
    
    The current program fits a global polynomial. The problem is that global polynomials 
    of degree 5 are prone to oscillations (Runge's phenomenon) and poor extrapolation 
    if the data range is wide or if the function is not well-approximated by a global 
    polynomial.
    
    How can I improve a global polynomial fit?
    1. Better conditioning: Use orthogonal polynomials (Chebyshev or Legendre) for the 
       fit, then convert back to the power basis. This improves numerical stability.
    2. Regularization: Ridge regression is already used. Maybe the regularization 
       strength is not optimal.
    3. Data preprocessing: The current program standardizes x. This is good.
    4. Model selection: The current program tries degrees 1-5 and various alphas.
    
    Let's look at the inspirations.
    Inspiration 1: Bayesian Linear Regression.
    Inspiration 2: Local RBF interpolation. But we can't return a local model.
    
    Wait, if I can't return a local model, how can I use local information?
    I can use local information to *select* the global polynomial parameters better, 
    or to *regularize* the fit in a way that penalizes oscillations.
    
    Alternative interpretation: Is it possible that the "local slow state x" implies 
    that the polynomial is applied locally? No, "single shared polynomial for every 
    cell".
    
    Let's reconsider the "Local Weighted Polynomial Regression" idea.
    If I fit a local polynomial at each point x_i, I get a non-parametric estimate 
    f_hat(x). This f_hat(x) is a smooth function. I can then fit a global degree-5 
    polynomial to this f_hat(x) curve. This is essentially a two-stage approach:
    1. Non-parametric smoothing (e.g., LOWESS or Nadaraya-Watson) to get a smooth 
       estimate of the true function.
    2. Parametric approximation of that smooth curve with a degree-5 polynomial.
    
    This decouples the "fitting the data" step from the "approximating with a 
    polynomial" step. The global polynomial will be the best L2 approximation of the 
    smoothed curve, which is likely to be smoother and less oscillatory than the 
    direct global polynomial fit to the noisy data.
    
    Let's try this strategy:
    1. Compute a local weighted linear/quadratic fit for each x in the training set 
       (or a grid) to get a smooth estimate y_smooth.
    2. Fit a global degree-5 polynomial to (x, y_smooth).
    3. Use cross-validation to select the bandwidth and local degree for step 1.
    
    This is materially different from the current program which does a single 
    global ridge regression.
    
    Implementation details:
    - Use a Gaussian kernel for local weighting.
    - Local degree: 1 or 2 (linear or quadratic).
    - Bandwidth: selected via CV.
    - The global fit is done on the smoothed data, which should be less noisy.
    
    Let's refine this.
    Step 1: For a given bandwidth h and local degree d, compute y_smooth for all x.
    Step 2: Fit a global polynomial of degree 5 to (x, y_smooth) using ridge regression.
    Step 3: Evaluate the global polynomial on a validation set of the original data.
    Step 4: Select h and d that minimize validation MSE.
    Step 5: Return the coefficients of the best global polynomial.
    
    This is efficient because:
    - Computing y_smooth can be done with a sliding window or vectorized operations.
    - The global fit is a single least squares problem.
    
    Let's code this.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 10:
        return np.array([np.mean(y)] + [0.0]*5, dtype=np.float64)

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    # Standardize x for better kernel behavior
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x < 1e-12:
        sigma_x = 1.0
    x_std = (x - mu_x) / sigma_x

    # We will fit a global polynomial in the original x basis, but use standardized x 
    # for the local weighting and the global fit to improve conditioning.
    # Actually, let's fit the global polynomial in the standardized basis, then convert.
    
    best_mse = np.inf
    best_coeffs_std = np.zeros(6)
    
    # Candidate bandwidths (in standardized units)
    # Smaller bandwidth = more local, larger = more global
    bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    local_degrees = [1, 2, 3]
    
    # Pre-sort x for efficient local windowing? 
    # Since we need to compute y_smooth for all points, and then fit a global poly,
    # we can just loop over bandwidths.
    
    # To save time, we'll subsample if n is very large
    if n > 5000:
        idx = np.random.RandomState(42).permutation(n)
        x_sub = x_std[idx[:5000]]
        y_sub = y[idx[:5000]]
        # We need to be careful: the final fit should be on the full data?
        # Or is 5000 enough? The task says "train on supplied... pairs".
        # Using a subsample for model selection is fine, but the final fit 
        # should ideally use all data. However, for speed, we might use subsample 
        # for everything if n is huge. Let's try to use full data for the final fit.
        # For now, let's just use the subsample for the search to keep it fast.
        x_search = x_sub
        y_search = y_sub
    else:
        x_search = x_std
        y_search = y

    n_search = len(x_search)
    
    # Simple 5-fold CV for selecting bandwidth and local degree
    K = 5
    fold_size = n_search // K
    indices = np.arange(n_search)
    
    for h in bandwidths:
        if time.time() > deadline:
            break
        for d_loc in local_degrees:
            if time.time() > deadline:
                break
            
            mse_sum = 0.0
            count = 0
            for k in range(K):
                lo = k * fold_size
                hi = lo + fold_size
                if k == K - 1:
                    hi = n_search
                
                val_idx = indices[lo:hi]
                train_idx = np.concatenate([indices[:lo], indices[hi:]])
                
                x_train = x_search[train_idx]
                y_train = y_search[train_idx]
                x_val = x_search[val_idx]
                y_val = y_search[val_idx]
                
                # Step 1: Compute smoothed y for training set
                # y_smooth[i] = local weighted poly fit at x_train[i] using x_train, y_train
                # This is O(N^2) if done naively. We need to be careful.
                # For N=5000, N^2 = 25M, which might be too slow in Python loop.
                # We can vectorize the kernel weights.
                
                # Efficient computation of local weighted fit:
                # For each i, w_j = exp(-0.5 * ((x_i - x_j)/h)^2)
                # Fit poly of degree d_loc to (x_j, y_j) with weights w_j.
                
                # To speed up, we can use a grid of x values for the smoothing, 
                # or just accept the O(N^2) cost if N is small enough.
                # 5000^2 = 25,000,000 operations. In NumPy, this is fast.
                
                # Compute kernel matrix K (N x N)
                # diff = x_train[:, None] - x_train[None, :]
                # K = np.exp(-0.5 * (diff / h) ** 2)
                
                # This uses O(N^2) memory. 5000^2 * 8 bytes = 200 MB. 
                # This is risky. Let's use a subsample of the training set for 
                # the local fit if N is large, or use a smaller N.
                
                # Let's reduce the search subsample to 2000 to be safe on memory/time.
                if n_search > 2000:
                    # Re-subsample for the search loop
                    pass # We already subsampled to 5000. Let's just use 2000 in the search.
                    # Actually, let's just use the first 2000 points for the CV search.
                    # This is a heuristic.
                    
                # Let's implement the local fit efficiently.
                # We will compute y_smooth for x_train.
                # To avoid O(N^2) memory, we can process in chunks or just use a 
                # smaller effective N for the search.
                
                # Let's use a smaller subsample for the CV search: 1000 points.
                if n_search > 1000:
                    idx_search = np.random.RandomState(k).permutation(n_search)
                    x_search_cv = x_search[idx_search[:1000]]
                    y_search_cv = y_search[idx_search[:1000]]
                else:
                    x_search_cv = x_search
                    y_search_cv = y_search
                
                n_cv = len(x_search_cv)
                
                # Split CV data into train/val for the local fit? 
                # No, the local fit is the model. We fit it on the CV-train data 
                # and evaluate on CV-val data.
                
                # But the local fit uses all training points to predict each point.
                # This is "leave-one-out" style if we use all points.
                # For CV, we should fit on the train fold and predict on the val fold.
                
                # So:
                # 1. Fit local model on x_train, y_train.
                # 2. Predict y_smooth_val for x_val using the local model.
                
                # To predict y_smooth_val, we need to do a local fit at each x_val 
                # using x_train, y_train.
                
                # This is O(N_val * N_train).
                # If N_train ~ 800, N_val ~ 200, this is 160,000 operations. Very fast.
                
                # Let's do this.
                y_smooth_val = np.zeros(len(x_val))
                
                for i, xv in enumerate(x_val):
                    # Compute weights for x_train relative to xv
                    diff = x_train - xv
                    w = np.exp(-0.5 * (diff / h) ** 2)
                    
                    # Fit local polynomial of degree d_loc
                    # V = [1, x, x^2, ...] for x_train
                    # We need to solve (V^T W V) c = V^T W y_train
                    # Where W = diag(w)
                    
                    # To improve conditioning, center x_train locally?
                    # Let's just use the raw x_train values for the local fit.
                    # Since x is standardized, the values are O(1).
                    
                    V = np.column_stack([x_train**k for k in range(d_loc + 1)])
                    
                    # Weighted least squares
                    # (V^T W V) c = V^T W y
                    # V^T W V = V.T @ (w[:, None] * V)
                    # V^T W y = V.T @ (w * y)
                    
                    Vw = V * w[:, None]
                    A = V.T @ Vw
                    b = V.T @ (w * y_train)
                    
                    # Add small ridge for stability
                    A += 1e-6 * np.eye(d_loc + 1)
                    
                    try:
                        c_local = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        c_local = np.zeros(d_loc + 1)
                    
                    # Predict at xv
                    pred = np.sum(c_local * (xv ** np.arange(d_loc + 1)))
                    y_smooth_val[i] = pred
                
                # Now we have y_smooth_val, which is the local model's prediction 
                # for the validation points.
                # But wait, the task requires a GLOBAL polynomial.
                # So we need to fit a global polynomial to (x_train, y_smooth_train) 
                # and then evaluate it on x_val.
                
                # Step 1b: Compute y_smooth_train for x_train
                y_smooth_train = np.zeros(len(x_train))
                for i, xv in enumerate(x_train):
                    diff = x_train - xv
                    w = np.exp(-0.5 * (diff / h) ** 2)
                    # Exclude the point itself? Leave-one-out?
                    # For simplicity, include it. The weight is 1, so it's dominated 
                    # by the local point.
                    V = np.column_stack([x_train**k for k in range(d_loc + 1)])
                    Vw = V * w[:, None]
                    A = V.T @ Vw
                    b = V.T @ (w * y_train)
                    A += 1e-6 * np.eye(d_loc + 1)
                    try:
                        c_local = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        c_local = np.zeros(d_loc + 1)
                    pred = np.sum(c_local * (xv ** np.arange(d_loc + 1)))
                    y_smooth_train[i] = pred
                
                # Step 2: Fit global polynomial of degree 5 to (x_train, y_smooth_train)
                # Use ridge regression for stability
                V_global = np.column_stack([x_train**k for k in range(6)])
                A_global = V_global.T @ V_global
                b_global = V_global.T @ y_smooth_train
                A_global += 1e-4 * np.eye(6)
                try:
                    c_global = np.linalg.solve(A_global, b_global)
                except np.linalg.LinAlgError:
                    c_global = np.zeros(6)
                
                # Step 3: Evaluate global polynomial on x_val
                V_val = np.column_stack([x_val**k for k in range(6)])
                pred_global = V_val @ c_global
                
                mse = np.mean((pred_global - y_val) ** 2)
                mse_sum += mse
                count += 1
            
            if count > 0:
                avg_mse = mse_sum / count
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    # Store the best hyperparameters
                    best_h = h
                    best_d_loc = d_loc
    
    # Final fit with best hyperparameters on full data
    if time.time() < deadline:
        # Compute y_smooth for full x using best_h and best_d_loc
        y_smooth_full = np.zeros(n)
        # To save time, subsample the training points for the local fit if n is large
        if n > 2000:
            idx_full = np.random.RandomState(42).permutation(n)
            x_fit_sub = x_std[idx_full[:2000]]
            y_fit_sub = y[idx_full[:2000]]
        else:
            x_fit_sub = x_std
            y_fit_sub = y
        
        n_fit = len(x_fit_sub)
        
        for i, xv in enumerate(x_std):
            diff = x_fit_sub - xv
            w = np.exp(-0.5 * (diff / best_h) ** 2)
            V = np.column_stack([x_fit_sub**k for k in range(best_d_loc + 1)])
            Vw = V * w[:, None]
            A = V.T @ Vw
            b = V.T @ (w * y_fit_sub)
            A += 1e-6 * np.eye(best_d_loc + 1)
            try:
                c_local = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                c_local = np.zeros(best_d_loc + 1)
            pred = np.sum(c_local * (xv ** np.arange(best_d_loc + 1)))
            y_smooth_full[i] = pred
        
        # Fit global polynomial to (x_std, y_smooth_full)
        V_global = np.column_stack([x_std**k for k in range(6)])
        A_global = V_global.T @ V_global
        b_global = V_global.T @ y_smooth_full
        A_global += 1e-4 * np.eye(6)
        try:
            c_global_std = np.linalg.solve(A_global, b_global)
        except np.linalg.LinAlgError:
            c_global_std = np.zeros(6)
        
        # Convert from standardized basis to original basis
        # p(x) = sum c_i ((x-mu)/sigma)^i
        # We need coefficients for sum d_k x^k
        # d_k = sum_{i=k}^{5} c_i * binom(i,k) * (-mu/sigma)^(i-k) * (1/sigma)^k
        # Wait, let's use the conversion function from the original code.
        coeffs_orig = np.zeros(6)
        for i in range(6):
            ci = c_global_std[i]
            inv_si = (1.0 / sigma_x) ** i
            for k in range(i + 1):
                # binom(i, k)
                Ck = 1.0
                if k > 0 and k < i:
                    mm = min(k, i - k)
                    num = 1
                    den = 1
                    for t in range(1, mm + 1):
                        num *= (i - mm + t)
                        den *= t
                    Ck = num / den
                coeffs_orig[k] += ci * Ck * ((-mu_x / sigma_x) ** (i - k)) * (1.0 / sigma_x ** k)
                # Correction: The term is c_i * ( (x-mu)/sigma )^i
                # = c_i * sigma^-i * (x-mu)^i
                # = c_i * sigma^-i * sum_{k=0}^i binom(i,k) x^k (-mu)^(i-k)
                # Coeff of x^k is c_i * sigma^-i * binom(i,k) * (-mu)^(i-k)
        
        # Let's re-calculate carefully
        coeffs_orig = np.zeros(6)
        for i in range(6):
            ci = c_global_std[i]
            for k in range(i + 1):
                # binom(i, k)
                Ck = 1.0
                if k > 0 and k < i:
                    mm = min(k, i - k)
                    num = 1
                    den = 1
                    for t in range(1, mm + 1):
                        num *= (i - mm + t)
                        den *= t
                    Ck = num / den
                coeffs_orig[k] += ci * (1.0 / sigma_x) ** i * Ck * ((-mu_x) ** (i - k))
        
        if np.all(np.isfinite(coeffs_orig)):
            return coeffs_orig.astype(np.float64)
    
    # Fallback: simple linear fit
    try:
        c_lin = np.polynomial.polynomial.polyfit(x[:100], y[:100], 1)
        best_coeffs = np.zeros(6)
        best_coeffs[:2] = c_lin
        return best_coeffs.astype(np.float64)
    except Exception:
        return np.array([np.mean(y)] + [0.0]*5, dtype=np.float64)
