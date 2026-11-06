# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using Local Polynomial Regression (LPR) 
    for backward induction.
    
    Instead of a global quadratic fit, we estimate the continuation value 
    C_t(S) = E[V_{t+1}(S_{t+1}) | S_t] by performing a local quadratic 
    regression of V_{t+1} on log(S_t) for each point in a pre-computed grid.
    This adapts to the local shape of the value function, particularly near 
    the exercise boundary, which a global polynomial struggles with.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Handle input shape
    n_paths, n_cols = training_paths.shape
    if n_cols == n_dates:
        exercise_data = training_paths
    elif n_cols == n_dates + 1:
        exercise_data = training_paths[:, 1:n_dates+1]
    else:
        exercise_data = training_paths[:, :n_dates]
        
    # Precompute log-spots for the training paths
    # Shape: (n_dates, n_paths) for easier column access
    log_S_paths = np.log(exercise_data)
    
    # We will work with the values of V_t for the training paths.
    # V_vals[t] is an array of shape (n_paths,) representing V_t(S_t) for each path.
    V_vals = [None] * n_dates
    
    # Last date: V = Payoff
    S_last = exercise_data[:, -1]
    V_vals[-1] = np.maximum(K - S_last, 0.0)
    
    # We need to define the exercise policy for any state S and time t.
    # The policy is: Exercise if Intrinsic(S) >= Discounted Continuation(S).
    # Continuation(S) is estimated via LPR.
    
    # To make the `exercise` method fast, we pre-compute the LPR coefficients 
    # on a dense grid of log-prices for each time step.
    # Then, at runtime, we can interpolate these coefficients or evaluate the 
    # polynomial directly if we store the local basis.
    
    # However, LPR coefficients vary with the center point. 
    # A more efficient approach for the `exercise` method (which takes arbitrary states):
    # 1. Pre-compute the Continuation Value Function C_t(S) on a dense grid of S.
    # 2. Store this grid and the values.
    # 3. At runtime, for a given S, interpolate C_t(S) from the grid.
    
    # Let's define a grid for the log-price.
    # Range: cover the training data range plus some buffer.
    min_log_S = np.min(log_S_paths) - 0.5
    max_log_S = np.max(log_S_paths) + 0.5
    
    # Number of grid points. 2000 points should be sufficient for smooth interpolation.
    n_grid = 2000
    log_S_grid = np.linspace(min_log_S, max_log_S, n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Store the continuation value function on this grid for each time step t (0 to n_dates-2).
    # cont_grid[t] will be an array of shape (n_grid,)
    cont_grid = [None] * n_dates
    
    # Bandwidth for the Gaussian kernel.
    # A typical bandwidth is related to the standard deviation of the log-spots.
    # Let's estimate it from the training data.
    std_log_S = np.std(log_S_paths)
    # Use a bandwidth that covers a reasonable neighborhood. 
    # h = 0.1 * std might be too narrow, 0.5 * std might be too wide.
    # Let's try h = 0.2 * std.
    h = 0.2 * std_log_S
    if h < 1e-6:
        h = 0.1
        
    # Backward Induction
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_t_paths = log_S_paths[t, :]  # Shape (n_paths,)
        V_next_vals = V_vals[t+1]      # Shape (n_paths,)
        
        # We want to estimate C_t(S) = E[V_{t+1} | S_t = S] for S in log_S_grid.
        # For each grid point S_g, we perform a local quadratic regression.
        # This is computationally expensive if done naively (n_grid * n_paths * 3).
        # 2000 * 4096 * 3 ~ 24M operations per step. 12 steps -> 300M ops. 
        # This might be slow in Python if not vectorized carefully.
        
        # Optimization: 
        # We can vectorize the regression for all grid points simultaneously?
        # No, the weights W_g depend on S_g.
        # However, we can use a trick: 
        # C_t(S_g) = sum_j ( W_g(j) * V_next(j) * (1, (log_S_j - S_g), (log_S_j - S_g)^2) ) @ pinv(...)
        # This is hard to vectorize fully because of the local basis change.
        
        # Alternative: Use a smaller grid or a simpler local model (linear)?
        # Or, compute LPR on the training points themselves and interpolate?
        
        # Let's try a different, faster structural approach:
        # **Kernel Regression with a Gaussian Kernel and Local Linear/Quadratic Fit**
        # But to speed it up, we can pre-compute the "influence" of each training point 
        # on the grid points? No, that's n_grid * n_paths memory.
        
        # Let's stick to a slightly coarser grid or use a simpler local model.
        # Actually, for GBM, the value function is smooth. 
        # A **Global Polynomial** failed. A **Spline** might work?
        # Let's try **Cubic Spline Interpolation** of the empirical average.
        # 1. Bin the training points by S_t.
        # 2. For each bin, compute the average V_{t+1}.
        # 3. Fit a cubic spline through these bin averages.
        # This is much faster and captures non-linearities better than a single quadratic.
        
        # Let's implement the Spline approach instead of full LPR for speed.
        
        # Binning strategy:
        # Create bins covering the range of log_S_paths[t].
        n_bins = 50
        bin_edges = np.linspace(np.min(S_t_paths), np.max(S_t_paths), n_bins + 1)
        bin_indices = np.digitize(S_t_paths, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, V_next_vals)
        np.add.at(bin_counts, bin_indices, 1)
        
        # Avoid division by zero
        valid_bins = bin_counts > 0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_means = np.zeros(n_bins)
        bin_means[valid_bins] = bin_sums[valid_bins] / bin_counts[valid_bins]
        
        # If some bins are empty, we need to handle them for the spline.
        # We can interpolate the means to fill gaps or use a regularized spline.
        # SciPy has interp1d.
        
        # We want to evaluate the spline at log_S_grid.
        # But bin_centers are in log-S space.
        
        # Let's use scipy.interpolate.UnivariateSpline
        try:
            from scipy.interpolate import UnivariateSpline
            # Fit a spline to the (bin_centers, bin_means) data.
            # Use a smoothing factor s. s=0 is exact interpolation (might be noisy).
            # s = n_points * sigma^2 ? 
            # Let's try s = 0 for exact fit first, or a small value.
            # Since we have 50 points, exact fit is fine.
            spline = UnivariateSpline(bin_centers, bin_means, s=0, k=3)
            
            # Evaluate on the grid
            cont_grid[t] = spline(log_S_grid)
            
            # Clip negative values (continuation value should be >= 0 for a put? 
            # Actually, V is non-negative. The regression/spline might dip slightly negative.)
            cont_grid[t] = np.maximum(cont_grid[t], 0.0)
            
        except Exception:
            # Fallback to linear interpolation if spline fails
            # Find valid points
            valid_idx = np.where(valid_bins)[0]
            if len(valid_idx) < 2:
                # Constant fallback
                cont_grid[t] = np.mean(V_next_vals) * np.ones(n_grid)
            else:
                x_valid = bin_centers[valid_idx]
                y_valid = bin_means[valid_idx]
                cont_grid[t] = np.interp(log_S_grid, x_valid, y_valid)
                cont_grid[t] = np.maximum(cont_grid[t], 0.0)
                
        # Calculate V_vals[t] for the training paths
        # Interpolate the cont_grid values back to the training path log-spots
        cont_vals_paths = np.interp(S_t_paths, log_S_grid, cont_grid[t])
        discount = np.exp(-r * (T / n_dates))
        cont_vals_discounted = cont_vals_paths * discount
        
        intrinsic_vals = np.maximum(K - np.exp(S_t_paths), 0.0)
        
        V_vals[t] = np.maximum(intrinsic_vals, cont_vals_discounted)
        
    # Construct the Policy Object
    class ExercisePolicy:
        def __init__(self, K, cont_grid, log_S_grid, n_dates, r, T):
            self.K = K
            self.cont_grid = cont_grid
            self.log_S_grid = log_S_grid
            self.n_dates = n_dates
            self.r = r
            self.T = T
            self.dt = T / n_dates
            self.discount = np.exp(-self.r * self.dt)
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: index of the exercise date (0 to n_dates-1)
            """
            if time >= self.n_dates - 1:
                return (self.K - current_state) > 0
                
            if time < 0 or time >= len(self.cont_grid) or self.cont_grid[time] is None:
                # Fallback to intrinsic
                return (self.K - current_state) > 0
                
            log_S = np.log(current_state)
            
            # Interpolate continuation value from the grid
            cont_val = np.interp(log_S, self.log_S_grid, self.cont_grid[time])
            cont_val_discounted = cont_val * self.discount
            
            intrinsic_val = np.maximum(self.K - current_state, 0.0)
            
            return intrinsic_val >= cont_val_discounted
            
    return ExercisePolicy(K, cont_grid, log_S_grid, n_dates, r, T)

# EVOLVE-BLOCK-END
