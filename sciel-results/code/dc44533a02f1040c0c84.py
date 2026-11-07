"""Surrogate-assisted Bayesian Optimization for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Surrogate-assisted Bayesian Optimization.
    
    Strategy:
    - Phase 1: Random exploration to gather initial samples.
    - Phase 2: Fit a Gaussian Process surrogate to the observed data.
    - Phase 3: Use Expected Improvement (EI) acquisition to select the next point.
    - Handle time and budget constraints strictly.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325  # Safety margin for 330s limit
        n = self.dim
        budget = self.budget
        
        # State
        X = []  # List of evaluated points
        Y = []  # List of function values
        best_f = float('inf')
        best_x = None
        calls_made = 0
        
        # Phase 1: Random Exploration
        # Allocate ~20% of budget for initial exploration to build a good surrogate
        n_explore = max(10, int(0.2 * budget))
        
        # Generate initial random points uniformly in [-5, 5]^n
        # Also include the center and some points near the center for better coverage
        X_init = []
        # Center
        X_init.append(np.zeros(n))
        # Random uniform
        for _ in range(n_explore - 1):
            x = np.random.uniform(-5, 5, size=n)
            X_init.append(x)
        
        # Evaluate initial points
        for x in X_init:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            f = func(x)
            calls_made += 1
            X.append(x)
            Y.append(f)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Surrogate-Assisted Optimization
        # We will fit a GP and use EI to select the next point.
        # To keep it fast, we re-fit the GP every few steps or use a simple kernel.
        
        # GP Parameters
        # Use a Matern 5/2 kernel for smoothness
        # Standardize inputs and outputs for better numerical stability
        
        def compute_ei(X_train, Y_train, x_test, best_y, sigma_y=1.0):
            """Compute Expected Improvement for a single test point."""
            # Simplified GP prediction
            # This is a placeholder for a full GP implementation.
            # For speed, we use a simple local model or a reduced GP.
            # Given the small dimension (5) and budget (500), a full GP is feasible but needs care.
            
            # Let's implement a simple GP with Matern 5/2 kernel.
            # To be fast, we might limit the number of points used in the GP to the most recent or best ones.
            
            # Standardize Y
            y_min = np.min(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_std = (Y_train - y_min) / y_std
            
            # Kernel
            def kernel(X1, X2):
                # Matern 5/2
                d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2))
                sqrt3d = np.sqrt(3) * d
                K = (1 + sqrt3d + (sqrt3d**2)/3.0) * np.exp(-sqrt3d)
                return K
            
            # Length scale parameter
            l = 1.0
            # Signal variance
            v = 1.0
            # Noise
            noise = 1e-6
            
            K = kernel(X_train, X_train) + noise * np.eye(len(X_train))
            Ks = kernel(X_train, x_test.reshape(1, -1))
            
            try:
                alpha = np.linalg.solve(K, Y_std)
                mu = Ks[0] @ alpha
                var = v - Ks[0] @ alpha
                if var < 0:
                    var = 0
                sigma = np.sqrt(var)
                
                # Unstandardize prediction
                mu_unstd = mu * y_std + y_min
                sigma_unstd = sigma * y_std
                
                # EI
                improvement = mu_unstd - best_y
                if sigma_unstd < 1e-8:
                    return improvement
                z = improvement / sigma_unstd
                from scipy.stats import norm
                ei = improvement * norm.cdf(z) + sigma_unstd * norm.pdf(z)
                return ei
            except:
                return 0.0

        # To avoid the high cost of solving the linear system for every candidate point,
        # we will generate a batch of candidate points and select the one with highest EI.
        
        # Number of candidate points to evaluate for EI in each step
        n_candidates = 50
        
        # We will re-fit the GP every `refit_interval` steps to balance cost and accuracy
        # Or simply fit it every step since n is small (5) and points are up to 500.
        # Solving a 500x500 system is fast (~ms).
        
        # Let's use a simpler approach:
        # 1. Generate candidates around the current best point and random points.
        # 2. Evaluate EI for each.
        # 3. Pick the best.
        
        # However, the `compute_ei` function above is called n_candidates times per step,
        # and each call involves a matrix solve.
        # 500 steps * 50 candidates * (solve 500x500) = 25000 solves.
        # This might be too slow in Python loops.
        
        # Optimization: Pre-compute K_inv * Y once per step, then EI is just a dot product.
        
        def fit_gp(X_train, Y_train):
            """Fit GP and return pre-computed terms for fast EI evaluation."""
            y_min = np.min(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_std = (Y_train - y_min) / y_std
            
            def kernel(X1, X2):
                d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2))
                sqrt3d = np.sqrt(3) * d
                K = (1 + sqrt3d + (sqrt3d**2)/3.0) * np.exp(-sqrt3d)
                return K
            
            l = 1.0
            v = 1.0
            noise = 1e-6
            
            K = kernel(X_train, X_train) + noise * np.eye(len(X_train))
            
            try:
                # Cholesky decomposition for stability and speed
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except:
                # Fallback to pinv if Cholesky fails
                try:
                    alpha = np.linalg.lstsq(K, Y_std, rcond=None)[0]
                    L = None
                except:
                    return None, y_min, y_std
            
            return (L, alpha, y_min, y_std, v)

        def compute_ei_fast(gp_params, x_test, best_y):
            """Compute EI using pre-computed GP parameters."""
            if gp_params is None:
                return 0.0
            L, alpha, y_min, y_std, v = gp_params
            
            def kernel_row(X_train, x_test):
                d = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
                sqrt3d = np.sqrt(3) * d
                K = (1 + sqrt3d + (sqrt3d**2)/3.0) * np.exp(-sqrt3d)
                return K
            
            # Note: We need X_train here, so we must pass it or store it.
            # Let's store X_train in the closure or pass it.
            # For simplicity in this structure, let's just pass X_train to the function.
            pass

        # Revised plan:
        # Since we need X_train for the kernel calculation in EI, and we want speed,
        # let's implement a simpler acquisition function that doesn't require a full GP solve per candidate,
        # or batch the candidates.
        
        # Given the constraints, let's use a simpler surrogate:
        # Inverse Distance Weighting (IDW) or a simple Kriging approximation.
        # Or, use a GP but only evaluate EI on a small set of candidates.
        
        # Let's stick to the GP but optimize the loop.
        # We will re-fit the GP every step.
        # We will generate 20 candidates.
        
        # To make it robust, let's implement the GP fit and EI evaluation carefully.
        
        # Helper: Kernel function
        def kernel_matern(X1, X2):
            d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2))
            sqrt3d = np.sqrt(3) * d
            K = (1 + sqrt3d + (sqrt3d**2)/3.0) * np.exp(-sqrt3d)
            return K

        # Main loop for surrogate phase
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Fit GP on current data
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            y_min = np.min(Y_arr)
            y_std = np.std(Y_arr)
            if y_std < 1e-8:
                y_std = 1.0
            Y_std = (Y_arr - y_min) / y_std
            
            K = kernel_matern(X_arr, X_arr) + 1e-6 * np.eye(len(X_arr))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except:
                # If GP fails, fall back to random exploration
                x = np.random.uniform(-5, 5, size=n)
                f = func(x)
                calls_made += 1
                X.append(x)
                Y.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                continue
            
            # Generate candidates
            # Mix of:
            # 1. Points near the current best (exploitation)
            # 2. Random points (exploration)
            # 3. Points from a coarse grid or Latin Hypercube (coverage)
            
            candidates = []
            
            # Exploitation: Around best_x
            for _ in range(10):
                x_cand = best_x + np.random.normal(0, 0.5, size=n)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # Exploration: Random
            for _ in range(10):
                x_cand = np.random.uniform(-5, 5, size=n)
                candidates.append(x_cand)
            
            # Coverage: Latin Hypercube-like
            for _ in range(10):
                x_cand = np.random.uniform(-5, 5, size=n)
                candidates.append(x_cand)
            
            # Evaluate EI for each candidate
            # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
            # Note: We are minimizing, so EI = (f_best - mu) * ...
            
            best_ei = -np.inf
            best_cand = None
            
            for x_cand in candidates:
                # Compute GP prediction at x_cand
                Ks = kernel_matern(X_arr, x_cand.reshape(1, -1))[0]
                
                # mu = Ks @ alpha
                mu_std = Ks @ alpha
                # var = v - Ks @ alpha
                # We need to compute Ks @ alpha carefully.
                # var = v - Ks @ L_inv @ L_inv.T @ Y_std
                # We have alpha = L_inv @ L_inv.T @ Y_std
                # So var = v - Ks @ alpha
                var = 1.0 - Ks @ alpha
                if var < 0:
                    var = 0
                sigma = np.sqrt(var) * y_std
                
                # Unstandardize
                mu = mu_std * y_std + y_min
                
                # Compute EI
                improvement = best_f - mu
                if sigma < 1e-8:
                    ei = improvement
                else:
                    z = improvement / sigma
                    from scipy.stats import norm
                    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                
                if ei > best_ei:
                    best_ei = ei
                    best_cand = x_cand
            
            if best_cand is None:
                break
            
            # Evaluate the best candidate
            f = func(best_cand)
            calls_made += 1
            X.append(best_cand)
            Y.append(f)
            
            if f < best_f:
                best_f = f
                best_x = best_cand.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
