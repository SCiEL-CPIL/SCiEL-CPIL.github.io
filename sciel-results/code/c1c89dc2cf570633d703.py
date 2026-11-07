"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = None
        best_fx = np.inf
        
        # Helper to update best
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
        # Phase 1: Initial Sampling to find a good starting point for CMA-ES
        # Spend a small fraction of the budget to identify the basin
        n_init = min(10, budget // 10)
        if n_init < 1:
            n_init = 1
            
        # Use a fixed seed for reproducibility in initial sampling if needed, 
        # but random is generally fine. We use np.random directly.
        init_x = np.random.uniform(-5, 5, (n_init, dim))
        for x in init_x:
            if n_calls >= budget:
                break
            fx = func(x)
            n_calls += 1
            update_best(x, fx)
            
        if best_x is None:
            return np.zeros(dim), 0.0

        # Phase 2: CMA-ES Optimization
        # Parameters for CMA-ES
        lam = 4  # Population size
        mu = lam // 2  # Number of weights
        
        # Calculate weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        # State variables
        x = best_x.copy()
        # Initial sigma: heuristic based on problem scale
        # For BBOB, a sigma of 1-2 is usually a good start.
        sigma = 1.5 
        
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        try:
            DC = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            DC = np.eye(dim)
            
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        stagnation_counter = 0
        best_fx_prev = best_fx
        
        # Main CMA-ES Loop
        while n_calls < budget:
            if time.time() > deadline - 5:
                break
                
            # Stagnation Check
            if best_fx == best_fx_prev:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fx_prev = best_fx
                
            # If stuck for a long time, try a small random perturbation or reset
            # This helps escape the "step" plateau if CMA-ES shrinks sigma too much
            if stagnation_counter > 30:
                # Reduce sigma slightly to explore finer details or perturb mean
                # If we are on a plateau, reducing sigma might help find the edge.
                # Alternatively, a random kick.
                if stagnation_counter > 60:
                    # Random restart of the local search around best
                    x = best_x + np.random.uniform(-0.1, 0.1, dim)
                    x = np.clip(x, -5, 5)
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    DC = np.eye(dim)
                    sigma = 0.5
                    stagnation_counter = 0
                else:
                    sigma *= 0.9
                    C = np.eye(dim) * 0.5 # Simplify covariance to isotropic
                    try:
                        DC = np.linalg.cholesky(C)
                    except:
                        DC = np.eye(dim) * np.sqrt(0.5)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)

            # Generate population
            Z = np.random.randn(lam, dim)
            Y = Z @ DC.T
            
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                xi = x + sigma * Y[i]
                xi = np.clip(xi, -5, 5)
                F[i] = func(xi)
                n_calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = xi.copy()
            
            if n_calls >= budget:
                break
                
            # Update CMA-ES state
            indices = np.argsort(F)
            Z_sorted = Z[indices]
            Y_sorted = Y[indices]
            
            z_bar = np.dot(weights, Z_sorted[:mu])
            mean_step = DC @ z_bar
            x = x + sigma * mean_step
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
            
            Y_sel = Y_sorted[:mu]
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += weights[i] * np.outer(Y_sel[i], Y_sel[i])
                
            h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
            
            C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
            C = (C + C.T) / 2
            
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 0.01 * np.eye(dim)
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    DC = np.eye(dim)
                    C = np.eye(dim)
                    # Reset state if covariance breaks
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)

        # Phase 3: Polish with Nelder-Mead if budget remains
        remaining = budget - n_calls
        if remaining > 10 and best_x is not None:
            try:
                # Use scipy's Nelder-Mead for final polish
                # maxfev is the number of function evaluations
                res = minimize(func, best_x, method='Nelder-Mead', 
                               options={'maxfev': remaining, 'maxiter': remaining, 
                                        'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
