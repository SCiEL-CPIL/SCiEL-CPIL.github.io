"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget
        
        if budget < 20:
            # Fallback for very small budgets
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            calls_made += 1
            return func(x)

        # Phase 1: Initial sampling to find a good starting point
        # Use 10-20 evaluations for this
        n_init = min(20, budget // 5)
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, n)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf

        # Phase 2: CMA-ES
        # Parameters for CMA-ES
        lam = 4 + int(3 * np.log(n)) # lambda = 4 + 3*ln(5) = 9
        mu = lam // 2 # mu = 4
        
        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log((lam + 1) / 2) - np.log(i + 1)
        weights /= weights.sum()
        mueff = (weights.sum() ** 2) / np.sum(weights ** 2)
        
        # Adaptation rates
        csigma = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Initial state
        m = best_x.copy()
        sigma = 0.5 # Initial step size, tuned for [-5,5] box
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        hsigma = np.sqrt((1 - cc) * (2 / (1 + cc)) + 1)
        
        # Pre-compute Cholesky factorization update helper
        def cholesky_update(C, w, y, p):
            # Rank-mu update
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - cc) * cc * C) + cmu * sum(wi * yi @ yi.T)
            # This is simplified. We will just recompute or use a stable update.
            # For n=5, we can just compute the new C and regularize.
            pass

        # Main Loop
        while calls_made < budget and time.time() < deadline:
            # Generate offspring
            Y = rng.standard_normal((lam, n))
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C = C + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(C)
            
            X = np.zeros((lam, n))
            for i in range(lam):
                X[i] = m + sigma * (Y[i] @ L)
                
            # Evaluate
            F = np.full(lam, np.inf)
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                F[i] = safe_call(X[i])
                
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update Mean
            m = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update Step Size
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*cc)*inv_chol(C) @ (m - m_old) / sigma
            # We need m_old. Let's store it.
            # Actually, standard CMA-ES updates ps using the difference of means.
            # Let's implement the standard update.
            
            # This implementation is getting complex. 
            # Alternative: Use a simpler but effective (1+1)-ES with CSA.
            # Given the risk of bugs in CMA-ES, I will switch to a robust (1+1)-ES with CSA.
            
            break # Stop CMA-ES attempt, switch to (1+1)-ES with CSA

        # Phase 2 (Revised): (1+1)-ES with Cumulative Step-size Adaptation (CSA)
        # This is safer and still very effective for n=5.
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        ps = np.zeros(n)
        
        # CSA parameters
        # c_sigma ~ 1/(n+3+2/sqrt(n))
        c_sigma = 1.0 / (n + 3 + 2.0 / np.sqrt(n))
        # d_sigma ~ 1 + 2*max(0, sqrt((mueff-1)/(n+1))-1) + csigma
        # For (1+1), mueff=1, so d_sigma ~ 1 + c_sigma
        d_sigma = 1.0 + c_sigma
        # h_sigma ~ 1.4 + 2/((n+1))
        h_sigma = 1.4 + 2.0 / (n + 1)
        
        # Path for step size adaptation
        # We need to track the path of normalized steps
        
        while calls_made < budget and time.time() < deadline:
            z = rng.standard_normal(n)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = safe_call(x_trial)
            if f_trial is None:
                break
            
            if f_trial < fx:
                # Success
                x = x_trial
                fx = f_trial
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Update path
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*cc)*z
                # For (1+1), cc is not directly used in the same way, but we adapt sigma based on ||ps||
                # Simplified CSA:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma)) * z
                norm_ps = np.linalg.norm(ps)
                # Compare norm_ps to expected value of norm of standard normal
                # E[||z||] = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
                # For n=5, E[||z||] ~ 2.08
                e_norm = np.sqrt(2) * np.exp(0.5 * (np.log(np.sqrt(2 * np.pi)) + np.sum(np.log(np.arange(1, n + 1)) - 0.5 * np.log(np.arange(1, n + 1))))) # Approximation
                # Actually, for n=5, E[||z||] is approx 2.08
                e_norm = 2.08
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / e_norm - 1))
            else:
                # Failure
                pass
                
            sigma = np.clip(sigma, 1e-8, 5.0)
            
            # If stuck, try a restart
            # Simple heuristic: if no improvement in last 50 calls, restart near best
            # We can track this with a counter.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
