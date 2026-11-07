"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    Uses a population-based evolutionary strategy that adapts the step size (sigma)
    and the search direction (via covariance matrix C) based on the selected best
    individuals. This is robust to rotation and scaling of the landscape and
    effective for multimodal functions like Schaffers F7.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- global best bookkeeping -------------------------------------
        x_best = None
        f_best = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best, f_best, n_calls
            if f < f_best:
                f_best = f
                x_best = x.copy()
            n_calls += 1
            return f_best

        def check_deadline():
            return n_calls >= budget or time.time() > deadline

        # ---- CMA-ES Implementation ---------------------------------------
        def cma_es(x0, max_evals):
            """Run CMA-ES from x0, using up to max_evals evaluations."""
            nonlocal n_calls, x_best, f_best, deadline
            
            n = dim
            # Population size
            lambda_pop = 4 + int(3 * np.log(n))
            # Step size
            sigma = 1.0
            # Mean vector
            mean = x0.copy()
            
            # Covariance matrix
            C = np.eye(n)
            # Inverse square root of C (initially identity)
            C_inv_sqrt = np.eye(n)
            
            # Adaptation rates
            mu = lambda_pop // 2  # number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc - 5 / (n + mueff + 5))
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)))
            
            # Path variables
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # Damping for step size control
            damps = 1.0 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            damps = 1.0 / damps
            
            # Initial evaluation
            fvals = [func(mean)]
            record(mean, fvals[0])
            if check_deadline():
                return
            
            # Main loop
            while n_calls < budget and time.time() < deadline:
                # Sample population
                ys = []
                xs = []
                fs = []
                
                for i in range(lambda_pop):
                    if check_deadline():
                        return
                    # Sample from N(0, C)
                    z = rng.standard_normal(n)
                    y = C_inv_sqrt @ z
                    x = mean + sigma * y
                    x = np.clip(x, -5.0, 5.0)
                    f = func(x)
                    record(x, f)
                    ys.append(y)
                    xs.append(x)
                    fs.append(f)
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update mean
                y_w = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
                mean = mean + sigma * y_w
                
                # Update path variables
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y_w)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_calls // lambda_pop + 1))) < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - cs) ** 2)
                
                if hsig:
                    pc = (1 - cc) * pc + cc * (y_w / sigma)
                    
                # Update covariance matrix
                # Delta h term
                delta_h = (1 - hsig) * cc * (2 - cc)
                
                # Rank one update
                rank_one = np.outer(pc, pc)
                # Rank mu update
                rank_mu = np.sum([
                    weights[i] * np.outer(ys[i], ys[i]) 
                    for i in range(mu)
                ], axis=0)
                
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2.0
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(n)
                
                # Update inverse square root
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Update step size
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) - 1))
                
                # Restart conditions:
                # 1. Sigma too large or too small
                # 2. Mean has left the box significantly
                # 3. No improvement in last several generations
                if sigma > 100 or sigma < 1e-10:
                    break
                if np.any(np.abs(mean) > 100):
                    break
                
                # Check for stagnation in global best
                # (Handled by outer loop restarts, but CMA-ES itself continues)
                # We'll rely on the outer loop to restart if global best doesn't improve enough.

        # ---- Strategy: CMA-ES with periodic restarts ----------------------
        # Start with a diverse random point
        x0 = rng.uniform(-5.0, 5.0, dim)
        
        # Run CMA-ES for a chunk of the budget
        # CMA-ES is efficient, so we can run it for a significant portion.
        # Let's try running it for 40% of the budget, then restart from best if needed.
        chunk_budget = int(budget * 0.4)
        
        while n_calls < budget and time.time() < deadline:
            if check_deadline():
                break
            
            # Start from current global best (or initial x0 if first time)
            if x_best is not None:
                x_start = x_best.copy()
            else:
                x_start = x0
            
            # Run CMA-ES with limited budget
            # We pass a dummy max_evals, the function checks global n_calls
            cma_es(x_start, chunk_budget)
            
            if check_deadline():
                break
            
            # Check if we should restart
            # If we've used up the chunk, or if time is up
            # For simplicity, we just let the main loop check deadline.
            # However, CMA-ES might stop early due to convergence.
            # If it stops early, we should probably restart with a perturbation
            # or just break if we are close to budget.
            
            # Heuristic: If CMA-ES converged (sigma small or spread small), restart with perturbation
            # But we don't have easy access to sigma here.
            # Let's just run one long CMA-ES until budget is nearly exhausted.
            # CMA-ES is robust enough to handle the whole budget if initialized well.
            # However, Schaffers F7 is multimodal. A single run might get stuck.
            
            # Let's implement a simple restart policy:
            # If n_calls has increased by less than 10% in the last chunk, restart.
            # But for now, let's just run one instance of CMA-ES for the remaining budget.
            # The "restart" logic in CMA-ES is internal (sigma adaptation).
            # External restarts are useful for multimodality.
            
            # Let's do 3 restarts of CMA-ES, each with 1/3 of the budget.
            # But the loop above runs cma_es which consumes budget until deadline or internal break.
            # If cma_es breaks early, we should restart.
            
            # To force restarts, we can limit the evaluations per cma_es call.
            # Let's modify the strategy:
            # Run cma_es with a limited budget per phase.
            
            # Actually, the current cma_es runs until n_calls >= budget or time > deadline.
            # So the while loop will only execute once if cma_es consumes the budget.
            # If cma_es breaks early (e.g., sigma too small), the loop will run again.
            # Let's add a small perturbation on restart.
            
            if n_calls > 0 and x_best is not None:
                # Perturb best for next run
                x0 = x_best + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5.0, 5.0)
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
