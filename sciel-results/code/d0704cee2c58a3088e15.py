import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        def _time_ok():
            return time.time() - start_time < time_limit

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls
            if calls >= budget or not _time_ok():
                return np.inf
            try:
                fx = func(x)
                calls += 1
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        def cma_es(x0, sigma0, max_calls, rng):
            nonlocal calls, best_x, best_f
            
            # CMA-ES Parameters
            n = dim
            # Population size
            lam = int(4 + 3 * np.log(n))
            # Weights
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            
            # Strategy parameters
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1)) + cs
            sigma_path = (1 - cs) ** (1 / max_calls)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # State
            xmean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial population
            # Generate initial points
            ys = rng.normal(size=(lam, n))
            zs = np.dot(ys, np.linalg.cholesky(C).T)
            xs = [xmean + sigma * z for z in zs]
            
            # Evaluate initial population
            fs = np.array([safe_eval(x) for x in xs])
            
            # Update global best
            for i in range(lam):
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = xs[i].copy()
            
            # Sort indices
            indices = np.argsort(fs)
            xs = [xs[i] for i in indices]
            fs = fs[indices]
            
            # Main loop
            while calls < budget and _time_ok():
                # Update mean
                xmean_old = xmean.copy()
                xmean = np.sum(weights[:, None] * np.array(xs[:mu]), axis=0)
                
                # Update step size (CSA)
                invsqrtC = np.linalg.inv(np.sqrtm(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xmean_old) / sigma
                if np.linalg.norm(ps) / chi_n < (1.4 + 1 / cs) * np.sqrt(1 - (1 - cs) ** (2 * (calls // lam + 1))):
                    sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance matrix
                diff = np.array(xs[:mu]) - xmean_old
                delta_sig = np.linalg.norm(ps) / chi_n - 1
                C = (1 + c1 * delta_sig * cs * (2 - cs)) * C
                C += c1 * np.outer(ps, ps)
                C -= cmu * np.sum(weights[:, None, None] * diff[:, None, :] * diff[:, :, None], axis=0)
                
                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Update evolution path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * invsqrtC @ (xmean - xmean_old) / sigma
                
                # Generate new population
                ys = rng.normal(size=(lam, n))
                zs = np.dot(ys, np.linalg.cholesky(C).T)
                xs = [xmean + sigma * z for z in zs]
                
                # Evaluate
                fs = np.array([safe_eval(x) for x in xs])
                
                # Update global best
                for i in range(lam):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xs[i].copy()
                
                # Sort
                indices = np.argsort(fs)
                xs = [xs[i] for i in indices]
                fs = fs[indices]
                
                # Convergence check
                if fs[-1] - fs[0] < 1e-12 * (1 + abs(fs[0])):
                    break

        # Strategy:
        # 1. Initial broad sampling to identify promising regions (especially for multimodal).
        # 2. Run CMA-ES from the best few initial points, allocating budget to them.
        # 3. CMA-ES is robust and adapts to the landscape, handling both unimodal and multimodal (via restarts/multi-start) well.
        
        n_init = 15
        starts = []
        
        for i in range(n_init):
            if not _time_ok() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_eval(x)
            starts.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            return np.zeros(dim), 0.0

        # Sort starts by fitness
        starts.sort(key=lambda t: t[1])
        
        # Allocate remaining budget to top candidates
        # We'll run CMA-ES on the top 3 starts, giving each a portion of the budget.
        # Or, run one long CMA-ES from the best start, and if it stagnates, restart from others.
        # Given the budget (300), and CMA-ES efficiency, running from the best start is usually sufficient for unimodal.
        # For multimodal, we might need multiple restarts.
        
        # Let's try a strategy: Run CMA-ES from the best start with a large portion of the budget.
        # If it doesn't improve significantly, try the next best.
        
        # Heuristic: Use 60% of remaining budget for the best, 20% for second, 20% for third.
        # Actually, let's just run CMA-ES from the best start until budget is nearly exhausted or convergence.
        # CMA-ES is very efficient. 300 calls in 5D is enough to find a very good solution if we are in the right basin.
        # The risk is being in a local minimum for multimodal functions.
        
        # To mitigate this, we can do a "restart" strategy:
        # Run CMA-ES from best start with budget B1.
        # If no improvement for last B2 calls, restart from second best with remaining budget.
        
        # Simpler: Just run CMA-ES from the best start. If it's multimodal, the initial sampling (15 points) might have found a good basin.
        # If not, we might be stuck. But 15 random points in 5D is sparse.
        
        # Let's do a hybrid:
        # 1. Run CMA-ES from best start with 70% of remaining budget.
        # 2. If budget remains, run CMA-ES from second best start with remaining.
        
        remaining_budget = budget - calls
        if remaining_budget <= 10:
            return best_x, best_f
            
        # Run CMA-ES from best start
        # Use a reasonable initial sigma, e.g., 1.0 or based on box size
        sigma0 = 1.0
        max_calls_1 = int(remaining_budget * 0.7)
        
        cma_es(starts[0][0], sigma0, max_calls_1, rng)
        
        # If we have budget left, try second best
        if calls < budget - 10 and _time_ok() and len(starts) > 1:
            x_start2, f_start2 = starts[1]
            # Only if it's distinct
            if not np.allclose(starts[0][0], x_start2):
                max_calls_2 = budget - calls
                cma_es(x_start2, sigma0, max_calls_2, rng)
                
        # If we still have budget, try third best
        if calls < budget - 10 and _time_ok() and len(starts) > 2:
            x_start3, f_start3 = starts[2]
            if not np.allclose(starts[0][0], x_start3) and not np.allclose(starts[1][0], x_start3):
                max_calls_3 = budget - calls
                cma_es(x_start3, sigma0, max_calls_3, rng)

        return best_x, best_f
