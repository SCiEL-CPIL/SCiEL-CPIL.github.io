import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        best_x = None
        best_f = float('inf')
        calls_made = 0
        dim = self.dim

        def eval_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                raise StopIteration()
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cmaes(x0, max_evals):
            nonlocal calls_made
            n = dim
            lam = 9
            mu = lam // 2
            alpha = 1.0 / np.sqrt(n)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            
            # Rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cmu = min(1 - cc, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2)**2 + mueff))
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            
            chi_lam = np.sqrt(lam) * (1 - 1 / (4 * n) + 1 / (24 * n**2))
            
            # State
            m = x0.copy()
            sigma = 1.0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            evals_left = max_evals - calls_made
            if evals_left <= 0:
                return

            while calls_made < max_evals:
                if time.time() > deadline:
                    break
                    
                # Generate population
                ys = []
                xs = []
                for i in range(lam):
                    if calls_made >= max_evals:
                        break
                    zi = C_sqrt @ np.random.standard_normal(n)
                    xi = m + sigma * zi
                    xi = np.clip(xi, -5, 5)
                    yi = eval_wrapper(xi)
                    ys.append(yi)
                    xs.append(xi)
                
                if len(xs) < mu:
                    break
                    
                # Sort
                idx = np.argsort(ys)
                xw = np.array([xs[i] for i in idx[:mu]])
                yw = np.array([ys[i] for i in idx[:mu]])
                
                # Update mean
                m_old = m.copy()
                m = np.dot(weights, xw.T)
                
                # Update paths
                # C_sqrt inverse update is expensive, but for n=5 it's fine
                # Actually, standard CMA-ES updates C_sqrt via SVD or Cholesky
                # Here we just update C and compute SVD each step for simplicity/robustness
                # In 5D, SVD is very fast.
                
                inv_c_sqrt = np.linalg.inv(C_sqrt)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_c_sqrt @ (m - m_old) / sigma
                if np.linalg.norm(ps) / chi_lam < (1 + cc) * np.sqrt(1 - (1 - cc)**2) + 1:
                    pc = (1 - cc) * pc + cc * (m - m_old) / sigma
                    C = (1 - cmu) * C + cmu * (1 / (1 - cc)) * np.outer(pc, pc)
                else:
                    pc = (1 - cc) * pc
                C = (1 - (cmu + cmu * (1 - cc) / cc)) * C + cmu * np.sum(weights[:, None] * np.outer(xw.T, xw.T), axis=1)
                
                # Update C_sqrt via SVD
                # C is symmetric positive definite
                try:
                    u, s, vh = np.linalg.svd(C)
                    C_sqrt = u @ np.diag(np.sqrt(s)) @ vh
                except:
                    C_sqrt = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_lam - 1))
                sigma = np.clip(sigma, 1e-8, 100)
                
                # Stopping criteria
                if sigma < 1e-8:
                    break
                if np.linalg.norm(m - m_old) < 1e-10:
                    break

        # Main strategy: Multi-start CMA-ES
        # First, do a quick random search to find a good starting point?
        # Or just start with several CMA-ES instances from random points.
        
        # Strategy:
        # 1. Use ~10% budget for initial random sampling to find a promising region.
        # 2. Use remaining budget for CMA-ES starting from the best point found.
        # 3. If CMA-ES stalls, restart from another random point or best point with different sigma.
        
        # Given budget 500, dim 5.
        # CMA-ES with lambda=9 needs ~9 evals per iteration.
        # 500 / 9 ~ 55 iterations. This is quite a lot for 5D.
        
        # Let's do:
        # Phase 1: Random search for 20 evals to get a decent start.
        # Phase 2: CMA-ES from the best point found in Phase 1.
        
        # Actually, Rastrigin is very rugged. A single CMA-ES might get stuck in a local minimum.
        # Multi-start is crucial.
        
        # Plan:
        # Split budget into chunks.
        # Chunk 1: 50 evals random search. Keep best 3 points.
        # Chunk 2: 150 evals CMA-ES from best of Chunk 1.
        # Chunk 3: 150 evals CMA-ES from 2nd best of Chunk 1 (or new random).
        # Chunk 4: 150 evals CMA-ES from 3rd best of Chunk 1 (or new random).
        
        # Simpler: Just run CMA-ES with a large lambda? No, lambda=9 is standard.
        # Let's use a "survivor" strategy.
        
        # 1. Randomly sample 10 points. (10 evals)
        # 2. Pick the best one. Run CMA-ES with budget 200.
        # 3. Pick the 2nd best one. Run CMA-ES with budget 150.
        # 4. Pick the 3rd best one. Run CMA-ES with budget 100.
        # 5. If budget remains, do more random restarts or refine best.
        
        # Let's implement this.
        
        if calls_made < self.budget:
            # Phase 1: Initial sampling
            n_init = min(10, self.budget - calls_made)
            init_xs = []
            init_fs = []
            for _ in range(n_init):
                if time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, dim)
                f = eval_wrapper(x)
                init_xs.append(x)
                init_fs.append(f)
            
            if not init_xs:
                return best_x, best_f
                
            # Sort by fitness
            init_order = np.argsort(init_fs)
            
            # Phase 2: CMA-ES on top candidates
            # We have remaining budget
            rem_budget = self.budget - calls_made
            
            # Allocate budget to top 3 candidates
            # 50%, 30%, 20% of remaining
            allocs = [
                int(rem_budget * 0.5),
                int(rem_budget * 0.3),
                int(rem_budget * 0.2)
            ]
            
            for i in range(min(3, len(init_xs))):
                if i >= len(allocs):
                    break
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x_start = init_xs[init_order[i]]
                max_evals = calls_made + allocs[i]
                cmaes(x_start, max_evals)
            
            # If budget still left, do a final refinement on the global best
            if calls_made < self.budget and best_x is not None:
                rem = self.budget - calls_made
                if rem > 20:
                    cmaes(best_x, self.budget)
            
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f
