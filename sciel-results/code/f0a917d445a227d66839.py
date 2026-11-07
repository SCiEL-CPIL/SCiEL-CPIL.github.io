<<<<<<< SEARCH
    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                val = float(func(x))
            except:
                return np.inf
            if np.isnan(val) or np.isinf(val):
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration with multiple random starts
        # Use about 30% of budget for exploration
        explore_budget = max(30, budget // 4)
        n_starts = min(10, explore_budget // 5)
        per_start_budget = explore_budget // n_starts

        candidates = []
        for i in range(n_starts):
            if calls >= explore_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            candidates.append((fx, x.copy()))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using (1+1)-ES on top candidates
        remaining_budget = budget - calls - 10  # reserve 10 calls for safety
        n_refine = min(3, len(candidates))
        per_refine_budget = max(10, remaining_budget // n_refine) if n_refine > 0 else 0

        for i in range(n_refine):
            if calls >= budget - 10 or time.time() > deadline:
                break
            x0 = candidates[i][1]
            fx0 = eval_func(x0)  # re-evaluate to ensure it's counted
            
            # (1+1)-ES refinement
            sigma = 1.0
            max_iter = per_refine_budget
            for _ in range(max_iter):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx0:
                    x0, fx0 = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        # Phase 3: Final local search around the best point found
        if best_x is not None and calls < budget - 5 and time.time() < deadline:
            x0 = best_x.copy()
            fx0 = eval_func(x0)
            sigma = 0.5
            for _ in range(min(50, budget - calls - 5)):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx0:
                    x0, fx0 = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 1e-7:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
=======
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return np.inf
            calls += 1
            try:
                val = float(func(x))
            except:
                return np.inf
            if np.isnan(val) or np.isinf(val):
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Implementation
        # Parameters for 5D, budget 500
        # lambda is typically 4 + floor(3 * log(N))
        lambda_ = 4 + int(3 * np.log(dim))
        # Initial sigma
        sigma = 1.0
        # Initial mean
        m = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalues and eigenvectors for C
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Step size control parameters
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 - 1 + 2. / (dim + 1.)) / ((dim + 2.) + lambda_ / 2.)))
        
        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weights for selection
        w = np.log(lambda_ / 2. + 0.5) - np.log(np.arange(1, lambda_ + 1))
        w = w / np.sum(w)
        mu = int(np.sum(w > 0))
        w = w[:mu]
        mu_eff = 1. / np.sum(w ** 2)
        
        # Initial evaluation of mean
        f_m = eval_func(m)
        
        while calls < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lambda_):
                # Sample from N(m, sigma^2 C)
                # Use Cholesky or Eigen decomposition
                # Since we update C, we maintain eigvecs/eigvals
                z = np.random.randn(dim)
                # Transform z using eigvecs and sqrt(eigvals)
                x = m + sigma * (eigvecs * np.sqrt(eigvals)) @ z
                # Clip to box? The problem says func clips, but for CMA-ES,
                # it's often better to let it go out of bounds if the function handles it,
                # or clip if the function is undefined outside.
                # Task says: "func(x) clips a finite length-5 vector to the box"
                # So we can pass unclipped x, or clip it. Clipping might distort the distribution.
                # Let's clip to be safe and consistent with the box constraint for the optimizer's internal state.
                x_clipped = np.clip(x, -5, 5)
                f_x = eval_func(x_clipped)
                population.append((f_x, x_clipped))
                
                if calls >= budget or time.time() > deadline:
                    break
            
            if len(population) < lambda_:
                break
            
            # Sort by fitness
            population.sort(key=lambda t: t[0])
            
            # Update mean
            m_new = np.sum([w[i] * population[i][1] for i in range(mu)], axis=0)
            
            # Update sigma
            # Calculate p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvecs.T @ (eigvecs @ m_new) / sigma - eigvecs.T @ (eigvecs @ m))
            # Note: The standard formula involves the normalized step.
            # Let's use a simpler robust update if the full CMA-ES is too complex/error-prone.
            # Given the budget is small (500) and dim is small (5), a simpler adaptive (1+1) or (mu+lambda) might be safer.
            
            # Actually, let's stick to a simpler, robust strategy:
            # Multi-start (1+1)-ES with adaptive sigma and restarts.
            # The previous code did this but maybe the restart logic was weak.
            # Let's implement a robust (1+1)-ES with a population of 1, but with aggressive sigma adaptation.
            
            # Since I already started the CMA-ES loop, let's finish it properly or abort.
            # CMA-ES is powerful but sensitive to implementation details.
            # Let's try a simpler approach: **DE (Differential Evolution)** or **PSO**?
            # No, for 5D, **CMA-ES** is best. Let's implement a simplified CMA-ES (CMa-ES).
            
            # However, writing a full CMA-ES in a SEARCH/REPLACE is risky due to complexity.
            # Let's revert to a very optimized **(1+1)-ES with multiple restarts**.
            # The key is to not waste calls on re-evaluating the same points.
            
            # Let's break the current CMA-ES attempt and use a cleaner (1+1)-ES with better restarts.
            break

        # Fallback: Simple robust (1+1)-ES with restarts
        # We will use the remaining budget.
        # Reset state for the new strategy if we broke out early.
        # Actually, let's just implement the (1+1)-ES strategy directly in the main loop to avoid mixed logic.
        
        # Re-implementation:
        # We will use a loop of (1+1)-ES runs.
        # Each run has a fixed budget or runs until sigma converges.
        
        # Clear previous partial work if any? No, we just didn't do much.
        # Let's implement a clean (1+1)-ES with restarts.
        
        # We need to be careful with `calls`.
        
        # Strategy:
        # 1. Start with a random point.
        # 2. Run (1+1)-ES with adaptive sigma.
        # 3. If no improvement for X steps, restart with a new random point.
        # 4. Keep the best point found so far.
        
        # Let's rewrite the whole block to be a clean (1+1)-ES with restarts.
        
        # Since I can't easily "unwrite" the code I just wrote in the SEARCH block,
        # I will provide the replacement code that is a complete, clean implementation.
        
        return best_x, best_f
>>>>>>> REPLACE

Wait, the previous replacement logic was messy. I will provide a clean, single replacement for the `__call__` method that implements a robust **Multi-Start (1+1)-ES**. This is simpler and often outperforms poorly tuned CMA-ES in low dimensions with small budgets. The key improvements over the current seed are:
1.  **No re-evaluation of cached points** (though we don't cache, we just don't waste calls on redundant checks).
2.  **Better restart logic**: Instead of fixed phases, we use a "no-improvement" counter to trigger restarts.
3.  **Adaptive Sigma**: More aggressive adaptation.
4.  **Time Guard**: Strict adherence to the deadline.

