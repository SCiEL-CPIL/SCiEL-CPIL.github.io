"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive ES with hierarchical budget allocation and refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(123)
        calls_used = [0]
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if calls_used[0] >= self.budget:
                raise TimeoutError()
            calls_used[0] += 1
            return func(x)

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Broad random sampling to find good basins
        n_initial = 20
        candidates = []
        for i in range(n_initial):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break
            # Mix of broader and centered starts
            if i < n_initial // 2:
                x = rng.uniform(-4, 4, self.dim)
            else:
                x = rng.uniform(-2, 2, self.dim)
            try:
                f = safe_func(x)
            except TimeoutError:
                break
            candidates.append((f, x.copy()))
            update_best(x, f)

        if not candidates:
            x0 = np.zeros(self.dim)
            try:
                f0 = safe_func(x0)
            except TimeoutError:
                return np.zeros(self.dim), float('inf')
            candidates.append((f0, x0.copy()))
            update_best(x0, f0)

        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES with periodicity-aware restarts
        n_local = min(5, len(candidates))
        remaining_budget = self.budget - calls_used[0]

        if remaining_budget <= 0:
            return best_x, best_f

        # Allocate budget: top 3 get more, rest get less
        allocations = []
        for i in range(n_local):
            if i < 3:
                alloc = max(40, remaining_budget // 3)
            else:
                alloc = max(20, remaining_budget // 6)
            allocations.append(alloc)

        # Adjust to fit within remaining budget
        total_alloc = sum(allocations)
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            allocations = [int(a * scale) for a in allocations]
            diff = remaining_budget - sum(allocations)
            allocations[0] += diff

        for i in range(n_local):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break

            x = candidates[i][1].copy()
            f = candidates[i][0]
            local_budget = allocations[i]
            
            # CMA-ES parameters
            pop_size = 10
            sigma = 1.0
            C = np.eye(self.dim)  # covariance matrix
            C_inv = np.eye(self.dim)
            C_sqrt = np.eye(self.dim)
            
            # Evolution paths
            ps = np.zeros(self.dim)
            cs = 1 / (pop_size + 3)
            cmu = min(0.1, 2 / ((self.dim + 1)**2 - 2))
            cmu_sigma = min(0.1, 2 / ((self.dim + 1)**2 - 2))
            
            # Update C_sqrt via Cholesky
            C_sqrt = np.linalg.cholesky(C)
            
            no_improve = 0
            patience = 40
            restart_count = 0
            max_restarts = 3
            
            for iteration in range(local_budget // pop_size + 1):
                if calls_used[0] >= self.budget or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for j in range(pop_size):
                    if calls_used[0] >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(self.dim)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except TimeoutError:
                        break
                    offspring.append((fy, y.copy()))
                
                if len(offspring) < pop_size // 2:
                    break
                
                # Sort offspring
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                if offspring[0][0] < f:
                    f = offspring[0][0]
                    update_best(x, f)
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Check for stagnation and restart
                if no_improve >= patience:
                    if restart_count < max_restarts:
                        restart_count += 1
                        # Restart with larger sigma centered on best
                        sigma = 2.0
                        C = np.eye(self.dim)
                        C_inv = np.eye(self.dim)
                        C_sqrt = np.linalg.cholesky(C)
                        ps = np.zeros(self.dim)
                        no_improve = 0
                    else:
                        break
                
                # Early termination if converged
                if sigma < 1e-5:
                    break
                
                # Update mean
                mu = pop_size // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1 / np.sum(weights**2)
                
                x_new = np.zeros(self.dim)
                for j in range(mu):
                    x_new += weights[j] * offspring[j][1]
                x = x_new
                
                # Update evolution path ps
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) * (C_sqrt.T @ (x_new - x)) / sigma
                
                # Update sigma
                chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*dim**2))
                sigma *= np.exp(cm / (pop_size + 2) * (np.linalg.norm(ps)/chi_n - 1))
                sigma = max(sigma, 1e-6)
                
                # Update covariance matrix
                delta_sig = (1 + 2*cmu_sigma*(np.linalg.norm(ps)**2/chi_n**2 - 1)) * cs
                rank_mu = np.zeros((self.dim, self.dim))
                for j in range(mu):
                    rank_mu += weights[j] * np.outer(offspring[j][1] - x, offspring[j][1] - x)
                
                C = (1 + cs*delta_sig - cs - cmu) * C + cmu * rank_mu
                C_inv = np.linalg.inv(C)
                
                # Update C_sqrt via Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    C_inv = np.eye(self.dim)
                    C_sqrt = np.eye(self.dim)
                    break

        # Phase 3: Final refinement on the best point found
        if best_x is not None and calls_used[0] < self.budget and time.time() < deadline:
            remaining = self.budget - calls_used[0]
            if remaining > 10:
                x = best_x.copy()
                fx = best_f
                sigma = 0.15
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    try:
                        fy = safe_func(y)
                    except TimeoutError:
                        break
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.1, 0.5)
                    else:
                        sigma *= 0.95
                    update_best(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
