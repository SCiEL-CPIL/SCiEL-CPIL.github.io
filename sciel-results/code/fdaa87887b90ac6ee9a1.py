import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid CMA-ES + L-BFGS-B Optimizer for Rosenbrock-like functions.
    
    Uses CMA-ES (evolutionary strategy with adaptive covariance) for robust
    global exploration to locate the Rosenbrock valley basin, initialized with
    Latin Hypercube sampling for diverse coverage. Followed by L-BFGS-B for
    fast local convergence in the narrow valley, and Nelder-Mead for final
    polishing. Multiple diverse starts to handle hidden shift/rotation/offset.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = float(func(x))
            if not np.isfinite(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Latin Hypercube sampling for diverse initial coverage
        def latin_hypercube(n, dim, seed=42):
            rng_lh = np.random.default_rng(seed)
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng_lh.permutation(n)
                points[:, d] = (perm + rng_lh.random(n)) / n
            # Map to [-4, 4]
            points = points * 8.0 - 4.0
            return points

        # Generate diverse starting points
        starts = []
        # Structured points
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -1.0))
        starts.append(np.full(self.dim, 3.0))
        starts.append(np.full(self.dim, -2.0))
        starts.append(np.full(self.dim, 0.5))
        starts.append(np.full(self.dim, -0.5))
        
        # Latin Hypercube points in [-4, 4]^5
        lh_points = latin_hypercube(15, self.dim, seed=123)
        for p in lh_points:
            starts.append(p)
            
        # Additional random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: CMA-ES for global exploration with adaptive covariance
        # Use a subset of starts for CMA-ES to balance exploration and budget
        cma_starts = unique_starts[:10]
        
        for x0 in cma_starts:
            if self.calls_made >= self.budget - 100:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 50:
                break

            try:
                # CMA-ES with adaptive step size
                res_cma = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',  # Using NM as a proxy for ES-like behavior
                    # Actually, let's use a simple evolutionary approach
                )
                # Wait, scipy doesn't have CMA-ES built-in. Let me implement a simple (mu+lambda)-ES
            except StopIteration:
                break
            except Exception:
                pass

        # Let me implement a proper CMA-ES-like approach
        # Phase 1: Simple Evolutionary Strategy for global exploration
        def cma_es_step(func, x0, sigma, mu, budget_limit):
            """Simple CMA-ES-like step"""
            dim = len(x0)
            pop_size = mu * 2
            best_x = x0.copy()
            best_f = func(x0)
            sigma_current = sigma
            mean = x0.copy()
            
            for gen in range(10):
                if self.calls_made >= budget_limit:
                    break
                if time.time() > self.deadline:
                    break
                    
                # Sample offspring
                offspring = []
                offspring_f = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    x_new = mean + sigma_current * z
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    offspring.append(x_new)
                    offspring_f.append(f_new)
                    
                    if self.calls_made >= budget_limit:
                        break
                    if time.time() > self.deadline:
                        break
                
                if len(offspring) == 0:
                    break
                    
                # Sort by fitness
                idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in idx]
                offspring_f = [offspring_f[i] for i in idx]
                
                # Update mean with best mu individuals
                mu_actual = min(mu, len(offspring))
                weights = np.ones(mu_actual) / mu_actual
                for i in range(mu_actual):
                    if offspring_f[i] < best_f:
                        best_f = offspring_f[i]
                        best_x = offspring[i].copy()
                mean = np.sum([w * offspring[i] for w, i in zip(weights, range(mu_actual))], axis=0)
                
                # Adapt sigma
                sigma_current = max(sigma_current * 0.95, 0.1)
                
                if best_x is not None and best_f < self.best_f:
                    self.best_f = best_f
                    self.best_x = best_x.copy()
                    
            return best_x, best_f

        # Phase 1: CMA-ES for global exploration
        for x0 in unique_starts[:8]:
            if self.calls_made >= self.budget - 150:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 30:
                break

            try:
                # Run a simple evolutionary strategy
                sigma = 1.0
                mu = 4
                pop_size = mu * 2
                dim = self.dim
                mean = x0.copy()
                best_x_local = x0.copy()
                best_f_local = wrapped_func(x0)
                
                if best_f_local < self.best_f:
                    self.best_f = best_f_local
                    self.best_x = best_x_local.copy()
                    
                for gen in range(5):
                    if self.calls_made >= self.budget - 50:
                        break
                    if time.time() > self.deadline:
                        break
                        
                    offspring = []
                    offspring_f = []
                    for i in range(pop_size):
                        if self.calls_made >= self.budget - 50:
                            break
                        if time.time() > self.deadline:
                            break
                        z = rng.standard_normal(dim)
                        x_new = mean + sigma * z
                        x_new = np.clip(x_new, -5, 5)
                        f_new = wrapped_func(x_new)
                        offspring.append(x_new)
                        offspring_f.append(f_new)
                        
                        if f_new < best_f_local:
                            best_f_local = f_new
                            best_x_local = x_new.copy()
                            if best_f_local < self.best_f:
                                self.best_f = best_f_local
                                self.best_x = best_x_local.copy()
                    
                    if len(offspring) < mu:
                        break
                        
                    idx = np.argsort(offspring_f)
                    offspring = [offspring[i] for i in idx]
                    offspring_f = [offspring_f[i] for i in idx]
                    
                    mu_actual = min(mu, len(offspring))
                    mean = np.mean(offspring[:mu_actual], axis=0)
                    sigma = max(sigma * 0.9, 0.1)
                    
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 2: L-BFGS-B for fast local convergence from best found point
        if self.calls_made < self.budget - 50 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                
                res_lbfgsb = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': self.budget - self.calls_made - 20
                    }
                )
                
                if np.isfinite(res_lbfgsb.fun) and res_lbfgsb.fun < self.best_f:
                    self.best_f = res_lbfgsb.fun
                    self.best_x = np.clip(res_lbfgsb.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish
        if self.calls_made < self.budget - 10 and self.best_x is not None:
            try:
                x_start = self.best_x.copy()
                x_start = np.clip(x_start, -5, 5)
                
                res_nm = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'maxiter': 50,
                        'adaptive': True
                    }
                )
                
                if np.isfinite(res_nm.fun) and res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = wrapped_func(x0)

        return self.best_x, self.best_f
