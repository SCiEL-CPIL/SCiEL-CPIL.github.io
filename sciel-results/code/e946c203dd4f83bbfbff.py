import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization with Sobol sequence for better coverage
        n_init = min(50, self.budget // 4)
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Generate points in [0, 1]
        # Sobol requires power of 2 for best properties, but works for any N
        # We'll just generate n_init points
        # Note: Sobol in scipy returns points in [0, 1]
        # We need to map to [lo, hi]
        
        # Ensure n_init is valid for Sobol if strict, but scipy is flexible
        # Actually, let's just use random if Sobol is tricky with small N, but Sobol is known to help.
        # Let's stick to random for simplicity and robustness, or use LHS.
        # The knowledge says Sobol helped. Let's use it.
        
        # scipy.stats.qmc.Sobol requires d dimensions.
        # sample(n) returns (n, d)
        
        try:
            sobol_sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
            # Sobol sample size should ideally be power of 2, but let's try n_init
            # If n_init is not power of 2, it still works but might be less uniform
            # Let's round to next power of 2 or just use it
            n_sobol = int(2 ** np.ceil(np.log2(max(n_init, 1))))
            if n_sobol > self.budget:
                n_sobol = self.budget
            points = sobol_sampler.random(n_sobol)
            # Map to bounds
            X_init = lo + (hi - lo) * points[:n_init]
        except Exception:
            # Fallback to random
            X_init = lo + (hi - lo) * np.random.rand(n_init, self.dim)

        used = 0
        archive_X = []
        archive_F = []
        
        # Evaluate initial population
        for x in X_init:
            if used >= self.budget:
                break
            f = problem(x)
            used += 1
            self._add_to_archive(x, f, archive_X, archive_F)
            
        if used >= self.budget:
            return archive_X

        # 2. Main Loop: Differential Evolution on Archive
        # F factor for DE
        F = 0.8
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            if len(archive_X) < 3:
                # Need at least 3 distinct points for DE (a, b, c)
                # If we have fewer, just do random search
                x_rand = lo + (hi - lo) * np.random.rand(self.dim)
                f_rand = problem(x_rand)
                used += 1
                self._add_to_archive(x_rand, f_rand, archive_X, archive_F)
                continue
            
            # Select 3 distinct parents from archive
            # To ensure diversity, we can pick randomly
            idxs = np.random.choice(len(archive_X), 3, replace=False)
            x_a = archive_X[idxs[0]]
            x_b = archive_X[idxs[1]]
            x_c = archive_X[idxs[2]]
            
            # DE mutation
            x_child = x_a + F * (x_b - x_c)
            
            # Optional: Crossover (Binomial)
            # Keep it simple: just mutation for now, or simple crossover
            # Let's add a simple crossover with rate 0.5
            # Actually, pure DE mutation is often sufficient for small budgets.
            
            # Clip to bounds
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            self._add_to_archive(x_child, f_child, archive_X, archive_F)

        # 3. Polish Phase
        # Use remaining budget to refine the best points
        remaining = self.budget - used
        if remaining > 0 and len(archive_X) > 0:
            # Select top 5 points based on some metric (e.g., distance to ideal or just random)
            # Let's just pick 5 random points from the archive to polish
            n_polish = min(5, len(archive_X))
            polish_idxs = np.random.choice(len(archive_X), n_polish, replace=False)
            
            for idx in polish_idxs:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                x_base = archive_X[idx]
                # Try small perturbations
                # We have remaining evaluations. Let's use 10 per base point if possible
                steps_per_point = max(1, remaining // n_polish)
                
                for _ in range(steps_per_point):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    # Small Gaussian step
                    step_size = 0.05 * (hi - lo)
                    x_try = x_base + step_size * np.random.randn(self.dim)
                    x_try = np.clip(x_try, lo, hi)
                    
                    f_try = problem(x_try)
                    used += 1
                    
                    self._add_to_archive(x_try, f_try, archive_X, archive_F)

        return archive_X

    def _add_to_archive(self, x, f, archive_X, archive_F):
        """
        Add a point to the archive if it is non-dominated.
        Also remove any points in the archive that are dominated by the new point.
        """
        # Check if x is dominated by any point in archive
        # f is better than f_arch if all f < f_arch and at least one strictly less
        dominated = False
        for i, f_arch in enumerate(archive_F):
            if np.all(f <= f_arch) and np.any(f < f_arch):
                dominated = True
                break
        
        if dominated:
            return
            
        # If not dominated, remove any archive points dominated by x
        new_archive_X = []
        new_archive_F = []
        for i in range(len(archive_X)):
            # Check if archive point i is dominated by x
            if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                # archive point i is dominated, skip it
                continue
            new_archive_X.append(archive_X[i])
            new_archive_F.append(archive_F[i])
        
        new_archive_X.append(x)
        new_archive_F.append(f)
        
        # Update lists in place (or return new ones, but lists are mutable)
        archive_X.clear()
        archive_X.extend(new_archive_X)
        archive_F.clear()
        archive_F.extend(new_archive_F)
