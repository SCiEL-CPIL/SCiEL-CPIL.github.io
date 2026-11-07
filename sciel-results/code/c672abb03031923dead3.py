import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using a decomposition + local search strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # Generate weight vectors for decomposition (2 objectives)
        # Use a set of evenly spaced weights
        n_weights = 40
        weights = np.array([(i / (n_weights - 1), 1 - i / (n_weights - 1)) for i in range(n_weights)])

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add solution to archive if non-dominated, remove dominated points."""
            # Check if new point is dominated
            if archive_F:
                for i, af in enumerate(archive_F):
                    if np.all(af <= f) and np.any(af < f):
                        return False
            # Remove points dominated by new point
            if archive_F:
                new_indices = []
                for i, af in enumerate(archive_F):
                    if np.all(f <= af) and np.any(f < af):
                        continue
                    new_indices.append(i)
                archive_X = [archive_X[i] for i in new_indices]
                archive_F = [archive_F[i] for i in new_indices]
            archive_X.append(x)
            archive_F.append(f)
            return True

        def evaluate(x):
            """Evaluate a single point, clipping to bounds."""
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            return x_clipped, f

        # Initialize population with points near the known Pareto front structure
        # ZDT6 Pareto front: x1 determines the front position
        # f1 = 1 + 8*sin^2(pi*x1)
        # f2 = (1 - sqrt(f1)) * g(x) where g depends on other variables
        # Optimal g is achieved when x2..x10 are set to specific values
        
        # Strategy: 
        # 1. Sample x1 uniformly, set other variables to values that minimize g
        # 2. Use local search to refine
        
        # For ZDT6, g(x) = 1 + 10*(n-1) + sum(x_i^2 * (1 + 10*sin^2(10*pi*x_i)))
        # g is minimized when x_i = 0 for i >= 2 (but with the sin term, x_i=0 gives sin(0)=0)
        # Actually g_min = 1 + 10*(n-1) when all x_i = 0 for i>=2
        
        # Let's initialize with a mix:
        # - Some points with x1 spread out and others optimized
        # - Some random points for diversity
        
        pop_size = min(30, budget // 6)
        X = np.zeros((pop_size, n))
        
        # First half: structured points along x1
        n_structured = pop_size // 2
        for i in range(n_structured):
            X[i, 0] = (i + 0.5) / n_structured  # spread x1
            X[i, 1:] = 0.0  # optimal for g
        
        # Second half: random
        for i in range(n_structured, pop_size):
            X[i] = lo + (hi - lo) * np.random.random(n)
        
        # Evaluate initial population
        F = np.zeros((pop_size, m))
        used = 0
        for i in range(pop_size):
            if used >= budget:
                break
            x, f = evaluate(X[i])
            F[i] = f
            used += 1
            add_to_archive(x, f)
        
        # Local search / refinement phase
        # Use Nelder-Mead or coordinate descent on promising points
        
        # Sort archive by first objective to get diverse starting points
        if archive_F:
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            
            # Select diverse points from archive for local search
            # Use a simple crowding-like selection
            n_local = min(10, len(archive_X_arr), (budget - used) // 20)
            
            if n_local > 0:
                # Select evenly spaced points from sorted archive
                idx = np.linspace(0, len(archive_X_arr) - 1, n_local, dtype=int)
                for idx_i in idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    x0 = archive_X_arr[idx_i].copy()
                    f0 = archive_F_arr[idx_i].copy()
                    
                    # Coordinate descent / local search
                    step_size = 0.05
                    for coord in range(n):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Try small perturbations in each direction
                        for direction in [-1, 1]:
                            if used >= budget or time.time() > deadline:
                                break
                            
                            x_trial = x0.copy()
                            x_trial[coord] += direction * step_size
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = np.array(problem(x_trial))
                            used += 1
                            
                            # Accept if it improves or is non-dominated
                            if np.any(f_trial < f0):
                                # Check if it's better in at least one objective
                                # and not worse in all
                                if not np.all(f_trial >= f0):
                                    x0 = x_trial
                                    f0 = f_trial
                                    add_to_archive(x0, f0)
                            
                            # Adaptive step size
                            step_size *= 0.7
        
        # Final polish: try to improve the best points in each region
        if archive_F and used < budget and time.time() < deadline:
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            
            # Focus on points that are close to the ideal front
            # ZDT6 ideal: f1 in [1, 9], f2 in [0, 1]
            # Reference point for hypervolume is [11, 11]
            
            n_polish = min(5, len(archive_X_arr), (budget - used) // 10)
            if n_polish > 0:
                idx = np.linspace(0, len(archive_X_arr) - 1, n_polish, dtype=int)
                for idx_i in idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    x0 = archive_X_arr[idx_i].copy()
                    f0 = archive_F_arr[idx_i].copy()
                    
                    # Fine-grained coordinate search
                    step = 0.01
                    for coord in range(n):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        for direction in [-1, 1]:
                            if used >= budget or time.time() > deadline:
                                break
                            
                            x_trial = x0.copy()
                            x_trial[coord] += direction * step
                            x_trial = np.clip(x_trial, lo, hi)
                            
                            f_trial = np.array(problem(x_trial))
                            used += 1
                            
                            if np.any(f_trial < f0) and not np.all(f_trial >= f0):
                                x0 = x_trial
                                f0 = f_trial
                                add_to_archive(x0, f0)
                        
                        step *= 0.5
        
        # Return the archive (the evaluator records all objective vectors returned)
        # We need to return something, but the return value is ignored per contract
        # However, we should return the best solutions found
        
        if archive_X:
            return np.array(archive_X)
        else:
            return np.zeros((1, n))
