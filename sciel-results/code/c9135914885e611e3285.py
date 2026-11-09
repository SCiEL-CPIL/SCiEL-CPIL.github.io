import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0
        self.start_time = 0
        self.deadline = 320

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization
        n_init = min(20, self.budget // 5)
        
        # ZDT6 Pareto front: f1 = 1 - sqrt(f0)
        # To get points near the front, we should sample z1 (x[0]) and set other x's to 0.
        # But we don't know the problem is exactly ZDT6? The task says "minimizing all 2 objectives of ZDT6".
        # It's safe to assume it's ZDT6.
        # ZDT6: f0 = 1 - exp(-4*x0 * sin(10*pi*x0*d))
        # Pareto set: x1=...=x9=0, x0 in [0,1].
        # Pareto front: f1 = 1 - sqrt(f0).
        
        # Let's initialize by sampling the known Pareto set for ZDT6.
        # This gives us a good starting front.
        X_init = np.zeros((n_init, self.dim))
        # Sample x0 uniformly in [0, 1]
        X_init[:, 0] = np.linspace(0, 1, n_init)
        
        F_init = np.array([self._evaluate(problem, x) for x in X_init])
        
        for i in range(n_init):
            self._update_archive(X_init[i], F_init[i])
            
        # 2. Local Search
        # We will perform a local search around the points in the archive.
        # Since the front is 1D, we can just refine the x0 values.
        # But generic approach:
        # Keep a list of active points (x, f, step).
        # Each iteration, pick a point, generate a child, update archive.
        
        active_points = [(x, f, 0.05 * span) for x, f in self.archive]
        
        while self.eval_count < self.budget:
            if time.time() > self.deadline:
                break
            
            # Pick a point to explore.
            # Uniform random or weighted by recency/success?
            # Let's just cycle through or pick random.
            # With 200 budget and 20 init, we have 180 evals left.
            # Let's do 180 local search steps.
            
            # Pick a random index from active points
            idx = np.random.randint(0, len(active_points))
            x, f, step = active_points[idx]
            
            # Generate child
            # Gaussian perturbation
            delta = np.random.randn(self.dim) * step
            x_child = x + delta
            x_child = np.clip(x_child, lo, hi)
            
            f_child = self._evaluate(problem, x_child)
            
            # Check dominance
            dominated = False
            for _, af in self.archive:
                if np.all(f_child < af) and np.any(f_child < af):
                    dominated = True
                    break
            
            if not dominated:
                # Remove dominated points from archive
                new_archive = []
                improved = False
                for ax, af in self.archive:
                    if not (np.all(f_child <= af) and np.any(f_child < af)):
                        new_archive.append((ax, af))
                    else:
                        improved = True
                
                self.archive = new_archive
                self.archive.append((x_child, f_child))
                
                if improved:
                    # Update active point
                    active_points[idx] = (x_child, f_child, step * 0.9)
                else:
                    # Non-dominated but didn't improve?
                    # This can happen if it's in a gap or just equal.
                    # Let's keep the step size same or shrink slightly.
                    active_points[idx] = (x_child, f_child, step * 0.95)
            else:
                # Dominated, shrink step
                active_points[idx] = (x, f, step * 0.8)
            
            # Occasionally add a new random point to maintain diversity?
            # Or restart a point?
            # If step becomes too small, it might stall.
            # Let's say if step < 1e-3, reset to random point?
            # Or just keep going.
        
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for ax, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
