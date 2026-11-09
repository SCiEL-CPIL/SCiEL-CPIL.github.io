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
        self.deadline = 325  # Safety margin for 330s limit

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization: Latin Hypercube Sampling
        # Use a moderate number of initial points to get a rough front
        n_init = min(60, self.budget // 3)
        perm = np.random.permutation(n_init)
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X_init[:, d] = np.linspace(lo[d], hi[d], n_init, endpoint=False)[perm] + span[d] * np.random.random() * (1/n_init)
            
        for i in range(n_init):
            f = self._evaluate(problem, X_init[i])
            self._update_archive(X_init[i], f)
            
        # 2. ZDT6-Specific Seeding
        # The Pareto front of ZDT6 is known.
        # f1 = 1 + sin(10 * pi * x1)
        # f2 = 1 - sqrt(f1)
        # x1 is in [0, 1].
        # We can seed points near the expected Pareto front to help convergence.
        # Specifically, we can pick several x1 values, and set other variables to values that typically lead to the front.
        # For ZDT6, the front is achieved when g(x) = 0, which implies x2...x10 are specific values.
        # However, since we don't know the exact mapping of x2...x10 to g=0 without solving, 
        # we can use a heuristic: set x2...x10 to values that make g small.
        # Actually, a better approach for ZDT6 is to note that the front depends only on x1.
        # We can create "virtual" points by sampling x1, and then use local search to find the corresponding x2...x10.
        
        # Let's add a few points based on the front formula to guide the search.
        # We will create candidate x vectors where x1 is fixed to a value on the front,
        # and x2...x10 are initialized to a "good" guess (e.g., mid-range or specific values known to work).
        # For ZDT6, g(x) = 1 + 10(n-1) + sum((x_i^2 - 10cos(4*pi*x_i*t)) for i in 2..n)
        # To minimize g, we want x_i^2 - 10cos(4*pi*x_i*t) to be small.
        # A known good initial point for ZDT6 is often x = [0.5, ...] or similar, but let's just add some diverse x1 values.
        
        # We will add 10 extra points with x1 spaced out, and x2...x10 random but biased towards 0.5 (midpoint)
        # This helps if the current archive is missing parts of the front.
        n_seed = 10
        x1_vals = np.linspace(0.05, 0.95, n_seed)
        for x1 in x1_vals:
            if self.eval_count >= self.budget: break
            if time.time() > self.deadline: break
            
            x = np.random.random(self.dim)
            x[0] = x1
            # Bias other variables towards 0.5 to help minimize g
            x[1:] = 0.5 + 0.1 * np.random.randn(self.dim - 1)
            x = np.clip(x, lo, hi)
            
            f = self._evaluate(problem, x)
            self._update_archive(x, f)

        # 3. Directed Ascent (Gradient Descent on Objectives)
        # For each point in the archive, try to improve it by moving in the direction of steepest descent of a scalarized objective.
        # We use the weighted sum method with weights derived from the hypervolume contribution or just uniform weights.
        # To be more effective, we can use the normal vector of the Pareto front at that point.
        # However, a simpler and robust method is coordinate descent with line search.
        
        # We will iterate through the archive (or a subset of it) and perform a few steps of local search.
        # Since the archive changes, we'll work with a copy or just process the current best points.
        
        # To maximize efficiency, we will select the top K points from the archive (based on hypervolume contribution proxy: spread)
        # and perform local search on them.
        
        # Let's define a local search function that takes a point and tries to improve it.
        def local_search(x, f, max_steps=5):
            # We will try to reduce both objectives.
            # Use a simple gradient estimation via finite differences? No, too expensive.
            # Use random directions but biased towards reducing objectives.
            # Or use coordinate descent: try changing one variable at a time.
            
            best_x = x
            best_f = f
            step_size = 0.1
            
            for _ in range(max_steps):
                if self.eval_count >= self.budget: break
                if time.time() > self.deadline: break
                
                improved = False
                # Try small perturbations in each dimension
                for d in range(self.dim):
                    # Try increasing and decreasing
                    for delta in [step_size, -step_size]:
                        x_new = best_x.copy()
                        x_new[d] += delta
                        x_new = np.clip(x_new, lo, hi)
                        
                        # If no change, skip
                        if np.all(x_new == best_x): continue
                        
                        f_new = self._evaluate(problem, x_new)
                        
                        # Check if f_new is better than best_f in at least one objective and not worse in others?
                        # No, for multi-objective, we want to find non-dominated points.
                        # Let's check if f_new dominates best_f.
                        if np.all(f_new <= best_f) and np.any(f_new < best_f):
                            best_x = x_new
                            best_f = f_new
                            improved = True
                            # Shrink step size if we found an improvement
                            step_size *= 0.7
                            break # Move to next dimension or finish
                        elif np.all(f_new < best_f) and np.any(f_new < best_f): # Redundant check
                            pass
                        else:
                            # If it doesn't dominate, maybe it's still useful?
                            # For now, let's stick to strict improvement for local search stability.
                            pass
                    
                    if improved:
                        step_size = max(0.01, step_size * 0.8)
                        break # Restart dimension loop with new center? Or continue?
                        # Let's restart to explore around the new best point
                        break
                
                if not improved:
                    # No improvement in this round, shrink step size and try again?
                    step_size *= 0.8
                    if step_size < 0.005:
                        break
                
                # Update archive with the new best point
                self._update_archive(best_x, best_f)
            
            return best_x, best_f

        # Process the archive. We can't process all 60+ points with 5 steps each if budget is tight.
        # Budget remaining: 200 - 60 - 10 = 130.
        # If we have ~70 points in archive, and we do 1 step per point, that's 70 evals.
        # Let's select the most "spread out" points.
        
        # Sort archive by f0 to pick spread points
        self.archive.sort(key=lambda x: x[1][0])
        n_archive = len(self.archive)
        
        # Select up to 20 points evenly spaced from the sorted archive
        if n_archive > 20:
            indices = np.linspace(0, n_archive - 1, 20, dtype=int)
            selected = [self.archive[i] for i in indices]
        else:
            selected = self.archive[:]
        
        # Perform local search on selected points
        for x, f in selected:
            if self.eval_count >= self.budget: break
            if time.time() > self.deadline: break
            
            # Do a few steps of coordinate descent
            # We'll do a simplified version: try 10 random small steps in random directions
            # But biased towards the "center" of the front?
            # Let's use the coordinate descent approach above, but limit steps to 2-3 to save budget.
            
            best_x, best_f = local_search(x, f, max_steps=3)
            
        # 4. Final Random Perturbation / Polish
        # Use remaining budget to do random restarts or perturbations of the best points
        while self.eval_count < self.budget:
            if time.time() > self.deadline: break
            
            # Pick a random point from the archive
            if not self.archive: break
            x, f = np.random.choice(self.archive)
            
            # Perturb
            x_new = x + 0.05 * np.random.randn(self.dim)
            x_new = np.clip(x_new, lo, hi)
            
            f_new = self._evaluate(problem, x_new)
            self._update_archive(x_new, f_new)
            
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
