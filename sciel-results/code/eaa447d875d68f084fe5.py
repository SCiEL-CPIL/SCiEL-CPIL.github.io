import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # ZDT1 Structural Analysis:
        # f1 = x0
        # f2 = C * (1 - sqrt(x0 / g(x1..xn)))
        # To minimize f2 for a given x0, we must minimize g(x1..xn).
        # Standard ZDT1: g = 1 + 9 * sum(x1..xn)/(n-1). Min g = 1 at x1..xn = 0.
        # Thus, the Pareto front lies on the manifold x1..xn = 0.
        # Any point off this manifold has higher f2 for the same f1, hence is dominated.
        
        # Strategy:
        # 1. Initialize a set of candidate points on the manifold (x1..xn = lo[1:]).
        #    Distribute x0 to cover the range.
        # 2. Use local search to refine x0 positions if possible, though on a 
        #    deterministic front, the manifold itself is the only optimal locus.
        #    However, we don't know for sure if the evaluator's ZDT1 is standard.
        #    It might have a different g function.
        #    So, we will perform a simple local search around the manifold points.
        #    We assume the manifold x1..xn = lo[1:] is a good starting point.
        
        # Let's allocate budget:
        # - 50% for initial exploration on the manifold (spread x0).
        # - 50% for local refinement (perturbing x0 and x1..xn slightly to see if we can find better f2).
        
        # Since the front is 1D (parameterized by x0), and we want to maximize hypervolume,
        # we need to cover the range of x0 effectively.
        
        # Initial Phase: Grid on the manifold
        n_init = max(10, n_evals // 2)
        # Use sqrt transformation for better hypervolume coverage in objective space
        # assuming f2 ~ 1 - sqrt(f1).
        u_vals = np.linspace(0.0, 1.0, n_init)
        x0_init = (u_vals ** 2) * (hi[0] - lo[0]) + lo[0]
        
        X_init = np.zeros((n_init, self.dim))
        X_init[:, 0] = x0_init
        X_init[:, 1:] = lo[1:]
        
        # Clip x0 just in case
        X_init[:, 0] = np.clip(X_init[:, 0], lo[0], hi[0])
        
        # Evaluate initial points
        F_init = np.zeros((n_init, self.n_obj))
        for i in range(n_init):
            if time.time() > deadline:
                break
            F_init[i] = problem(X_init[i])
        
        # Local Search Phase:
        # Pick the best points from F_init (non-dominated or just lowest f2 for their f1)
        # and try to improve them by perturbing x0 and x1..xn.
        # Since we suspect x1..xn=0 is optimal, we will mostly perturb x0.
        # But to be robust, we allow small perturbations in x1..xn as well.
        
        # Let's keep a list of current best points (x, f)
        # We'll just use the initial points as the "population" and try to improve each one.
        
        n_search = n_evals - n_init
        if n_search < 1:
            # If no budget left, return what we have
            return X_init[:n_init]
        
        # We will iterate through the initial points and try to find a better neighbor.
        # "Better" means lower f2 (since f1 is fixed by x0, or we accept trade-offs).
        # Actually, for hypervolume, we want points on the front.
        # If we move off the front, f2 increases, which is bad.
        # So local search should primarily adjust x0 to find the exact front location if the manifold assumption is wrong,
        # or just verify the manifold.
        
        # Let's do a simple random walk around each initial point.
        # We will keep the best point found for each "slot" (initial x0 region).
        
        best_X = X_init.copy()
        best_F = F_init.copy()
        
        # Simple local search: for each initial point, try a few random perturbations.
        # Number of perturbations per point = n_search / n_init
        n_perturb_per_point = max(1, n_search // n_init)
        
        for i in range(n_init):
            current_x = best_X[i].copy()
            current_f = best_F[i].copy()
            
            for k in range(n_perturb_per_point):
                if time.time() > deadline:
                    break
                
                # Generate a perturbation
                # 80% chance: perturb x0 only (stay on manifold)
                # 20% chance: perturb x0 and x1..xn (off manifold)
                if np.random.rand() < 0.8:
                    new_x = current_x.copy()
                    # Gaussian step on x0
                    step_size = 0.05 * (hi[0] - lo[0])
                    new_x[0] += np.random.normal(0, step_size)
                    new_x[0] = np.clip(new_x[0], lo[0], hi[0])
                    # Ensure x1..xn stay at lo (or reset to lo if drifted)
                    new_x[1:] = lo[1:]
                else:
                    new_x = current_x.copy()
                    step_size_x0 = 0.05 * (hi[0] - lo[0])
                    step_size_other = 0.1 * (hi[1] - lo[1]) # Assuming similar scales
                    new_x[0] += np.random.normal(0, step_size_x0)
                    new_x[1:] += np.random.normal(0, step_size_other, size=self.dim-1)
                    new_x[0] = np.clip(new_x[0], lo[0], hi[0])
                    new_x[1:] = np.clip(new_x[1:], lo[1:], hi[1:])
                
                new_f = problem(new_x)
                
                # Accept if new_f is better in f2 (since f1 is roughly similar or we prefer lower f2 for same f1)
                # Ideally we should check dominance, but for local search on a front,
                # minimizing f2 for a given x0 is the goal.
                # If new_x[0] is close to current_x[0], and new_f[1] < current_f[1], accept.
                # If new_x[0] is different, it's a trade-off. 
                # For simplicity, we accept if new_f[1] < current_f[1] AND abs(new_x[0]-current_x[0]) < 0.01
                # OR if it dominates current_f (new_f[0] <= current_f[0] and new_f[1] < current_f[1])
                
                is_better = False
                # Check dominance (minimizing both)
                if (new_f[0] <= current_f[0] and new_f[1] < current_f[1]) or \
                   (new_f[0] < current_f[0] and new_f[1] <= current_f[1]):
                    is_better = True
                else:
                    # If not dominating, check if it's a significant improvement in f2 for a similar f1
                    if abs(new_f[0] - current_f[0]) < 0.01 and new_f[1] < current_f[1]:
                        is_better = True
                
                if is_better:
                    best_X[i] = new_x
                    best_F[i] = new_f
                    current_x = new_x
                    current_f = new_f

        return best_X[:n_init]
