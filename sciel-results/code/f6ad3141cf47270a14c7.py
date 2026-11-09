"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

class Optimizer:
    """Surrogate-guided optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Storage for surrogate
        self.X = []
        self.Y = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            self.X.append(x.copy())
            self.Y.append(fx)
            return fx

        # Phase 1: Initial Exploration
        # Sample a small number of points to seed the surrogate.
        # Use a mix of random and stratified points to cover the space.
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Add origin and some random points
        points = [np.zeros(self.dim)]
        for _ in range(n_init - 1):
            points.append(rng.uniform(-5, 5, self.dim))
            
        for p in points:
            if time.time() > deadline:
                break
            wrapped_func(p)

        # Phase 2: Surrogate-Guided Search
        # Use RBF interpolation to predict the function value and select
        # the point with the lowest predicted value (greedy global optimization).
        
        # We will run this loop until budget or time is exhausted, 
        # but reserve some budget for local refinement at the end.
        local_budget_reserve = 50
        max_iterations = self.budget - local_budget_reserve - self.call_count
        
        X_arr = np.array(self.X)
        Y_arr = np.array(self.Y)
        
        # RBF Kernel: Gaussian
        # We need to choose a width parameter. 
        # A common heuristic is to use the median distance between points.
        
        def get_rbf_model(X, Y):
            n = len(X)
            if n == 1:
                return lambda x: Y[0]
            
            # Calculate pairwise distances
            # For efficiency, we only compute distances for the new candidate
            # but for model building we need the matrix.
            # Since n is small (< 500), we can build the matrix each time 
            # or update it. Given the low dimensionality, rebuilding is fine.
            
            # To avoid singularities, we add a small jitter or use a fixed width.
            # Let's use a fixed width based on the domain size.
            # Domain is [-5, 5]^5, max distance ~ 14.
            # A width of 1.0 to 2.0 seems reasonable.
            width = 1.5
            
            # Precompute kernel matrix K
            # K_ij = exp(-width * ||x_i - x_j||^2)
            # Actually, standard RBF is exp(-gamma * ||x_i - x_j||^2)
            # Let's use gamma = 1.0 / (width^2)
            gamma = 1.0 / (width * width)
            
            # Efficiently compute K
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            sq_norms = np.sum(X**2, axis=1)
            K = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
            K = np.exp(-gamma * K)
            
            # Add small regularization to ensure invertibility
            K += 1e-8 * np.eye(n)
            
            # Solve for weights w
            try:
                w = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                w = np.linalg.pinv(K) @ Y
                
            def predict(x_new):
                # x_new is a single point (dim,) or batch (n, dim)
                if x_new.ndim == 1:
                    x_new = x_new.reshape(1, -1)
                
                # Calculate distances from x_new to X
                # ||x_new - x_i||^2
                sq_norms_new = np.sum(x_new**2, axis=1)
                dist_sq = sq_norms_new[:, None] + sq_norms[None, :] - 2 * (x_new @ X.T)
                dist_sq = np.maximum(dist_sq, 0) # Numerical stability
                K_new = np.exp(-gamma * dist_sq)
                
                return K_new @ w
            
            return predict

        # We will rebuild the model every few iterations to save time,
        # or every iteration since n is small.
        # Let's try rebuilding every 5 iterations to balance accuracy and speed.
        
        for i in range(max_iterations):
            if time.time() > deadline - 10: # Keep 10s for local phase
                break
            if self.call_count >= self.budget - local_budget_reserve:
                break

            # Build/Update Model
            if i % 5 == 0 or i == 0:
                model = get_rbf_model(np.array(self.X), np.array(self.Y))
            
            # Candidate Generation
            # Generate a set of candidate points and pick the one with lowest predicted f
            n_candidates = 50
            candidates = []
            
            # 1. Random candidates
            candidates.extend(rng.uniform(-5, 5, (n_candidates // 2, self.dim)))
            
            # 2. Local perturbations around best point
            if self.best_x is not None:
                perturbations = self.best_x + rng.normal(0, 0.5, (n_candidates // 2, self.dim))
                perturbations = np.clip(perturbations, -5, 5)
                candidates.extend(perturbations)
            
            candidates = np.array(candidates)
            
            # Predict values
            preds = model(candidates)
            
            # Select best candidate
            best_idx = np.argmin(preds)
            x_new = candidates[best_idx]
            
            # Evaluate
            wrapped_func(x_new)

        # Phase 3: Local Refinement
        # Use Nelder-Mead from the best point found so far.
        if self.best_x is not None and self.call_count < self.budget:
            x0 = self.best_x.copy()
            
            def nm_func(x):
                if self.call_count >= self.budget:
                    raise ValueError("Budget exceeded")
                self.call_count += 1
                fx = func(x)
                if fx < self.best_f:
                    self.best_f = fx
                    self.best_x = x.copy()
                return fx

            try:
                # Use strict tolerances to exploit remaining budget
                minimize(nm_func, x0, method='Nelder-Mead', 
                         options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
            except ValueError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
