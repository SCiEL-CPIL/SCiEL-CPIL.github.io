"""Strategy: Deterministic decomposition via weighted-sum scalarization with Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use a set of weight vectors to cover the objective space.
        # For 3 objectives, we can generate points on the simplex.
        # Let's use a fixed number of directions, e.g., 10-15, to keep evals per run low enough for Nelder-Mead to converge.
        
        # Generate weight vectors on the simplex summing to 1
        n_weights = 12
        weights = []
        # Use a simple grid or random-ish deterministic set on the 2D simplex
        # Since n_obj=3, weights are (w1, w2, w3) with w1+w2+w3=1, wi>=0
        # We can parameterize by two angles or just use a predefined set.
        # Let's use a set of points that includes the vertices and edge midpoints and some interior points.
        
        # Vertices
        w_list = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3],
            [0.7, 0.15, 0.15],
            [0.15, 0.7, 0.15],
            [0.15, 0.15, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
        
        # Shuffle deterministically to ensure different starting points for different seeds?
        # No, the evaluator seeds np.random. But we don't use np.random for the weights.
        # We should use the global rng or just fixed order.
        # The problem states: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)... and calls it once."
        # We can use np.random to pick starting points for Nelder-Mead to leverage the seed.
        
        rng = np.random.default_rng() # Uses the global seed state
        
        solutions = []
        
        # Budget allocation: 
        # Total budget 200. 
        # If we run 16 optimizations, that's 12.5 evals each. Nelder-Mead might need more to converge well in 12D.
        # Let's reduce the number of weights or increase budget per weight?
        # Actually, Nelder-Mead in 12D is expensive. 
        # Alternative: Use a global optimizer like Differential Evolution or Basin Hopping? 
        # Basin Hopping is good for single objective.
        # Let's stick to Nelder-Mead but with a strict maxiter.
        
        # Let's try 8 weights to allow ~25 evals each.
        selected_weights = w_list[:8]
        
        for w in selected_weights:
            if time.time() > deadline - 10.0:
                break
                
            # Define scalar objective
            def scalar_obj(x):
                f = problem(x)
                return np.dot(w, f)
            
            # Initial guess: random within bounds
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            # Bounds for Nelder-Mead? Nelder-Mead doesn't natively support bounds well in older scipy, 
            # but we can clip inside the function or use a transformation.
            # The problem() function already clips x to bounds.
            # So we can just pass x0 and let it run.
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.success or res.fun < 1e10:
                    solutions.append(res.x)
            except Exception:
                continue
            
        # If we have time, we can also add a few random points to ensure we don't miss extremes
        if time.time() < deadline - 5.0:
             for _ in range(5):
                 x_rand = lo + (hi - lo) * rng.random(self.dim)
                 solutions.append(x_rand)
                 
        # If we have very little time, just return what we have.
        # The evaluator records ALL objective vectors returned by problem().
        # Wait, the contract says: "Every returned objective vector is recorded".
        # My current code calls problem() inside scalar_obj.
        # So every evaluation in Nelder-Mead is recorded.
        # The return value of __call__ is ignored.
        # So I just need to make sure I stay within budget and time.
        
        return None
# EVOLVE-BLOCK-END
