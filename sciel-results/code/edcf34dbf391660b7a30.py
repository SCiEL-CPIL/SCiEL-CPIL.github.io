import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 290  # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate weight vectors on the simplex
        n_weights = 20
        weights = self._generate_weights(n_weights, self.n_obj)
        
        # Initialize population
        # We will keep the best solution for each weight vector
        pop_x = np.random.uniform(lo, hi, (n_weights, self.dim))
        pop_f = np.array([problem(x) for x in pop_x])
        
        # Ideal point approximation (min of pop_f)
        ideal = np.min(pop_f, axis=0)
        
        evals = n_weights
        
        # Neighborhood: for simplicity, use a fixed neighborhood size or just global search
        # Given small budget, let's use a simple strategy:
        # Cycle through weights. For each weight, try to improve its current best.
        
        # We need to track the Tchebycheff value for each weight
        # g(x, lambda) = max_i (lambda_i * |f_i(x) - ideal_i|)
        
        # To make it faster, we can use a smaller population or fewer weights if time is tight,
        # but 200 evals is very small. 20 weights is reasonable.
        
        # Mutation step size
        # Adaptive or fixed? Fixed relative to range.
        step_size = 0.1 * (hi - lo)
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select a weight index to work on
            # Round-robin or random? Random might be better for exploration initially.
            # Let's pick the weight with the largest current Tchebycheff value (worst progress)
            # or just cycle. Cycling ensures all parts of the front are touched.
            k = evals % n_weights
            
            # Generate child
            # Mutate the current best for weight k
            parent = pop_x[k]
            child = parent + np.random.normal(0, step_size, self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            # Update ideal point
            new_ideal = np.minimum(ideal, f_child)
            
            # Check if child is better for weight k
            # Calculate Tchebycheff value for child and current pop_x[k] with new ideal
            # Note: Using the updated ideal point makes the comparison consistent
            
            # We need to re-calculate the "best" for all weights if ideal changes?
            # That's expensive. Let's stick to a fixed ideal or update it carefully.
            # Standard MOEA/D updates the ideal point.
            
            # Let's calculate the scalarized value for the child and the current best for weight k
            # using the UPDATED ideal point.
            
            val_child = self._tchebycheff(f_child, weights[k], new_ideal)
            val_current = self._tchebycheff(pop_f[k], weights[k], new_ideal)
            
            if val_child < val_current:
                pop_x[k] = child
                pop_f[k] = f_child
                ideal = new_ideal
            else:
                # Even if it doesn't improve the specific subproblem, 
                # it might be non-dominated in the global sense.
                # But since all points are recorded, we don't strictly need to update the population 
                # for scoring, only for future search guidance.
                # However, keeping the population diverse is key.
                pass
        
        return pop_x

    def _generate_weights(self, n, n_obj):
        # Generate n points on the (n_obj-1) simplex
        # Using Dirichlet distribution or lattice
        # For n_obj=3, simplex is a triangle.
        # Simple method: random points and normalize
        weights = np.random.dirichlet(np.ones(n_obj), size=n)
        return weights

    def _tchebycheff(self, f, lambda_vec, ideal):
        # f: 1D array of objectives
        # lambda_vec: 1D array of weights
        # ideal: 1D array of ideal points
        # Returns scalarized value
        return np.max(lambda_vec * np.abs(f - ideal))
