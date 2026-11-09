"""Combined program exploiting WFG4 structure: dense sampling of the shape variable and local refinement of bias variables."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for WFG4 exploiting structural knowledge.
    WFG4's Pareto front is primarily determined by the first variable (x_1).
    The other variables affect the distance from the front (bias).
    Strategy:
    1. Sample x_1 densely to cover the front.
    2. For each x_1, use local search (hill climbing) to minimize bias variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 typically has x_1 as the shape variable.
        # Let's assume the first variable is the most critical for the front shape.
        # We will allocate a portion of the budget to sample x_1 and the rest to refine the other variables.
        
        # Number of distinct x_1 positions to sample.
        # If we use 20 samples, and 10 evals per refinement, that's 200 evals.
        # Let's be dynamic based on budget.
        n_samples = 20
        evals_per_sample = max(1, self.budget // n_samples)
        
        # Ensure we don't exceed budget
        if n_samples * evals_per_sample > self.budget:
            n_samples = self.budget // max(1, evals_per_sample)
            
        # Pre-allocate storage for results to return
        # The contract says "return value of __call__ is ignored" but we should return something valid.
        # We will store the best non-dominated solutions found.
        solutions = []
        
        # Define a local search function for the bias variables (x_2...x_dim)
        # given a fixed x_1.
        def local_search(x1_val, initial_bias, max_evals):
            current_x = np.empty(self.dim)
            current_x[0] = x1_val
            current_x[1:] = initial_bias
            
            current_f = problem(current_x)
            best_x = current_x.copy()
            best_f = current_f.copy()
            evals_used = 1
            
            # Simple hill climbing with random perturbations
            # We only perturb the bias variables (indices 1 to dim-1)
            step_size = 0.1 * (hi - lo)
            
            for _ in range(max_evals - 1):
                if time.time() > deadline:
                    break
                if evals_used >= max_evals:
                    break
                    
                # Generate a candidate by perturbing bias variables
                candidate = best_x.copy()
                # Perturb only bias variables
                perturb = np.random.standard_normal(self.dim - 1) * step_size
                candidate[1:] += perturb
                candidate = np.clip(candidate, lo, hi)
                
                cand_f = problem(candidate)
                evals_used += 1
                
                # Check if candidate is better (dominates) or just better in sum (for simplicity)
                # For WFG4, minimizing all objectives is good. 
                # We accept if it dominates best_f or if it's non-dominated and closer to ideal point.
                # Simple heuristic: accept if sum of objectives is lower, or if it dominates.
                if np.all(cand_f <= best_f) and np.any(cand_f < best_f):
                    best_x = candidate
                    best_f = cand_f
                elif np.all(best_f <= cand_f) and np.any(best_f < cand_f):
                    # Candidate is dominated, reject
                    pass
                else:
                    # Non-dominated. Use a simple archive or just keep the one with lower sum?
                    # To maintain diversity in the final set, we might want to keep both, 
                    # but local search usually converges to one point per x1.
                    # Let's use a simple weighted sum or just keep the one with lower max deviation.
                    # Actually, for a fixed x1, the optimal point on the front is unique (or close).
                    # So we just want to get as close to the front as possible.
                    # Minimizing the distance to the ideal point [0,0,0] is a good proxy.
                    # Let's check distance to ideal.
                    dist_best = np.linalg.norm(best_f)
                    dist_cand = np.linalg.norm(cand_f)
                    if dist_cand < dist_best:
                        best_x = candidate
                        best_f = cand_f

            return best_x, best_f, evals_used

        # Main loop: Sample x_1 and refine
        # We sample x_1 uniformly between lo[0] and hi[0]
        x1_samples = np.linspace(lo[0], hi[0], n_samples)
        
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            
            # Initial bias variables: random or midpoint?
            # Midpoint is often a good start for WFG problems as they are often biased towards the center.
            initial_bias = (lo[1:] + hi[1:]) / 2.0
            
            # Perform local search
            best_x, best_f, used = local_search(x1_val, initial_bias, evals_per_sample)
            
            # Record the solution
            solutions.append(best_x)
            
            # Check budget (approximate, local_search tracks internally but we need global check)
            # Actually, local_search returns used evals. We should track total.
            # Let's just rely on the loop count and deadline for now.
            # The local_search is limited by evals_per_sample.
            
        # The problem asks to return the best result found.
        # Since the return value is ignored, we just need to ensure the evaluator sees the calls.
        # However, returning the array of solutions is a good practice.
        return np.array(solutions)

# EVOLVE-BLOCK-END
