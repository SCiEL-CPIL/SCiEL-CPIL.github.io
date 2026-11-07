import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use global numpy random to respect external seeding
        # np.random.seed is called by evaluator
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # Population size
        N = 10
        # Ensure we don't exceed budget in init
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, n))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Pareto Archive
        # Store indices or vectors? Vectors are safer for final return, but we just need to record evals.
        # The evaluator records every problem(x) call.
        # We need to manage the population and potentially the archive for selection.
        # For DE, we just need the population. The archive is useful for "best" mutation or final output.
        # Since return value is ignored, we just need to make good evaluations.
        
        # We will run DE generations.
        # Each generation consumes N evaluations.
        # Max generations = (budget - N) // N
        
        F_mut = 0.8
        CR = 0.9
        
        while used < budget:
            # If remaining budget is less than N, we can still evaluate fewer
            # but DE usually works in batches. Let's just do full batches if possible.
            # If remaining < N, we break or do partial. 
            # Given 200 budget and N=10, 190 remaining -> 19 gens. Fits perfectly.
            
            # Generate trial vectors
            for i in range(N):
                if used >= budget:
                    break
                    
                # Select distinct random indices r1, r2, r3 != i
                while True:
                    r1 = np.random.randint(N)
                    r2 = np.random.randint(N)
                    r3 = np.random.randint(N)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break
                
                # DE/rand/1/bin
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(n)
                u = np.empty(n)
                for j in range(n):
                    if np.random.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = X[i][j]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                fu = problem(u)
                used += 1
                
                # Selection
                # If u dominates X[i], replace
                if self._dominates(fu, F[i]):
                    X[i] = u
                    F[i] = fu
                # If X[i] dominates u, keep X[i]
                # If non-dominated, we can keep X[i] to preserve diversity in population
                # or use a strategy to keep the one with better crowding?
                # Simple DE usually just keeps the better one. 
                # To maintain diversity, we might want to keep u if it is non-dominated 
                # and X[i] is "crowded". But simple dominance is standard.
                # Let's stick to standard selection: replace if strictly better.
                # This might collapse diversity. 
                # Alternative: If non-dominated, keep u with prob 0.5? 
                # Or keep the one that is further from the centroid?
                
                # Let's try a diversity-preserving selection:
                # If non-dominated, keep the one that is more "different" from the current population?
                # Too complex. 
                # Let's use a simple rule: If non-dominated, keep u if it improves the spread?
                # Actually, for ZDT6, the front is disconnected/multi-modal? No, ZDT6 is continuous but has a specific shape.
                # Standard DE often works well.
                
                # Let's add a small probability of keeping the new one if non-dominated to encourage exploration
                if not self._dominates(fu, F[i]) and not self._dominates(F[i], fu):
                    if np.random.random() < 0.5:
                        X[i] = u
                        F[i] = fu

        return X

    def _dominates(self, a, b):
        # a dominates b if a <= b in all objectives and a < b in at least one
        return np.all(a <= b) and np.any(a < b)
