"""Multi-objective optimizer on DTLZ7 using MOEA/D (Decomposition)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        eval_count = 0
        all_points = []
        all_objectives = []
        
        def eval_point(x):
            nonlocal eval_count
            eval_count += 1
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_objectives.append(f)
            return f

        # Population size
        N = 40
        
        # Generate weight vectors in the simplex for n_obj objectives
        # For 3 objectives, weights are (w1, w2, w3) with w1+w2+w3=1, wi >= 0
        # We can generate points on the 2D simplex
        weights = []
        # Use a grid or random points. Random points on simplex are good for diversity.
        # To ensure coverage, we can use a regular grid if N allows, or random.
        # Let's use random points on the simplex for simplicity, but ensure some extremes.
        # Actually, a systematic generation is better.
        # For 3 objectives, we can use:
        # for i in range(N):
        #    u = rng.random()
        #    v = rng.random()
        #    ... 
        # Simpler: generate random vectors and normalize.
        raw_weights = rng.random((N, self.n_obj))
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        # Initialize population
        X = np.zeros((N, self.dim))
        F = np.zeros((N, self.n_obj))
        
        # Ideal point (lower bound of objectives seen so far)
        # Initialize to infinity
        ideal = np.full(self.n_obj, np.inf)
        
        for i in range(N):
            X[i] = lo + span * rng.random(self.dim)
            F[i] = eval_point(X[i])
            ideal = np.minimum(ideal, F[i])
            
        # MOEA/D Loop
        # We have budget - N evaluations left.
        # Each iteration updates one individual.
        # Number of iterations = (budget - N) // N * N ? No, we just run until budget is exhausted.
        # We will iterate through individuals in a round-robin fashion.
        
        # DE parameters
        F_factor = 0.5
        CR = 0.5
        
        # Neighbor selection: For simplicity, we pick random distinct indices from the whole population.
        # In standard MOEA/D, we pick from neighbors. With N=40, random is fine and simpler.
        
        current_idx = 0
        while eval_count < self.budget:
            i = current_idx % N
            current_idx += 1
            
            # Select 3 distinct parents r1, r2, r3 different from i
            # We need to ensure r1, r2, r3 are distinct and != i
            # Simple way: choose 3 random indices from range(N), ensuring no duplicates and != i
            # For N=40, this is fast.
            candidates = [j for j in range(N) if j != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
            
            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            v = X[r1] + F_factor * (X[r2] - X[r3])
            
            # Crossover: Binary DE crossover
            j_rand = rng.integers(self.dim)
            trial = X[i].copy()
            for j in range(self.dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = eval_point(trial)
            
            # Update Ideal Point
            # Check if trial improves the ideal point
            for k in range(self.n_obj):
                if f_trial[k] < ideal[k]:
                    ideal[k] = f_trial[k]
                    # Note: If ideal changes, the Tchebycheff values of ALL existing points change.
                    # Recalculating Tchebycheff for all N points every time ideal updates is O(N*M).
                    # This is acceptable for N=40.
            
            # Calculate Tchebycheff aggregation for trial and current incumbent
            # g(x, lambda) = max_i { lambda_i * | f_i(x) - z_i^* | }
            
            # We need to compare trial and X[i] using weight lambda_i
            # But wait, if ideal updated, X[i]'s score might have changed.
            # It's safer to just compute the score for trial and X[i] with the CURRENT ideal.
            
            lambda_i = weights[i]
            
            # Compute score for trial
            # f_trial is shape (M,), ideal is shape (M,), lambda_i is shape (M,)
            scores_trial = lambda_i * np.abs(f_trial - ideal)
            score_trial = np.max(scores_trial)
            
            # Compute score for incumbent X[i]
            scores_inc = lambda_i * np.abs(F[i] - ideal)
            score_inc = np.max(scores_inc)
            
            # Accept if trial is better
            if score_trial < score_inc:
                X[i] = trial
                F[i] = f_trial
                
        return all_points
# EVOLVE-BLOCK-END
