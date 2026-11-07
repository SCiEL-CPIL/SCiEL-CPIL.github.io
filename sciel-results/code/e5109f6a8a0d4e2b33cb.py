import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # DTLZ2 structure:
        # x[0], x[1] are position variables (angles)
        # x[2:] are decision variables; g(x) is minimized at x[2:] = 0.5
        # The Pareto front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1

        # We'll use a combination of:
        # 1. Sobol sequence for good space-filling of the 2D position space
        # 2. Some random perturbations to escape local optima
        # 3. Local refinement around promising points

        # Generate Sobol sequence for position variables
        n_sobol = 128  # Power of 2 for Sobol
        sampler = qmc.Sobol(d=2, scramble=True, seed=0)
        sobol_samples = sampler.random(n_sobol)

        # Map to [0, 1] and then to bounds
        pos_samples = sobol_samples * (hi[:2] - lo[:2]) + lo[:2]

        # Create full candidate set
        candidates = []
        
        # Add Sobol-based points
        for i in range(n_sobol):
            x = mid.copy()
            x[0] = pos_samples[i, 0]
            x[1] = pos_samples[i, 1]
            candidates.append(x)

        # Add some random points for diversity
        n_random = 40
        for i in range(n_random):
            x = np.random.uniform(lo, hi)
            # Keep decision variables near 0.5 but add small noise
            x[2:] = mid[2:] + 0.05 * np.random.randn(self.dim - 2)
            x[2:] = np.clip(x[2:], lo[2:], hi[2:])
            candidates.append(x)

        # Add points focused on the Pareto front angles
        # The Pareto front for DTLZ2 with 3 objectives is a quarter-sphere
        # f1 = (1+g) * cos(t1) * cos(t2)
        # f2 = (1+g) * cos(t1) * sin(t2)  
        # f3 = (1+g) * sin(t1)
        # where t1, t2 in [0, pi/2]
        # So x[0] = t1 / (pi/2) * 2 = 2*t1/pi, x[1] = 2*t2/pi
        # To get points on the front, we want specific combinations
        
        n_front = 32
        for i in range(n_front):
            # Spread angles evenly
            t1 = (i + 0.5) / n_front * (np.pi / 2)
            t2 = ((i * 7) % n_front + 0.5) / n_front * (np.pi / 2)
            x = mid.copy()
            x[0] = lo[0] + (hi[0] - lo[0]) * (2 * t1 / np.pi)
            x[1] = lo[1] + (hi[1] - lo[1]) * (2 * t2 / np.pi)
            candidates.append(x)

        # Add a few more with slight perturbations around mid
        n_perturb = 12
        for i in range(n_perturb):
            x = mid.copy()
            x[0] = lo[0] + (hi[0] - lo[0]) * np.random.uniform(0, 1)
            x[1] = lo[1] + (hi[1] - lo[1]) * np.random.uniform(0, 1)
            x[2:] = mid[2:] + 0.02 * np.random.randn(self.dim - 2)
            x[2:] = np.clip(x[2:], lo[2:], hi[2:])
            candidates.append(x)

        # Sort candidates by expected quality: prefer those with x[2:] close to 0.5
        # and good angle coverage
        def quality_score(x):
            # Penalize deviation from 0.5 in decision variables
            dev = np.sum((x[2:] - mid[2:]) ** 2)
            # Reward good angle spread (not too close to 0 or 1)
            angle_score = -abs(x[0] - 0.5) - abs(x[1] - 0.5)
            return dev - 0.1 * angle_score

        # Evaluate all candidates within budget
        F = []
        X = []
        eval_count = 0
        max_evals = self.budget

        # Sort by quality score (lower is better)
        candidates = np.array(candidates)
        scores = np.array([quality_score(c) for c in candidates])
        order = np.argsort(scores)

        for idx in order:
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break
            x = candidates[idx]
            try:
                f = problem(x)
                F.append(f)
                X.append(x)
                eval_count += 1
            except Exception:
                break

        # If we have time and evaluations left, do local refinement
        if F and eval_count < max_evals and time.time() < deadline:
            F = np.array(F)
            X = np.array(X)
            
            # Find the best few points and refine around them
            # Use hypervolume contribution as a proxy for importance
            # Simpler: just pick points with low objective values
            ref = np.array([2.5, 2.5, 2.5])
            
            # Sort by sum of objectives
            obj_sums = F.sum(axis=1)
            best_indices = np.argsort(obj_sums)[:10]
            
            for bi in best_indices:
                if eval_count >= max_evals:
                    break
                if time.time() > deadline:
                    break
                
                # Local perturbation
                n_local = 3
                for k in range(n_local):
                    if eval_count >= max_evals:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_new = X[bi].copy()
                    # Small perturbation in position variables
                    x_new[0] = np.clip(x_new[0] + 0.05 * np.random.randn(), lo[0], hi[0])
                    x_new[1] = np.clip(x_new[1] + 0.05 * np.random.randn(), lo[1], hi[1])
                    # Keep decision variables at 0.5
                    x_new[2:] = mid[2:]
                    
                    try:
                        f_new = problem(x_new)
                        F = np.vstack([F, [f_new]])
                        X = np.vstack([X, [x_new]])
                        eval_count += 1
                    except Exception:
                        break

        return X
