"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using LHS on the simplex.
    
    Strategy:
    - Use the known structure of DTLZ4's Pareto front to directly sample points
      that lie on (or very close to) the Pareto front.
    - DTLZ4 with 3 objectives has a Pareto front that is a 2D simplex in objective space.
    - The first 198 evaluations use Latin Hypercube Sampling on this simplex.
    - The final 2 evaluations place points at extreme vertices to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # For DTLZ4 with M=3, n=12:
        # The Pareto front is the unit sphere in the positive orthant.
        # Parameterize using two angles: theta1, theta2 in [0, pi/2]
        # f1 = sin(theta1) * sin(theta2)
        # f2 = cos(theta1) * sin(theta2)
        # f3 = cos(theta2)
        #
        # To construct decision variables that produce these objectives on the Pareto front:
        # For DTLZ4, the position variables are x1, x2 (first M-1=2 vars)
        # The remaining variables x3..x12 control g(x), and on PF, g=0
        # 
        # Key insight: For DTLZ4, if we set x1 = f1, x2 = f2, and x3..x12 = 0,
        # then g = 0 (since all min(x_i, 0.2) = 0 for i>=3), and:
        # f1 = x1, f2 = x2, f3 = (1 - x1^2 - x2^2) * 0 = 0
        # That gives f3 = 0, not what we want.
        #
        # Let me reconsider. The standard DTLZ formulation:
        # f1 = x1
        # f2 = x2
        # f3 = (1 - x1^2 - x2^2) * g(x3..x12)
        #
        # On PF, g=0, so f3 = 0. This means the PF is only where f3=0,
        # which is x1^2 + x2^2 = 1. That's a 1D curve, not 2D.
        #
        # I think the issue is that I'm using the wrong DTLZ variant.
        # Let me use a robust approach: sample in decision space using LHS,
        # then use local search to push points toward the Pareto front.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: LHS in decision space (198 points)
        n_lhs = min(198, budget)
        n_remaining = budget - n_lhs
        
        def lhs_sample(n, dim, rng):
            samples = np.zeros((n, dim))
            for d in range(dim):
                intervals = np.linspace(0, 1, n + 1)
                points = intervals[:-1] + rng.random(n) * (intervals[1] - intervals[0])
                perm = rng.permutation(n)
                samples[:, d] = points[perm]
            return samples
        
        X_lhs = lhs_sample(n_lhs, dim, rng)
        F_lhs = np.array([problem(x) for x in X_lhs])
        
        # Phase 2: Local search to improve each point toward Pareto front
        # For each point, try small perturbations to reduce objectives
        X_all = X_lhs.copy()
        F_all = F_lhs.copy()
        
        # Sort by sum of objectives to prioritize worst points
        sums = np.sum(F_all, axis=1)
        order = np.argsort(sums)[::-1]  # worst first
        
        # Use remaining budget for local search
        evals_used = n_lhs
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for idx in order:
            if evals_used >= budget:
                break
            
            best_x = X_all[idx].copy()
            best_f = F_all[idx].copy()
            best_sum = np.sum(best_f)
            
            for step in step_sizes:
                if evals_used >= budget:
                    break
                
                # Try random directions
                for _ in range(3):
                    if evals_used >= budget:
                        break
                    
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    # Try both directions
                    for sign in [1, -1]:
                        if evals_used >= budget:
                            break
                        
                        new_x = best_x + sign * step * (hi - lo) * direction
                        new_x = np.clip(new_x, lo, hi)
                        
                        new_f = problem(new_x)
                        evals_used += 1
                        
                        if np.sum(new_f) < best_sum:
                            best_x = new_x
                            best_f = new_f
                            best_sum = np.sum(new_f)
            
            X_all[idx] = best_x
            F_all[idx] = best_f
        
        # Phase 3: Place vertices at the end if budget allows
        # The three extreme points of the Pareto front (in objective space)
        # correspond to decision vectors that minimize individual objectives
        
        if evals_used < budget:
            # Try to find points near each vertex
            # Vertex 1: minimize f1 -> x1=0, x2=1, x3..x12=0 (gives f1=0, f2=1, f3=0)
            # Vertex 2: minimize f2 -> x1=1, x2=0, x3..x12=0 (gives f1=1, f2=0, f3=0)
            # Vertex 3: minimize f3 -> x1=0, x2=0, x3..x12=0 (gives f1=0, f2=0, f3=1)
            
            vertices = [
                np.array([0.0, 1.0] + [0.0] * (dim - 2)),
                np.array([1.0, 0.0] + [0.0] * (dim - 2)),
                np.array([0.0, 0.0] + [0.0] * (dim - 2)),
            ]
            
            for v in vertices:
                if evals_used >= budget:
                    break
                
                v_clipped = np.clip(v, lo, hi)
                fv = problem(v_clipped)
                evals_used += 1
                
                # Replace the worst point
                worst_idx = int(np.argmax(np.sum(F_all, axis=1)))
                X_all[worst_idx] = v_clipped
                F_all[worst_idx] = fv
        
        return X_all
# EVOLVE-BLOCK-END
