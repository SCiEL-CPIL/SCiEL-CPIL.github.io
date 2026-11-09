"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using MOPSO (Multi-Objective Particle Swarm Optimization)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Population size
        pop_size = min(40, max(10, self.budget // 5))
        
        # Initialize positions
        X = lo + span * rng.random((pop_size, self.dim))
        
        # Initialize velocities
        V = span * (rng.random((pop_size, self.dim)) - 0.5) * 0.2
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Personal bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Velocity limits
        V_max = 0.2 * span
        V_min = -V_max

        # MOPSO parameters
        w = 0.7  # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            nd_idx = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            return np.array(nd_idx)

        while evals < self.budget:
            # Determine how many evaluations we can do in this step
            # We can do up to pop_size, but not exceed budget
            steps_to_take = min(pop_size, self.budget - evals)
            
            # Update Personal Bests
            for i in range(pop_size):
                if dominates(F[i], Pbest_F[i]):
                    Pbest_F[i] = F[i]
                    Pbest_X[i] = X[i]
            
            # Get Non-Dominated Set from current population
            nd_idx = non_dominated_indices(F)
            
            if len(nd_idx) == 0:
                gbest_idx = 0
            else:
                # Select Gbest: Random from ND set
                gbest_idx = nd_idx[rng.integers(len(nd_idx))]
            
            Gbest_X = X[gbest_idx]
            
            # Update Velocities and Positions for the whole population
            # Even if we only evaluate a subset, we update all to maintain swarm dynamics?
            # No, we must only evaluate the ones we will keep/record.
            # If we update all, we need to evaluate all.
            # If steps_to_take < pop_size, we should only update/evaluate that many.
            # But MOPSO relies on the whole swarm.
            # Strategy: If budget allows full pop, do full pop.
            # If not, do a partial update?
            # Simpler: If remaining budget < pop_size, just do random local search or stop.
            # But we must use the budget.
            
            if steps_to_take == pop_size:
                r1 = rng.random((pop_size, 1))
                r2 = rng.random((pop_size, 1))
                
                V = w * V + c1 * r1 * (Pbest_X - X) + c2 * r2 * (Gbest_X - X)
                V = np.clip(V, V_min, V_max)
                
                X_new = X + V
                X_new = np.clip(X_new, lo, hi)
                
                F_new = np.array([problem(x) for x in X_new])
                evals += pop_size
                
                X = X_new
                F = F_new
            else:
                # Partial step: Pick 'steps_to_take' particles to update
                # Or just do single-step updates for the remaining budget
                # To keep it simple and safe, we can just do single evaluations
                # using the same logic but one by one.
                for _ in range(steps_to_take):
                    # Pick a random particle to update
                    i = rng.integers(pop_size)
                    
                    r1 = rng.random(1)
                    r2 = rng.random(1)
                    
                    Vi = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (Gbest_X - X[i])
                    Vi = np.clip(Vi, V_min, V_max)
                    
                    Xi_new = X[i] + Vi
                    Xi_new = np.clip(Xi_new, lo, hi)
                    
                    Fi_new = problem(Xi_new)
                    evals += 1
                    
                    # Update state
                    X[i] = Xi_new
                    F[i] = Fi_new
                    V[i] = Vi
                    
                    # Update Pbest
                    if dominates(Fi_new, Pbest_F[i]):
                        Pbest_F[i] = Fi_new
                        Pbest_X[i] = Xi_new

        return X
# EVOLVE-BLOCK-END
