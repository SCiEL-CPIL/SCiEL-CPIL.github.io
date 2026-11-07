"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Multi-Objective Particle Swarm Optimization (MOPSO)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Swarm parameters
        swarm_size = 20
        max_archive_size = 50
        
        # Initialize positions and velocities
        X = rng.uniform(lo, hi, size=(swarm_size, self.dim))
        V = rng.uniform(-0.1 * span, 0.1 * span, size=(swarm_size, self.dim))
        
        # Personal bests
        pbest_X = X.copy()
        pbest_F = np.array([problem(x) for x in X])
        used = swarm_size
        
        # External Archive
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for k in range(self.n_obj):
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_range = F[order[-1], k] - F[order[0], k]
                if f_range == 0:
                    continue
                for m in range(1, n - 1):
                    cd[order[m]] += (F[order[m + 1], k] - F[order[m - 1], k]) / f_range
            return cd

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            new_archive_X = []
            new_archive_F = []
            is_dominated = False
            
            for j in range(len(archive_X)):
                if dominates(archive_F[j], f):
                    is_dominated = True
                    break
                elif dominates(f, archive_F[j]):
                    continue
                else:
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
            
            if not is_dominated:
                new_archive_X.append(x)
                new_archive_F.append(f)
            
            if len(new_archive_X) > max_archive_size:
                F_arr = np.array(new_archive_F)
                cd = crowding_distance(F_arr)
                order = np.argsort(cd)
                keep_indices = order[max_archive_size - len(new_archive_X):]
                archive_X = [new_archive_X[i] for i in keep_indices]
                archive_F = [new_archive_F[i] for i in keep_indices]
            else:
                archive_X = new_archive_X
                archive_F = new_archive_F

        # Initialize archive with initial population
        for i in range(swarm_size):
            update_archive(X[i], pbest_F[i])
            
        # PSO Loop
        w = 0.9
        c1 = 1.5
        c2 = 1.5
        
        while used < self.budget:
            # Select gbest from archive using crowding distance
            if len(archive_X) > 0:
                F_arr = np.array(archive_F)
                cd = crowding_distance(F_arr)
                # Pick the point with the largest crowding distance (most diverse)
                gbest_idx = np.argmax(cd)
                gbest = np.array(archive_X[gbest_idx])
            else:
                gbest = np.zeros(self.dim)
                
            for i in range(swarm_size):
                if used >= self.budget:
                    break
                
                # Update velocity
                r1 = rng.random()
                r2 = rng.random()
                V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -0.5 * span, 0.5 * span)
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                f = problem(X[i])
                used += 1
                
                # Update personal best
                if dominates(f, pbest_F[i]):
                    pbest_F[i] = f
                    pbest_X[i] = X[i].copy()
                
                # Update archive
                update_archive(X[i], f)
                
        return np.array(archive_X) if archive_X else np.zeros((1, self.dim))
# EVOLVE-BLOCK-END
