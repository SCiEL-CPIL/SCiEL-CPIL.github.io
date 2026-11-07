"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective Particle Swarm Optimization (PSO) with a hypervolume-aware archive.
    Uses a dynamic population and velocity-based exploration/exploitation balance.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop if we exceed 280 seconds
        deadline = start_time + 280.0
        
        # Parse bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_range = hi - lo

        # PSO Parameters
        # We don't strictly need a fixed population size if we use an archive,
        # but we need a set of active particles to drive the swarm.
        # Let's use a smaller swarm size to allow for more iterations within budget,
        # or a larger one for better coverage? 
        # Given 200 budget, a swarm of size 10-20 is reasonable.
        swarm_size = 15
        # Max velocity is typically 20% of the range
        max_vel = 0.2 * bounds_range
        
        # Initialize Swarm
        # Use a deterministic seed for reproducibility if needed, 
        # but the evaluator sets np.random.seed. 
        # We should use np.random directly or a local rng. 
        # The prompt says "evaluator calls np.random.seed(seed)". 
        # So we can rely on the global numpy random state.
        
        # Positions
        X = lo + (hi - lo) * np.random.rand(swarm_size, self.dim)
        
        # Velocities
        V = np.random.rand(swarm_size, self.dim) * (2 * max_vel) - max_vel
        
        # Evaluate initial swarm
        F = np.array([problem(x) for x in X])
        used = swarm_size
        
        # Personal Bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Global Best (Archive)
        # We maintain a list of non-dominated solutions.
        # For efficiency with small budgets, a simple list is fine.
        archive_X = []
        archive_F = []
        
        # Add initial non-dominated points to archive
        for i in range(swarm_size):
            if self._is_non_dominated(F[i], archive_F):
                archive_X.append(X[i])
                archive_F.append(F[i])
        
        # If archive is empty (shouldn't happen with minimization and distinct points), 
        # just take the first one
        if not archive_F:
            archive_X.append(X[0])
            archive_F.append(F[0])
            
        # Global Best Selection Strategy:
        # Instead of a single global best, we pick from the archive.
        # A common strategy is to pick the point in the archive that maximizes 
        # the hypervolume contribution or is furthest from others.
        # For simplicity and speed, we can pick randomly from the archive,
        # or the one with the best "spread" metric.
        # Let's use a simple crowding distance or just random selection from archive
        # to encourage exploration.
        
        def select_global_best(archive_F):
            if len(archive_F) == 1:
                return 0
            # Pick the point in the archive that is "least crowded" or random.
            # Random is often effective in early stages.
            # Let's pick the one with the highest objective sum (worst) to push exploration?
            # No, PSO usually converges to best. 
            # Let's pick a random index from the archive to maintain diversity.
            return np.random.randint(len(archive_F))

        gbest_idx = select_global_best(archive_F)
        Gbest_X = np.array(archive_X[gbest_idx])
        
        # Main Loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Update PSO parameters
            # Inertia weight: linearly decreasing from 0.9 to 0.4
            progress = used / self.budget
            w = 0.9 - 0.5 * progress
            c1 = 1.5  # Cognitive
            c2 = 1.5  # Social
            
            for i in range(swarm_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select global best for this particle (from archive)
                gbest_idx = select_global_best(archive_F)
                Gbest_X = np.array(archive_X[gbest_idx])
                
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (Gbest_X - X[i])
                
                # Clamp velocity
                V[i] = np.clip(V[i], -max_vel, max_vel)
                
                # Update position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F_new = problem(X[i])
                used += 1
                
                # Update Personal Best
                if self._dominates(F_new, Pbest_F[i]):
                    Pbest_X[i] = X[i].copy()
                    Pbest_F[i] = F_new.copy()
                
                # Update Archive
                # If new point is non-dominated, add to archive
                # Then remove any archived points dominated by the new point
                if self._is_non_dominated(F_new, archive_F):
                    # Check if it dominates any existing archive points
                    new_archive_X = []
                    new_archive_F = []
                    for j in range(len(archive_F)):
                        if not self._dominates(F_new, archive_F[j]):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    
                    new_archive_X.append(X[i].copy())
                    new_archive_F.append(F_new.copy())
                    
                    archive_X = new_archive_X
                    archive_F = new_archive_F
            
        return X

    def _dominates(self, f1, f2):
        """
        Returns True if f1 dominates f2.
        f1 <= f2 for all objectives, and f1 < f2 for at least one.
        """
        if np.all(f1 <= f2) and np.any(f1 < f2):
            return True
        return False

    def _is_non_dominated(self, f_new, archive_F):
        """
        Returns True if f_new is non-dominated by any point in archive_F.
        """
        for f_arch in archive_F:
            if self._dominates(f_arch, f_new):
                return False
        return True
# EVOLVE-BLOCK-END
