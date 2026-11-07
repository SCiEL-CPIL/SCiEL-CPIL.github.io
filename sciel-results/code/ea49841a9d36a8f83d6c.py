import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 280s to be safe
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # MOPSO Parameters
        # Budget is 200. 
        # We need to balance exploration and exploitation.
        # Small swarm for high dim (12) to fit in budget? 
        # If swarm size is N, we use N evals for init.
        # Then (Budget - N) iterations.
        # Let's try N=10. 190 iterations.
        n_particles = 10
        
        # Initialize positions
        X = lo + range_ * np.random.random((n_particles, self.dim))
        
        # Evaluate initial positions
        F = np.array([problem(x) for x in X])
        used = n_particles
        
        # Initialize velocities
        # Velocity bounds: typically +/- 0.1 * range or similar
        V = (hi - lo) * 0.1 * np.random.random((n_particles, self.dim))
        
        # Personal bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Archive: List of non-dominated points (objective vectors)
        # We will maintain a set of non-dominated points.
        # To keep it efficient, we limit the archive size or just keep all non-dominated.
        # With budget 200, max 200 points. O(N^2) dominance check is fine.
        archive = []
        
        def dominates(a, b):
            # a dominates b if a is better or equal in all and strictly better in at least one
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(f, f_set):
            # Check if f is non-dominated by any in f_set
            for j in range(len(f_set)):
                if dominates(f_set[j], f):
                    return False
            return True
            
        def update_archive(f, x):
            # f: (n_obj,)
            # x: (dim,) - not strictly needed for archive if we only care about objectives for HV, 
            # but MOPSO needs leaders from archive. We store objectives in archive for dominance.
            # Actually, we just need the objective vectors for the archive to select leaders.
            new_archive = []
            for i in range(len(archive)):
                if not dominates(f, archive[i]):
                    new_archive.append(archive[i])
            if not is_non_dominated(f, new_archive):
                return new_archive
            new_archive.append(f)
            return new_archive

        # Initialize archive
        for i in range(n_particles):
            archive = update_archive(F[i], X[i])
            
        # Reference point for HI (Hypervolume Indicative)
        # The task says ref point is [3.0, 5.0, 7.0]
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # In MOPSO, we select a leader from the archive.
        # Common strategy: Pick the point in the archive that maximizes the HI value
        # or just pick randomly from non-dominated archive to maintain diversity.
        # Given the small budget, random selection from the archive is robust and simple.
        # A more sophisticated approach is to pick the point that is farthest from other archive points
        # or has the highest contribution to hypervolume.
        # Let's use a simple "niche" preservation: pick the archive point that is least crowded
        # or just random. Random is often surprisingly effective for small budgets.
        # Let's try picking the point with the smallest sum of distances to other archive points
        # to encourage exploring sparse regions.
        
        def select_leader(archive):
            if len(archive) == 0:
                return None
            if len(archive) == 1:
                return archive[0]
            
            # Calculate distance to other points for each point in archive
            # Use normalized distance
            # To save time, simple random is O(1). 
            # Crowding is O(N^2). With N<=200, it's fast.
            # But we have 190 iterations. 190 * 200^2 = 7.6M ops. Fine.
            
            # Let's just pick randomly to keep it simple and fast, 
            # unless the archive is very small, then pick the one with best "spread".
            # Actually, let's use a simple heuristic: pick the point that is 
            # furthest from the centroid of the archive? No, that pushes to extremes.
            # Let's stick to random selection from the archive. It's standard for small scale MOPSO.
            idx = np.random.randint(len(archive))
            return archive[idx]

        # Inertia weight decay
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Update inertia weight
            progress = used / self.budget
            w = w_max - (w_max - w_min) * progress
            
            # Select leader (Global Best) from Archive
            leader_F = select_leader(archive)
            
            # If archive is empty (shouldn't happen), fallback to random
            if leader_F is None:
                leader_F = ref_point # Fallback, though shouldn't happen
            
            for i in range(n_particles):
                if time.time() > deadline:
                    break
                
                # Update velocity
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # We need the position corresponding to leader_F?
                # Standard MOPSO updates velocity using Pbest and Leader.
                # Leader is an objective vector. We don't have its position.
                # This is a known issue with MOPSO. 
                # Solution: We store positions in the archive? 
                # If we store positions, the archive is in decision space.
                # But dominance is in objective space.
                # Usually, we keep a separate archive of non-dominated *solutions* (X, F).
                # Let's modify archive to store (X, F) pairs.
                
                pass # Placeholder, need to restructure archive

        # RESTRUCTURE: Archive must store X as well to update velocity towards leader.
        
        # Let's restart the logic inside __call__ with corrected structure.
        # Since I can't edit the code above easily in my head without rewriting, 
        # I will write the full correct implementation below.
        
        return None # Placeholder, actual code below

    # Re-implementing __call__ properly
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        n_particles = 10
        
        # Initialize
        X = lo + range_ * np.random.random((n_particles, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_particles
        
        V = (hi - lo) * 0.1 * np.random.random((n_particles, self.dim))
        
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Archive stores list of dicts or tuples (x, f)
        # We maintain a list of non-dominated points.
        archive = []
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(f, f_set):
            for j in range(len(f_set)):
                if dominates(f_set[j], f):
                    return False
            return True
            
        def add_to_archive(x, f):
            # Remove points dominated by f
            new_archive = []
            for item in archive:
                if not dominates(f, item['f']):
                    new_archive.append(item)
            
            # Check if f is dominated by any in new_archive
            if is_non_dominated(f, [item['f'] for item in new_archive]):
                new_archive.append({'x': x, 'f': f})
            return new_archive

        # Initialize archive
        for i in range(n_particles):
            archive = add_to_archive(X[i], F[i])
            
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5
        c2 = 1.5
        
        # Velocity limits
        v_max = 0.5 * range_
        v_min = -0.5 * range_
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            progress = used / self.budget
            w = w_max - (w_max - w_min) * progress
            
            # Select leader from archive
            # Strategy: Pick the archive point that is "best" in some sense.
            # For hypervolume, picking the one with the largest "niche" is good.
            # Simple approach: Random selection from archive.
            if len(archive) > 0:
                leader_idx = np.random.randint(len(archive))
                leader = archive[leader_idx]
            else:
                # Fallback if archive empty (should not happen)
                leader = {'x': X[0], 'f': F[0]}
                
            for i in range(n_particles):
                if time.time() > deadline:
                    break
                
                # Update Velocity
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (leader['x'] - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], v_min, v_max)
                
                # Update Position
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F[i] = problem(X[i])
                used += 1
                
                # Update Pbest
                if dominates(F[i], Pbest_F[i]):
                    Pbest_F[i] = F[i].copy()
                    Pbest_X[i] = X[i].copy()
                
                # Update Archive
                # Only add if it is non-dominated w.r.t current archive
                # Optimization: Check against Pbest_F[i] first? No, must check against archive.
                # To speed up, we can check if it dominates any archive point or is non-dominated.
                # add_to_archive does this.
                # We can optimize by checking if F[i] is dominated by Pbest_F[i] (it isn't, by definition of Pbest update logic if we update Pbest first? 
                # Actually, if F[i] dominates Pbest_F[i], Pbest is updated. 
                # If F[i] is dominated by Pbest_F[i], it is likely dominated by archive points? Not necessarily, but likely.
                # Let's just call add_to_archive. It's O(A). A <= 200.
                # 190 iterations * 10 particles * 200 checks = 380k checks. Very fast.
                archive = add_to_archive(X[i], F[i])
                
                # Optional: If archive gets too large, we might need to truncate.
                # But with budget 200, max 200 points. No truncation needed.

        # Return the non-dominated points from the archive
        # The task says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # But the evaluator records "Every returned objective vector". 
        # Wait, "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This phrasing is slightly ambiguous. 
        # "Every returned objective vector is recorded" -> Does it mean every vector returned by `problem(x)`?
        # "and the return value of __call__ is ignored."
        # Usually in these tasks, the evaluator wraps `problem` to record every call's output.
        # "Every returned objective vector is recorded" likely refers to the outputs of `problem(x)`.
        # The scoring is "exact hypervolume of all recorded vectors".
        # This implies ALL 200 evaluated points are recorded, not just the final non-dominated set.
        # If so, the final return value doesn't matter for the score, only the calls to `problem`.
        # However, the contract says: "Every returned objective vector is recorded... Scoring... hypervolume of all recorded vectors".
        # This confirms: The evaluator records every `problem(x)` call result.
        # So we just need to make sure we call `problem` exactly 200 times (or <= 200) and that the set of points evaluated yields high hypervolume.
        # We don't need to return the Pareto front specifically, but it's good practice.
        
        return F # Return all evaluated points, though ignored.
