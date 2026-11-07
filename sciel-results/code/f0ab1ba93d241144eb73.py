"""New program: Parallel Simulated Annealing with Global Archive."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using Parallel Simulated Annealing.
    
    Strategy:
    1. Initialize a global archive with random points.
    2. Run multiple independent SA chains.
    3. Each chain explores its local neighborhood.
    4. Solutions are added to the global archive if non-dominated.
    5. Acceptance in SA chains uses dominance and a scalarized distance for incomparable points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Global Archive
        archive_X = []
        archive_F = []
        
        used = 0

        def dominates(a, b):
            """a dominates b if a <= b in all objectives and a < b in at least one."""
            return np.all(a <= b) and np.any(a < b)

        def update_archive(x, f):
            """
            Add (x, f) to archive if it is non-dominated by existing archive points.
            Remove any archive points dominated by (x, f).
            """
            # Check if dominated or duplicate
            for af in archive_F:
                if dominates(af, f) or np.allclose(af, f):
                    return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af) and not np.allclose(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            
            return True

        # 1. Initialization Phase
        # Use 20% of budget for initial random sampling to populate archive
        n_init = max(10, self.budget // 5)
        for _ in range(n_init):
            if used >= self.budget:
                break
            x = lo + rng.random(self.dim) * (hi - lo)
            f = problem(x)
            used += 1
            update_archive(x, f)

        # 2. Parallel Simulated Annealing Phase
        # Determine number of chains and steps per chain
        n_chains = 16
        remaining_budget = self.budget - used
        steps_per_chain = max(1, remaining_budget // n_chains)
        
        # Adjust n_chains if steps_per_chain is too small
        if steps_per_chain < 5:
            n_chains = max(1, remaining_budget // 10)
            steps_per_chain = max(1, remaining_budget // n_chains)

        # Initialize chains from archive or random
        chain_states = []
        for i in range(n_chains):
            if used >= self.budget:
                break
            if archive_X and i < len(archive_X):
                # Start from a diverse point in archive
                idx = int(i * len(archive_X) / n_chains)
                x = archive_X[idx].copy()
                f = archive_F[idx].copy()
                chain_states.append((x, f))
            else:
                # Start from random
                x = lo + rng.random(self.dim) * (hi - lo)
                if used < self.budget:
                    f = problem(x)
                    used += 1
                    chain_states.append((x, f))

        n_chains = len(chain_states)
        if n_chains == 0:
            return archive_X

        steps_per_chain = max(1, (self.budget - used) // n_chains)

        # Temperature parameters
        T_start = 1.0
        T_end = 0.01

        # Ideal point for scalarization (component-wise min of archive)
        if archive_F:
            ideal_point = np.min(archive_F, axis=0)
        else:
            ideal_point = np.zeros(self.n_obj)

        for c in range(n_chains):
            x_curr, f_curr = chain_states[c]
            
            # Temperature schedule for this chain
            for step in range(steps_per_chain):
                if used >= self.budget:
                    break
                
                # Cooling
                if steps_per_chain > 1:
                    T = T_start * (T_end / T_start) ** (step / (steps_per_chain - 1))
                else:
                    T = T_start

                # Generate neighbor
                # Gaussian perturbation
                sigma = 0.1 * (hi - lo)
                x_new = x_curr + rng.normal(0, sigma, self.dim)
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Acceptance Logic
                accept = False
                
                if dominates(f_new, f_curr):
                    accept = True
                elif dominates(f_curr, f_new):
                    accept = False
                else:
                    # Non-dominated
                    dist_curr = np.sum(f_curr - ideal_point)
                    dist_new = np.sum(f_new - ideal_point)
                    delta = dist_new - dist_curr
                    if delta < 0:
                        accept = True
                    else:
                        if T > 0:
                            prob = np.exp(-delta / T)
                        else:
                            prob = 0.0
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    x_curr = x_new
                    f_curr = f_new
                    update_archive(x_curr, f_curr)

        return archive_X
# EVOLVE-BLOCK-END
