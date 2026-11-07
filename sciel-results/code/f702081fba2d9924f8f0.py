import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Archive of non-dominated points: list of (x, f)
        archive = []
        
        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f."""
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            """Add (x, f) to archive if non-dominated, remove dominated points."""
            # Check if new point is dominated
            if is_dominated(f, [af for _, af in archive]):
                return False
            
            # Remove points dominated by new point
            new_archive = []
            for px, pf in archive:
                if not (np.all(f <= pf) and np.any(f < pf)):
                    new_archive.append((px, pf))
            
            new_archive.append((x, f))
            return new_archive

        def evaluate_and_update(x):
            nonlocal archive
            f = problem(x)
            # Check if dominated by current archive
            dominated = False
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove points dominated by new point
                archive = [(px, pf) for px, pf in archive if not (np.all(f <= pf) and np.any(f < pf))]
                archive.append((x, f))
                return True
            return False

        # 1. Initialization
        init_size = 20
        for _ in range(init_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate_and_update(x)
            
        budget_used = init_size
        
        # 2. Evolutionary Loop
        # Use a population of parents from the archive
        # To ensure diversity, we pick parents randomly from the archive
        
        while budget_used < self.budget:
            # Select 2 parents from archive
            # If archive is small, just use what we have
            if len(archive) == 0:
                # Should not happen if init worked
                break
                
            # Randomly select 2 distinct parents if possible
            if len(archive) > 1:
                i1, i2 = rng.choice(len(archive), size=2, replace=False)
                p1 = archive[i1][0]
                p2 = archive[i2][0]
            else:
                p1 = archive[0][0]
                p2 = archive[0][0]
                
            # SBX Crossover
            # eta_c = 20
            eta_c = 20.0
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    # Calculate beta
                    if abs(p1[i] - p2[i]) < 1e-14:
                        beta = 1.0
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                        
                    child1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                    child2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
                else:
                    child1[i] = p1[i]
                    child2[i] = p2[i]
            
            # Polynomial Mutation
            # eta_m = 20
            eta_m = 20.0
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1.0))
                    else:
                        delta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_m + 1.0))
                    
                    delta = (hi[i] - lo[i]) * (delta_q - 1.0) if u > 0.5 else (hi[i] - lo[i]) * (1.0 - delta_q)
                    # Wait, standard polynomial mutation:
                    # delta1 = (2*u)^(1/(eta+1))
                    # delta2 = (1/(2*(1-u)))^(1/(eta+1))
                    # if u < 0.5: delta = delta1 - 1
                    # else: delta = 1 - delta2
                    # new_val = val + (hi-lo) * delta
                    
                    if u < 0.5:
                        delta = (2*u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_m + 1.0))
                    
                    child1[i] = child1[i] + (hi[i] - lo[i]) * delta
                if rng.random() < 0.5:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2*u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_m + 1.0))
                    child2[i] = child2[i] + (hi[i] - lo[i]) * delta
            
            # Clip to bounds
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate child1
            evaluate_and_update(child1)
            budget_used += 1
            
            if budget_used < self.budget:
                # Evaluate child2
                evaluate_and_update(child2)
                budget_used += 1
                
        return None
