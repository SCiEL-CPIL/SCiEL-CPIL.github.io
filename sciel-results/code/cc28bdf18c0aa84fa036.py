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
        deadline = 285.0  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        
        # Helper: Check if a point is dominated by any in the archive
        # Returns True if dominated, False otherwise
        def is_dominated(point, archive_F_list):
            for f in archive_F_list:
                if np.all(f <= point) and np.any(f < point):
                    return True
            return False
        
        # Helper: Add a point to the archive if non-dominated, remove dominated points
        def update_archive(x, f):
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return False
            
            # Remove points dominated by new point
            new_archive_F = []
            new_archive_X = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_F.append(archive_F[i])
                    new_archive_X.append(archive_X[i])
            
            # Add new point
            new_archive_F.append(f)
            new_archive_X.append(x)
            
            archive_F.clear()
            archive_F.extend(new_archive_F)
            archive_X.clear()
            archive_X.extend(new_archive_X)
            
            return True
        
        # Helper: Compute hypervolume contribution of a point to the archive
        # This is an approximation: area of the box dominated by the point and not by others
        def hypervolume_contribution(f, archive_F_list, ref):
            # For 2 objectives, the contribution is the area of the rectangle
            # [f[0], ref[0]] x [f[1], ref[1]] minus areas covered by other points
            # Simple approximation: just use the box area, but this is not exact.
            # For better accuracy, we can compute the exact contribution for 2D.
            # In 2D, the contribution of point p is (ref[0]-p[0])*(ref[1]-p[1])
            # minus the parts covered by other points that are better in at least one dimension.
            # Actually, for hypervolume indicator, the contribution of a point p to the set S
            # is the volume of the region dominated by p and not dominated by any other point in S.
            
            # For 2D, we can compute this exactly:
            # Sort all points (including p) by first objective.
            # The contribution of p is the area of the rectangle from p to ref,
            # but clipped by the Pareto front.
            
            # A simpler way: compute the hypervolume of the set S, and the hypervolume of S \ {p}.
            # Contribution = HV(S) - HV(S \ {p}).
            # But we don't want to recompute HV every time.
            
            # Let's use a simpler heuristic for selection: 
            # We'll just use the "uncovered" area approximation.
            # For now, let's return a simple value that favors points in the "center" of the front.
            # Actually, let's implement exact 2D hypervolume contribution.
            
            points = list(archive_F_list) + [f]
            # Filter points that are better than ref
            points = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            
            if not points:
                return 0.0
            
            # Sort by first objective
            points.sort(key=lambda x: x[0])
            
            # Compute hypervolume of the set
            # In 2D, we can compute HV by sorting by x and integrating the min y
            # Actually, the standard way is:
            # Sort by x. For each point, the contribution is (x_{i+1} - x_i) * (ref_y - y_i)
            # where x_{n+1} = ref_x.
            
            def compute_hv(points):
                if not points:
                    return 0.0
                points = sorted(points, key=lambda x: x[0])
                hv = 0.0
                for i in range(len(points)):
                    x1 = points[i][0]
                    x2 = points[i+1][0] if i + 1 < len(points) else ref[0]
                    y1 = points[i][1]
                    if x2 > x1 and y1 < ref[1]:
                        hv += (x2 - x1) * (ref[1] - y1)
                return hv
            
            hv_with = compute_hv(points)
            
            # Compute HV without f
            points_without = [p for p in points if p is not f]
            # Note: f might be equal to another point, but we assume unique points for now.
            # Actually, we should remove one instance of f.
            # Since we added f to the list, we can just filter by identity or value.
            # Let's filter by value, but if there are duplicates, this is tricky.
            # For now, assume f is unique.
            points_without = [p for p in points if not (p[0] == f[0] and p[1] == f[1])]
            
            hv_without = compute_hv(points_without)
            
            return hv_with - hv_without

        ref = np.array([3000.0, 0.06])
        
        # Phase 1: Sobol Initialization
        n_init = min(50, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        X_init = sampler.random(n_init)
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            update_archive(x, f)
        
        # Phase 2: HypE-like Evolution
        # We maintain a population and select parents based on hypervolume contribution.
        # We'll use a simple population of size 20, but update the archive.
        
        pop_size = 20
        # Initialize population from archive or random
        if len(archive_X) >= pop_size:
            # Sample from archive
            indices = np.random.choice(len(archive_X), pop_size, replace=False)
            pop_X = np.array([archive_X[i] for i in indices])
        else:
            # Use all archive points and fill with random
            pop_X = np.array(archive_X) if archive_X else np.empty((0, dim))
            if len(pop_X) < pop_size:
                n_fill = pop_size - len(pop_X)
                X_fill = lo + (hi - lo) * np.random.random((n_fill, dim))
                for x in X_fill:
                    f = problem(x)
                    update_archive(x, f)
                pop_X = np.vstack([pop_X, np.array(archive_X[len(pop_X):])]) if len(archive_X) > len(pop_X) else pop_X
                # Ensure pop_X has pop_size points
                if len(pop_X) < pop_size:
                    X_fill2 = lo + (hi - lo) * np.random.random((pop_size - len(pop_X), dim))
                    pop_X = np.vstack([pop_X, X_fill2])
        
        # Evaluate population
        pop_F = np.array([problem(x) for x in pop_X])
        
        # Main loop
        while True:
            if time.time() > deadline:
                break
            
            # Select parent based on hypervolume contribution
            # Compute contribution for each point in the population
            # This is expensive, so we'll do it every few iterations or use a simpler method.
            # For now, let's just use random selection from the population, but weight by contribution.
            
            # Actually, let's compute contributions for the population points.
            # We'll use the archive as the set for contribution calculation.
            # This is still expensive. Let's use a simpler approach:
            # Select a parent randomly from the population.
            
            p_idx = np.random.randint(len(pop_X))
            x_parent = pop_X[p_idx]
            
            # Perturb the parent to create an offspring
            # Use a Gaussian perturbation
            sigma = 0.1 * (hi - lo)
            x_offspring = x_parent + np.random.normal(0, sigma)
            x_offspring = np.clip(x_offspring, lo, hi)
            
            f_offspring = problem(x_offspring)
            update_archive(x_offspring, f_offspring)
            
            # Update population: replace the parent with the offspring if it's better
            # Or just keep the population as is and rely on the archive.
            # Let's replace the parent with the offspring in the population.
            pop_X[p_idx] = x_offspring
            pop_F[p_idx] = f_offspring
            
            # Occasionally, add a new random point to the population to maintain diversity
            if np.random.random() < 0.1:
                x_new = lo + (hi - lo) * np.random.random(dim)
                f_new = problem(x_new)
                update_archive(x_new, f_new)
                # Replace a random point in the population
                idx = np.random.randint(len(pop_X))
                pop_X[idx] = x_new
                pop_F[idx] = f_new
        
        return archive_X
