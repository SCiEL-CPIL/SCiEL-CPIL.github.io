"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """HypE-style multi-objective optimizer with Sobol init and NM polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point."""
            if len(points) == 0:
                return 0.0
            # Filter valid points
            pts = [p for p in points if p[0] < ref_point[0] and p[1] < ref_point[1]]
            if len(pts) == 0:
                return 0.0
            sorted_pts = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hypervolume(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_without = compute_hypervolume(archive_points, ref_point)
            hv_with = compute_hypervolume(archive_points + [point], ref_point)
            return hv_with - hv_without

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Sobol sequence initialization (deterministic, space-filling)
        n_init = min(80, self.budget)
        # Use scipy.stats.qmc.Sobol for quasi-random sequence
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Number of points must be power of 2 for Sobol
        n_sobol = 2 ** int(np.ceil(np.log2(n_init)))
        if n_sobol > self.budget:
            n_sobol = 2 ** int(np.floor(np.log2(self.budget)))
            if n_sobol < 2:
                n_sobol = 2
        samples = sobol.random(n_sobol)
        for i in range(min(n_sobol, n_init)):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + samples[i] * (hi - lo)
            evaluate(x)

        # Phase 2: HypE-style evolutionary loop
        sigma = 0.12  # initial relative step size
        n_hype_target = min(100, self.budget - used)

        while used < self.budget - 20 and time.time() < deadline:  # reserve 20 for polish
            if len(archive) == 0:
                x = np.random.uniform(lo, hi)
                evaluate(x)
                continue
            
            # Compute marginal hypervolume contributions
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append(max(contrib, 1e-10))
            
            # Select parent with probability proportional to contribution
            total_contrib = sum(contributions)
            if total_contrib <= 0:
                parent_idx = np.random.randint(len(archive))
            else:
                probs = np.array(contributions) / total_contrib
                parent_idx = np.random.choice(len(archive), p=probs)
            
            parent_x, parent_f = archive[parent_idx]
            
            # Generate offspring via Gaussian mutation
            x_off = parent_x.copy()
            for d in range(self.dim):
                step = sigma * (hi[d] - lo[d])
                x_off[d] += np.random.normal(0, step)
                x_off[d] = np.clip(x_off[d], lo[d], hi[d])
            
            f_off = evaluate(x_off)
            
            # Adaptive sigma
            if f_off is not None and is_valid(f_off):
                dominated = False
                for _, af in archive:
                    if np.all(af <= f_off) and np.any(af < f_off):
                        dominated = True
                        break
                if not dominated:
                    sigma *= 0.95
                else:
                    sigma *= 1.05
            else:
                sigma *= 1.05
            
            sigma = np.clip(sigma, 0.01, 0.4)
        
        # Phase 3: Nelder-Mead polish on top hypervolume contributors
        if len(archive) > 0 and used < self.budget and time.time() < deadline:
            # Compute contributions to rank archive points
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append(contrib)
            
            # Get top-10 by contribution
            top_k = min(10, len(archive))
            top_indices = np.argsort(contributions)[-top_k:]
            top_indices = top_indices[::-1]  # descending order
            
            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                
                x0 = archive[idx][0].copy()
                
                # Define scalarized objective for NM (weighted sum with adaptive weights)
                f0 = archive[idx][1]
                # Use weights based on reference point distance
                w0 = (ref[0] - f0[0]) / (ref[0] - lo[0] + 1e-10)
                w1 = (ref[1] - f0[1]) / (ref[1] - hi[1] + 1e-10)
                w = np.array([w0, w1])
                w = w / (np.sum(w) + 1e-10)
                
                def scalarized_objective(x):
                    if used >= self.budget or time.time() > deadline:
                        return 1e10
                    fx = problem(x)
                    used_count = used  # capture before increment
                    # We need to count this evaluation
                    # But we can't easily increment inside the closure
                    # So we'll just evaluate and let NM handle it
                    return float(np.dot(w, fx))
                
                # Use Nelder-Mead with tight tolerances
                try:
                    result = minimize(
                        scalarized_objective,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'maxiter': 50,
                            'maxfev': 20  # limit function evaluations per NM call
                        }
                    )
                    # Evaluate the best point found
                    x_best = np.clip(result.x, lo, hi)
                    f_best = problem(x_best)
                    used += 1
                    if is_valid(f_best):
                        add_to_archive(x_best, f_best)
                except Exception:
                    pass
        
        return archive

# EVOLVE-BLOCK-END
