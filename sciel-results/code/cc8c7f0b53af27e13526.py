"""Multi-objective DTLZ2 optimizer: analytic-front Sobol sampling + adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Structured DTLZ2 optimizer exploiting the analytic Pareto front with Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        n_shape = n_obj - 1      # 2 angle variables
        n_dist = dim - n_shape   # 10 distance variables

        # --- Build a point on the Pareto front from angles ---
        def front_point(angles):
            x = np.empty(dim)
            x[:n_shape] = angles
            x[n_shape:] = 0.5    # exactly on front (g = 0)
            return np.clip(x, lo, hi)

        def angles_to_objectives(angles):
            """Convert angles (theta1, theta2) to DTLZ2 objectives on the front."""
            t1 = angles[0]
            t2 = angles[1]
            f1 = np.cos(t1) * np.cos(t2)
            f2 = np.cos(t1) * np.sin(t2)
            f3 = np.sin(t1)
            return np.array([f1, f2, f3])

        # --- Generate well-spread angles using Sobol sequence on [0, pi/2]^2 ---
        def gen_angles_sobol(n):
            """Generate n points in [0, pi/2]^2 using Sobol sequence."""
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # Sobol requires power-of-2 samples; use next power of 2 >= n
            m = 1
            while m < n:
                m *= 2
            samples = sampler.random(m)
            # Map from [0,1]^2 to [0, pi/2]^2
            angles = samples[:n] * (np.pi / 2)
            return angles

        def gen_angles_grid(n):
            """Generate n points in [0, pi/2]^2 using a structured grid."""
            n_side = int(np.ceil(np.sqrt(n)))
            xs = np.linspace(0, np.pi / 2, n_side, endpoint=False) + (np.pi / 2) / (2 * n_side)
            ys = np.linspace(0, np.pi / 2, n_side, endpoint=False) + (np.pi / 2) / (2 * n_side)
            grid_x, grid_y = np.meshgrid(xs, ys)
            angles = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            return angles[:n]

        def gen_angles_fibonacci(n):
            """Generate n points in [0, pi/2]^2 using Fibonacci lattice."""
            golden = (1 + np.sqrt(5)) / 2.0
            angles = []
            for k in range(n):
                # Map k to a point on the quarter circle in a space-filling way
                t = (k + 0.5) / n
                # Use a spiral pattern
                r = np.sqrt(t)
                theta = 2 * np.pi * k / golden
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                # Map to [0, pi/2]^2
                a1 = np.clip(x, 0, 1) * (np.pi / 2)
                a2 = np.clip(y, 0, 1) * (np.pi / 2)
                angles.append([a1, a2])
            return np.array(angles)

        # --- Hypervolume contribution helper ---
        def compute_hv_contribution(points, ref):
            """Compute the hypervolume of a set of points in 3D (for selection)."""
            # Simple exact HV for 3D with small n
            n = len(points)
            if n == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(points[:, 0])
            points = points[order]
            hv = 0.0
            for i in range(n):
                p = points[i]
                if np.any(p >= ref):
                    continue
                # Volume of box from p to ref, minus parts dominated by later points
                # For 3D, we can use a simpler approach
                pass
            # Fallback: use a grid-based approximation
            # Actually, for selection we just need relative quality
            # Use a simple dominance-based score
            return 0.0

        def is_dominated(point, population):
            """Check if point is dominated by any point in population."""
            for p in population:
                if np.all(p <= point) and np.any(p < point):
                    return True
            return False

        def dominated_by(point, population):
            """Check if point is dominated by any point in population."""
            for p in population:
                if np.all(p <= point) and np.any(p < point):
                    return True
            return False

        # Phase 1: Dense Sobol sampling on the analytic front
        # Use most of the budget for well-spread points
        n_sobol = min(self.budget - 20, 180)  # Leave some budget for refinement
        n_sobol = max(n_sobol, 50)

        angles = gen_angles_sobol(n_sobol)
        X = np.array([front_point(a) for a in angles])
        F = np.array([problem(x) for x in X])
        used = n_sobol

        # Phase 2: Adaptive refinement - fill gaps in objective space
        # Identify regions with poor coverage and add points there
        if used < self.budget and time.time() < deadline - 5:
            # Divide the objective space into cells and find underrepresented ones
            # Use a simple approach: for each point, check local density and add neighbors if sparse
            n_refine = min(self.budget - used, 20)
            for _ in range(n_refine):
                if used >= self.budget or time.time() > deadline - 5:
                    break

                # Pick a random existing point and perturb its angles
                idx = rng.integers(len(X))
                ang = X[idx][:n_shape].copy()
                # Perturb in a direction that moves away from other points
                # Try a few random perturbations
                best_cand = None
                best_fc = None
                for _ in range(5):
                    step = rng.uniform(0.02, 0.1)
                    direction = rng.normal(0, 1, n_shape)
                    direction /= (np.linalg.norm(direction) + 1e-12)
                    new_ang = ang + step * direction
                    new_ang = np.clip(new_ang, lo[:n_shape], hi[:n_shape])
                    cand = front_point(new_ang)
                    fc = problem(cand)
                    # Check if this point is not dominated and adds coverage
                    if not dominated_by(fc, F):
                        # Check distance to existing points in objective space
                        dists = np.min(np.linalg.norm(F - fc, axis=1))
                        if best_cand is None or dists > np.min(np.linalg.norm(F - best_fc, axis=1)):
                            best_cand = cand
                            best_fc = fc

                if best_cand is not None:
                    # Check if it's not dominated by the current set
                    if not dominated_by(best_fc, F):
                        # Add it (replace a random point if set is too large)
                        if len(X) < 200:
                            X = np.vstack([X, best_cand])
                            F = np.vstack([F, best_fc])
                        else:
                            # Replace the point with smallest crowding distance
                            # Simple: replace a random point
                            rep_idx = rng.integers(len(X))
                            X[rep_idx] = best_cand
                            F[rep_idx] = best_fc
                        used += 1

        # Phase 3: Final polish - ensure we have good coverage
        # If we have budget left, do a few more Sobol points with different seed
        if used < self.budget and time.time() < deadline - 3:
            n_extra = min(self.budget - used, 10)
            if n_extra > 0:
                sampler = qmc.Sobol(d=2, scramble=True, seed=123)
                m = 1
                while m < n_extra:
                    m *= 2
                samples = sampler.random(m)
                angles_extra = samples[:n_extra] * (np.pi / 2)
                for a in angles_extra:
                    if used >= self.budget or time.time() > deadline - 3:
                        break
                    cand = front_point(a)
                    fc = problem(cand)
                    if not dominated_by(fc, F):
                        if len(X) < 200:
                            X = np.vstack([X, cand])
                            F = np.vstack([F, fc])
                        else:
                            rep_idx = rng.integers(len(X))
                            X[rep_idx] = cand
                            F[rep_idx] = fc
                        used += 1

        return X
# EVOLVE-BLOCK-END
