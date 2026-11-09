"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction with Hammersley sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # For DTLZ1, the Pareto front is characterized by:
        # f0 = 0.5 * x[0] * (1 + g)
        # f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
        # f2 = 0.5 * x[0] * x[1] * (1 + g)
        # where g = sum_{i=3}^{n} (x_i - 0.5)^2 - 10*cos(4*pi*x_i*0.5)
        # g_min = 0 achieved at x_i = 0.5 for all i >= 2
        #
        # Strategy:
        # 1. Set x[2:] to 0.5 (known optimal for g=0)
        # 2. Sample points on the Pareto front using Hammersley low-discrepancy sequence
        # 3. Use remaining budget for local refinement

        rng = np.random.default_rng(42)
        best_x2 = np.full(dim - 2, 0.5)
        best_g_val = np.inf

        # Phase 1: Verify optimal x[2:] configuration
        # Probe candidates for x[2:]
        candidates = [
            np.zeros(dim - 2),
            np.ones(dim - 2),
            np.full(dim - 2, 0.5),
        ]
        for _ in range(5):
            candidates.append(rng.random(dim - 2))

        for x2 in candidates:
            if time.time() > deadline:
                break
            x_test = np.concatenate([[0.5, 0.5], x2])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_val = np.sum(f)
            if g_val < best_g_val:
                best_g_val = g_val
                best_x2 = x2.copy()

        # Local search to refine x[2:]
        n_local = min(10, self.budget - 8)
        for _ in range(n_local):
            if time.time() > deadline:
                break
            x2_pert = best_x2 + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(dim - 2)
            x2_pert = np.clip(x2_pert, lo[2:], hi[2:])
            x_test = np.concatenate([[0.5, 0.5], x2_pert])
            x_test = np.clip(x_test, lo, hi)
            f_test = problem(x_test)
            g_test = np.sum(f_test)
            if g_test < best_g_val:
                best_g_val = g_test
                best_x2 = x2_pert.copy()

        evaluated = 8 + n_local

        # Phase 2: Sample points on the Pareto front using Hammersley sequence
        # The Pareto front for DTLZ1 (m=3) is a 2D simplex in objective space:
        # f0 + f1 + f2 = 0.5 * x0 * (1 + g_min)
        # With g_min ≈ 0 and x0 = 1: f0 + f1 + f2 = 0.5
        # We parameterize by (x0, x1) where x0 ∈ [0, 1], x1 ∈ [0, 1-x0]

        recorded_F = []
        remaining_budget = self.budget - evaluated

        # Generate Hammersley points in 2D for the simplex
        # Hammersley sequence uses van der Corput sequence in each dimension
        def van_der_corput(n, base=2):
            """Generate n points in [0,1) using van der Corput sequence."""
            result = []
            for i in range(n):
                f = 1.0 / base
                result.append(0.0)
            result = np.zeros(n)
            for i in range(n):
                f = 1.0 / base
                val = 0.0
                idx = i
                while idx > 0:
                    val += f * (idx % base)
                    idx //= base
                    f /= base
                result[i] = val
            return result

        # We want to fill the 2D simplex: x0 >= 0, x1 >= 0, x0 + x1 <= 1
        # Use Hammersley points in [0,1]^2 and map to simplex
        n_hammersley = remaining_budget

        # Generate Hammersley points
        vdc1 = van_der_corput(n_hammersley, base=2)
        vdc2 = van_der_corput(n_hammersley, base=3)  # Use different base for second dimension

        # Map Hammersley points to the simplex
        # For a point (u1, u2) in [0,1]^2, map to simplex:
        # x0 = u1 * (1 - u2), x1 = u2 * (1 - u1)
        # This ensures x0 >= 0, x1 >= 0, x0 + x1 <= 1
        x0_vals = vdc1 * (1 - vdc2)
        x1_vals = vdc2 * (1 - vdc1)

        # Add boundary points for better coverage
        boundary_points = []
        n_boundary = min(20, n_hammersley // 5)
        for i in range(n_boundary):
            # Points along the edges of the simplex
            t = i / max(1, n_boundary - 1)
            boundary_points.append((t, 0.0))       # Edge x1=0
            boundary_points.append((0.0, t))       # Edge x0=0
            boundary_points.append((t * (1-t), t * (1-t)))  # Diagonal-ish

        # Evaluate Hammersley points
        for i in range(n_hammersley):
            if time.time() > deadline:
                break
            if evaluated >= self.budget:
                break
            x_sample = np.concatenate([[x0_vals[i], x1_vals[i]], best_x2])
            x_sample = np.clip(x_sample, lo, hi)
            f_sample = problem(x_sample)
            recorded_F.append(f_sample)
            evaluated += 1

        # Add boundary points for better edge coverage
        for bp in boundary_points:
            if time.time() > deadline:
                break
            if evaluated >= self.budget:
                break
            x_cp = np.concatenate([list(bp), best_x2])
            x_cp = np.clip(x_cp, lo, hi)
            f_cp = problem(x_cp)
            # Check if this point is already recorded (avoid duplicates)
            is_dup = False
            for existing in recorded_F:
                if np.allclose(existing, f_cp, atol=1e-10):
                    is_dup = True
                    break
            if not is_dup:
                recorded_F.append(f_cp)
                evaluated += 1

        # If we still have budget, add more Hammersley points with different base
        if evaluated < self.budget and time.time() < deadline:
            n_extra = self.budget - evaluated
            # Use a different Hammersley sequence with base 4 and 5
            vdc1_extra = van_der_corput(n_extra, base=4)
            vdc2_extra = van_der_corput(n_extra, base=5)
            x0_extra = vdc1_extra * (1 - vdc2_extra)
            x1_extra = vdc2_extra * (1 - vdc1_extra)

            for i in range(n_extra):
                if time.time() > deadline:
                    break
                if evaluated >= self.budget:
                    break
                x_sample = np.concatenate([[x0_extra[i], x1_extra[i]], best_x2])
                x_sample = np.clip(x_sample, lo, hi)
                f_sample = problem(x_sample)
                recorded_F.append(f_sample)
                evaluated += 1

        return recorded_F
# EVOLVE-BLOCK-END
