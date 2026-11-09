import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # 5s margin from 330s limit

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        def time_left():
            return self.deadline - (time.time() - self.start_time)

        # --- Phase 1: Coarse Global Search ---
        # For d=5, a moderate number of random samples to identify promising basins.
        # Step ellipsoidal (f7) is separable-ish but rotated, so random search is decent for global.
        scout_budget = 40
        best_x = None
        best_f = np.inf
        scout_points = []  # list of (fx, x)

        for i in range(scout_budget):
            if time_left() < 10:
                break
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            scout_points.append((best_f, best_x))

        # Sort by function value
        scout_points.sort(key=lambda t: t[0])

        # --- Phase 2: Multi-Start Local Refinement ---
        # Select top K distinct starting points
        K = 8
        starts = []
        threshold = 1.5  # Distance threshold for distinctness

        for fx, x in scout_points:
            is_distinct = True
            for sx in starts:
                if np.linalg.norm(x - sx) < threshold:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x)
            if len(starts) >= K:
                break

        # Fill remaining starts with perturbations of best
        while len(starts) < K:
            starts.append(np.clip(best_x + rng.normal(0, 1.0, d), lo, hi))

        remaining_budget = budget - len(scout_points)
        budget_per_start = max(20, remaining_budget // K)

        global_best_x = best_x.copy()
        global_best_f = best_f

        # --- Phase 3: Adaptive Coordinate Descent with Step Halving ---
        # For each start, perform coordinate descent with adaptive step size.
        # This is well-suited for step-ellipsoidal functions where plateaus exist.

        for start_idx, start_x in enumerate(starts):
            if time_left() < 15:
                break

            m = start_x.copy()
            # Get f(m) - check if it's in scout_points
            m_f = None
            for fx, sx in scout_points:
                if np.allclose(sx, m):
                    m_f = fx
                    break
            if m_f is None:
                m_f = func(m)

            if m_f < global_best_f:
                global_best_f = m_f
                global_best_x = m.copy()

            # Adaptive step sizes per coordinate
            steps = np.ones(d) * 0.5
            no_improve = 0
            max_no_improve = 15

            calls_this_start = 0
            while calls_this_start < budget_per_start:
                if time_left() < 5:
                    break

                improved = False
                for i in range(d):
                    if time_left() < 5:
                        break
                    if calls_this_start >= budget_per_start:
                        break

                    # Try positive step
                    x_plus = m.copy()
                    x_plus[i] = np.clip(x_plus[i] + steps[i], lo, hi)
                    f_plus = func(x_plus)
                    calls_this_start += 1

                    if f_plus < m_f:
                        m = x_plus.copy()
                        m_f = f_plus
                        improved = True
                        # Shrink step slightly on success
                        steps[i] *= 0.8
                        if m_f < global_best_f:
                            global_best_f = m_f
                            global_best_x = m.copy()
                    else:
                        # Try negative step
                        x_minus = m.copy()
                        x_minus[i] = np.clip(x_minus[i] - steps[i], lo, hi)
                        f_minus = func(x_minus)
                        calls_this_start += 1

                        if f_minus < m_f:
                            m = x_minus.copy()
                            m_f = f_minus
                            improved = True
                            steps[i] *= 0.8
                            if m_f < global_best_f:
                                global_best_f = m_f
                                global_best_x = m.copy()
                        else:
                            # No improvement in this coordinate, shrink step
                            steps[i] *= 0.5
                            # If step becomes too small, try a larger random jump
                            if steps[i] < 0.01:
                                # Random restart of step size for this coordinate
                                steps[i] = rng.uniform(0.5, 2.0)

                if not improved:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        # Try a larger random perturbation
                        perturbation = rng.normal(0, 1.0, d)
                        x_jump = np.clip(m + perturbation, lo, hi)
                        f_jump = func(x_jump)
                        calls_this_start += 1
                        if f_jump < m_f:
                            m = x_jump.copy()
                            m_f = f_jump
                            no_improve = 0
                            if m_f < global_best_f:
                                global_best_f = m_f
                                global_best_x = m.copy()
                        else:
                            # Reduce step sizes overall
                            steps *= 0.5
                            no_improve = 0
                else:
                    no_improve = 0

        return global_best_x, global_best_f
