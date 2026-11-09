"""Bermudan exercise-policy: model-and-optimize with Gauss-Hermite quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]

    # Gauss-Hermite quadrature with 200 points for high accuracy
    n_gh = 200
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(n_gh)
    z_nodes = np.sqrt(2.0) * gh_nodes
    z_weights = gh_weights / np.sqrt(np.pi)
    z_weights /= z_weights.sum()

    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    exp_factors = np.exp(drift + vol * z_nodes)  # (n_gh,)
    ef = exp_factors
    zw = z_weights

    def run_backward(grid, use_spline=True):
        g = grid
        n_g = len(g)
        V = np.maximum(K - g, 0.0)
        decisions = {}

        for t in range(n_steps, 0, -1):
            # Vectorized: (n_grid, n_gh) matrix of next spots
            ns = g[:, None] * ef[None, :]
            ns_c = np.clip(ns, g[0], g[-1])

            if use_spline:
                # Use 'not-a-knot' boundary condition which is often smoother for option values
                try:
                    cs = CubicSpline(g, V, bc_type='not-a-knot')
                    Vi = cs(ns_c)
                except Exception:
                    # Fallback to natural if not-a-knot fails for some reason
                    cs = CubicSpline(g, V, bc_type='natural')
                    Vi = cs(ns_c)
            else:
                # Linear interpolation fallback
                idx = np.searchsorted(g, ns_c)
                idx = np.clip(idx, 1, len(g) - 1)
                sl = g[idx - 1]
                su = g[idx]
                denom = su - sl
                denom = np.where(denom < 1e-12, 1e-12, denom)
                w = np.clip((ns_c - sl) / denom, 0.0, 1.0)
                Vl = V[idx - 1]
                Vu = V[idx]
                Vi = (1.0 - w) * Vl + w * Vu

            cont = discount * (Vi @ zw)
            ex = np.maximum(K - g, 0.0)
            V_new = np.maximum(ex, cont)
            if t < n_steps:
                decisions[t] = ex >= cont
            V = V_new

        return V, decisions

    # Pass 1: Base grid
    s_min = 0.1
    s_max = 3.0 * K
    n_grid1 = 12000
    grid1 = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_grid1))

    V1, dec1 = run_backward(grid1, use_spline=True)

    grid_final = grid1
    decisions_final = dec1

    # Pass 2: Adaptive refinement near exercise boundaries
    if time.time() < deadline - 50:
        boundary_regions = []
        for t in range(1, n_steps):
            if t in dec1:
                d = dec1[t]
                flips = np.where(d[:-1] != d[1:])[0]
                for f in flips:
                    lo = max(0, f - 200)
                    hi = min(len(grid1) - 1, f + 200)
                    boundary_regions.append((grid1[lo], grid1[hi]))

        if boundary_regions:
            extra_points = []
            for lo, hi in boundary_regions:
                n_extra = 800
                extra_points.append(np.exp(np.linspace(np.log(lo), np.log(hi), n_extra)))
            extra_grid = np.concatenate(extra_points)
            grid2 = np.unique(np.concatenate([grid1, extra_grid]))

            if time.time() < deadline - 40:
                V2, dec2 = run_backward(grid2, use_spline=True)
                grid_final = grid2
                decisions_final = dec2

    # Pass 3: Final polish on a very tight grid around the boundary
    if time.time() < deadline - 30:
        boundary_regions3 = []
        for t in range(1, n_steps):
            if t in decisions_final:
                d = decisions_final[t]
                flips = np.where(d[:-1] != d[1:])[0]
                for f in flips:
                    lo = max(0, f - 100)
                    hi = min(len(grid_final) - 1, f + 100)
                    boundary_regions3.append((grid_final[lo], grid_final[hi]))

        if boundary_regions3:
            extra_points3 = []
            for lo, hi in boundary_regions3:
                n_extra3 = 1200
                extra_points3.append(np.exp(np.linspace(np.log(lo), np.log(hi), n_extra3)))
            extra_grid3 = np.concatenate(extra_points3)
            grid3 = np.unique(np.concatenate([grid_final, extra_grid3]))

            if time.time() < deadline - 25:
                V3, dec3 = run_backward(grid3, use_spline=True)
                grid_final = grid3
                decisions_final = dec3

    # Pass 4: Ultra-fine polish if time permits
    if time.time() < deadline - 15:
        boundary_regions4 = []
        for t in range(1, n_steps):
            if t in decisions_final:
                d = decisions_final[t]
                flips = np.where(d[:-1] != d[1:])[0]
                for f in flips:
                    lo = max(0, f - 50)
                    hi = min(len(grid_final) - 1, f + 50)
                    boundary_regions4.append((grid_final[lo], grid_final[hi]))

        if boundary_regions4:
            extra_points4 = []
            for lo, hi in boundary_regions4:
                n_extra4 = 1500
                extra_points4.append(np.exp(np.linspace(np.log(lo), np.log(hi), n_extra4)))
            extra_grid4 = np.concatenate(extra_points4)
            grid4 = np.unique(np.concatenate([grid_final, extra_grid4]))

            if time.time() < deadline - 10:
                V4, dec4 = run_backward(grid4, use_spline=True)
                grid_final = grid4
                decisions_final = dec4

    g, dec = grid_final, decisions_final

    class ExercisePolicy:
        def __init__(self, grid, decisions, n_steps, K):
            self.grid = grid
            self.decisions = decisions
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            cs = np.asarray(current_state, dtype=np.float64)
            if time not in self.decisions:
                return cs < self.K
            idx = np.searchsorted(self.grid, cs)
            idx = np.clip(idx, 0, len(self.grid) - 1)
            return self.decisions[time][idx]

    return ExercisePolicy(g, dec, n_steps, K)
# EVOLVE-BLOCK-END
