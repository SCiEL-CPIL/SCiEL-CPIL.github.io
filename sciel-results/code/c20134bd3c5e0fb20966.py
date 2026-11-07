"""Black-box optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse grid + (1+1)-ES with adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        calls = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Lattice-based grid search to exploit Rastrigin periodicity
        best_x = None
        best_f = float('inf')

        # Rastrigin has minima at integer coordinates (after shift/rotation)
        # Evaluate a structured grid: integers and half-integers in [-4, 4]
        # This gives us a good chance of finding a point near a local minimum

        # Generate lattice points: integers and half-integers
        lattice_values = []
        for i in range(-4, 5):
            lattice_values.append(float(i))
            lattice_values.append(float(i) + 0.5)

        # Evaluate a subset of the full 5D grid (too large to do all 10^5)
        # Use a smart selection: center + axis-aligned + random subset
        candidates = []

        # Center
        candidates.append(np.zeros(self.dim))

        # Pure integer points near origin (most likely to be near a minimum)
        for d in range(self.dim):
            for val in [-2, -1, 0, 1, 2]:
                pt = np.zeros(self.dim)
                pt[d] = val
                candidates.append(pt)

        # Half-integer points along axes
        for d in range(self.dim):
            for val in [-1.5, -0.5, 0.5, 1.5]:
                pt = np.zeros(self.dim)
                pt[d] = val
                candidates.append(pt)

        # Random lattice points (sample from the lattice)
        for _ in range(30):
            pt = np.array([rng.choice(lattice_values) for _ in range(self.dim)])
            candidates.append(pt)

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(np.round(c, 10))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        for c in unique_candidates:
            if not check_time():
                break
            if calls >= self.budget:
                break
            fc = evaluate(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # Phase 2: (1+1)-ES with adaptive step size, tuned for Rastrigin
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Start smaller since we're likely near a local min
        success_count = 0
        total_trials = 0
        window = 15  # Shorter window for faster adaptation

        while calls < self.budget and check_time():
            # Adaptive sigma: 1/5 rule
            if total_trials >= window:
                success_rate = success_count / total_trials
                if success_rate > 0.25:
                    sigma *= 1.2
                elif success_rate < 0.15:
                    sigma *= 0.8
                success_count = 0
                total_trials = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.005, 3.0)

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5.0, 5.0)
            fy = evaluate(y)
            total_trials += 1

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                best_x = x.copy()
                best_f = fx
            else:
                # Periodic restarts to escape local minima
                if total_trials % 30 == 0:
                    # Try a lattice point nearby
                    jump_base = np.round(x).astype(int)
                    for offset in [0.5, -0.5, 1.0, -1.0]:
                        jump = jump_base.copy()
                        jump[rng.integers(self.dim)] += offset
                        jump = np.clip(jump, -5.0, 5.0)
                        fj = evaluate(jump)
                        if fj < fx:
                            x = jump
                            fx = fj
                            best_x = x.copy()
                            best_f = fx
                            sigma = 0.5
                            break

        # Phase 3: Fine polishing with decreasing step sizes
        if calls < self.budget and check_time():
            sigma_fine = 0.05
            for _ in range(50):
                if calls >= self.budget or not check_time():
                    break
                y = x + sigma_fine * rng.standard_normal(self.dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    best_x = x.copy()
                    best_f = fx
                sigma_fine *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
