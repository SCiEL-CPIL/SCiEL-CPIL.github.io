"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Halton-space-filling + CMA-ES style optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, idx, base):
        """Compute the Halton sequence value at index idx for a given base."""
        f = 1.0
        r = 0.0
        i = idx
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_point(self, idx, dim):
        """Generate a Halton sequence point in [0, 1]^dim."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        return np.array([self._halton(idx + 1, primes[d]) for d in range(dim)])

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Phase 1: Deterministic Halton space-filling search (first 60 evaluations)
        # This systematically covers the domain to find the basin of attraction
        halton_n = min(60, budget)
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(halton_n):
            if time.time() > deadline:
                break
            h = self._halton_point(i, dim)
            x = -5.0 + 10.0 * h  # Map to [-5, 5]^dim
            fx = func(x)
            candidates.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using a simple (1+1)-ES with adaptive step size
        # Start from the best Halton point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_made = halton_n

        # Also try a few other promising points from Halton search
        candidates.sort(key=lambda t: t[1])
        top_candidates = [c[0].copy() for c in candidates[:min(5, len(candidates))]]

        for start_idx in range(len(top_candidates)):
            if calls_made >= budget or time.time() > deadline:
                break
            x = top_candidates[start_idx].copy()
            fx = func(x) if start_idx > 0 else best_f
            calls_made += 1 if start_idx > 0 else 0
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            no_improve = 0
            max_no_improve = 30

            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.9

                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
