"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic multi-scale CMA-ES variant with fixed geometric decay and single restart."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Precompute a rich set of deterministic directions
        # Axes
        dirs = []
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = 1.0
            dirs.append(d)
        # Pairwise diagonals
        for i in range(dim):
            for j in range(i + 1, dim):
                d = np.zeros(dim)
                d[i] = 1.0
                d[j] = 1.0
                dirs.append(d)
        # Triple diagonals
        for i in range(dim):
            for j in range(i + 1, dim):
                for k in range(j + 1, dim):
                    d = np.zeros(dim)
                    d[i] = 1.0
                    d[j] = 1.0
                    d[k] = 1.0
                    dirs.append(d)
        # All-ones
        dirs.append(np.ones(dim))
        # Hadamard-like sign patterns for better coverage
        # Generate 2^(dim-1) sign vectors (Hadamard rows) for dim=5 -> 16 vectors
        # Use a fixed set of 16 orthogonal-ish directions
        sign_patterns = []
        # First 2^(dim-1) = 16 sign patterns via Gray code or simple enumeration
        for mask in range(1 << (dim - 1)):
            s = np.zeros(dim)
            s[0] = 1.0
            for i in range(1, dim):
                # XOR of bits determines sign
                bit = (mask >> (i - 1)) & 1
                s[i] = 1.0 if bit == 0 else -1.0
            # Normalize
            s = s / np.linalg.norm(s)
            sign_patterns.append(s)
        dirs.extend(sign_patterns)
        
        # Normalize all directions
        dirs = np.array([d / np.linalg.norm(d) for d in dirs])
        n_dirs = len(dirs)

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Fixed geometric sequence of sigma values from 4.0 down to 1e-4
        # Number of scales: log(4.0/1e-4) / log(2) ≈ 15.3, use ~15 scales
        sigma_start = 4.0
        sigma_end = 1e-4
        n_scales = 15
        sigmas = np.geomspace(sigma_start, sigma_end, n_scales)

        # Track best improvement per scale to detect plateau
        best_fx_at_scale_start = best_fx
        restart_done = False

        for sigma in sigmas:
            if calls_made >= budget:
                break

            # At each scale, try all directions in a deterministic order
            # Use a rotating offset to avoid always trying the same direction first
            offset = int(calls_made % n_dirs)
            
            improvements_this_scale = 0
            fx_before_scale = fx

            for idx in range(n_dirs):
                if calls_made >= budget:
                    break
                d = dirs[(offset + idx) % n_dirs]

                # Try positive step
                y = np.clip(x + sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    improvements_this_scale += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

                # Try negative step
                if calls_made >= budget:
                    break
                y = np.clip(x - sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    improvements_this_scale += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

            # After this scale, check if we made any progress
            if improvements_this_scale == 0 and not restart_done:
                # Plateau detected - do a single restart
                restart_done = True
                # Choose restart point deterministically based on current best
                # Try a point offset from best in a deterministic direction
                rng = np.random.default_rng(42)
                # Generate several candidate restart points and pick the best
                candidates = []
                for trial in range(5):
                    cand = best_x + rng.normal(0, 1.5, size=dim)
                    cand = np.clip(cand, -4.5, 4.5)
                    candidates.append(cand)
                
                best_cand = None
                best_cand_fx = np.inf
                for cand in candidates:
                    if calls_made >= budget:
                        break
                    fc = func(cand)
                    calls_made += 1
                    if fc < best_cand_fx:
                        best_cand_fx = fc
                        best_cand = cand

                if best_cand is not None and best_cand_fx < best_fx:
                    x, fx = best_cand.copy(), best_cand_fx
                    best_x, best_fx = x.copy(), fx
                else:
                    # If no better candidate, still restart from a new location
                    # to escape potential local minima
                    x = np.clip(best_x + 2.0 * dirs[0], -4.5, 4.5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx

                # Reset to a mid-range sigma for the restart
                # We'll continue with the remaining smaller sigmas

        # Final polishing phase with remaining budget
        remaining = budget - calls_made
        if remaining > 0:
            # Fine coordinate descent with adaptive step
            fine_sigma = 1e-3
            for _ in range(remaining):
                if calls_made >= budget:
                    break
                improved = False
                # Try each coordinate in both directions
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Positive
                    y = x.copy()
                    y[i] = np.clip(y[i] + fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if improved:
                    continue
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Negative
                    y = x.copy()
                    y[i] = np.clip(y[i] - fine_sigma, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        improved = True
                        break
                if not improved:
                    fine_sigma *= 0.5
                    if fine_sigma < 1e-7:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
