"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    # Strategy: sparse signed sequences with local optimization
    # Use Bernoulli sampling to create sparse structures where
    # autoconvolution peak decays faster than the squared integral

    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Try various sparsity levels and random sign patterns
    sparsity_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for density in sparsity_levels:
        if time.time() > deadline:
            break

        n_restarts = 20
        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Create sparse signed sequence
            mask = rng.random(N) < density
            x = np.zeros(N)
            x[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            
            # Ensure non-zero sum
            s = np.sum(x)
            if abs(s) < 1e-12:
                continue

            val = score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

            # Local search: try flipping signs of non-zero entries
            improved = True
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False

                # Try flipping a random non-zero entry
                non_zero_idx = np.where(x != 0)[0]
                if len(non_zero_idx) == 0:
                    break
                    
                trial_idx = rng.choice(non_zero_idx)
                orig = x[trial_idx]
                x[trial_idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                    improved = True
                else:
                    x[trial_idx] = orig

                # Also try adding a new non-zero entry
                zero_idx = np.where(x == 0)[0]
                if len(zero_idx) > 0:
                    add_idx = rng.choice(zero_idx)
                    x[add_idx] = rng.choice([-1.0, 1.0])
                    new_val = score(x)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_x = x.copy()
                        improved = True
                    else:
                        x[add_idx] = 0

                # Try removing a non-zero entry
                non_zero_idx = np.where(x != 0)[0]
                if len(non_zero_idx) > 1:
                    rem_idx = rng.choice(non_zero_idx)
                    x[rem_idx] = 0
                    new_val = score(x)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_x = x.copy()
                        improved = True
                    else:
                        x[rem_idx] = orig if orig != 0 else 1.0

    # Final polish: try small perturbations on the best solution
    if time.time() < deadline - 5:
        x = best_x.copy()
        for trial in range(300):
            if time.time() > deadline - 2:
                break
            idx = rng.choice(N)
            if x[idx] != 0:
                orig = x[idx]
                x[idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                else:
                    x[idx] = orig
            else:
                x[idx] = rng.choice([-1.0, 1.0])
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                else:
                    x[idx] = 0

    return best_x
# EVOLVE-BLOCK-END
