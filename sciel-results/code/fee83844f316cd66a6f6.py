# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Direct optimization of the 64 tube base positions (128 DOF) using
    simulated annealing with a fast vectorized union-volume estimator.
    This breaks the separable assumption and can find non-product configurations.
    """
    N = 8
    start_time = time.time()
    deadline = 630.0
    safety_margin = 15.0

    # Precompute tube indices
    k_idx = np.arange(N * N)
    i_idx = (k_idx // N).astype(np.float64)  # 0..7
    j_idx = (k_idx % N).astype(np.float64)   # 0..7
    width = 1.0 / N

    def compute_volume(pos):
        """
        pos: shape (64, 2), base positions (x_k, y_k)
        Returns the Monte Carlo union volume estimate.
        """
        x = pos[:, 0]  # (64,)
        y = pos[:, 1]  # (64,)

        # Compute box bounds
        # x_k + c*i/8 + width, max over c in [0,1] -> x_k + i/8 + width
        x_max = x + i_idx / N + width
        x_min = x
        y_max = y + j_idx / N + width
        y_min = y

        X0 = x_min.min()
        X1 = x_max.max()
        Y0 = y_min.min()
        Y1 = y_max.max()

        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return np.inf

        # Use a fixed small sample for speed during optimization
        rng = np.random.default_rng(42)
        n_samples = 20000
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)

        # Check if each point is inside at least one tube
        # Vectorize: for each sample point, check all 64 tubes
        # a, b, c have shape (n_samples,)
        # x_k + c*i/8 <= a <= x_k + c*i/8 + width
        # y_k + c*j/8 <= b <= y_k + c*j/8 + width

        # For each tube k, the x-interval at height c is [x_k + c*i_k/N, x_k + c*i_k/N + width]
        # We can vectorize over tubes and samples
        # Shape: (64, n_samples)
        x_starts = x[:, None] + c[None, :] * i_idx[:, None] / N  # (64, n_samples)
        x_ends = x_starts + width
        y_starts = y[:, None] + c[None, :] * j_idx[:, None] / N  # (64, n_samples)
        y_ends = y_starts + width

        # Check inside
        inside_x = (a[None, :] >= x_starts) & (a[None, :] <= x_ends)  # (64, n_samples)
        inside_y = (b[None, :] >= y_starts) & (b[None, :] <= y_ends)  # (64, n_samples)

        inside = inside_x & inside_y  # (64, n_samples)
        any_inside = inside.any(axis=0)  # (n_samples,)

        fraction = any_inside.mean()
        V = fraction * box_area
        if V <= 0:
            return np.inf
        return V

    def final_volume(pos):
        """
        Compute the exact Monte Carlo volume as specified in the task.
        """
        x = pos[:, 0]
        y = pos[:, 1]

        x_max = x + i_idx / N + width
        x_min = x
        y_max = y + j_idx / N + width
        y_min = y

        X0 = x_min.min()
        X1 = x_max.max()
        Y0 = y_min.min()
        Y1 = y_max.max()

        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return np.inf

        rng = np.random.default_rng(12345)
        n = 500000
        a = rng.uniform(X0, X1, n)
        b = rng.uniform(Y0, Y1, n)
        c = rng.uniform(0, 1, n)

        x_starts = x[:, None] + c[None, :] * i_idx[:, None] / N
        x_ends = x_starts + width
        y_starts = y[:, None] + c[None, :] * j_idx[:, None] / N
        y_ends = y_starts + width

        inside_x = (a[None, :] >= x_starts) & (a[None, :] <= x_ends)
        inside_y = (b[None, :] >= y_starts) & (b[None, :] <= y_ends)
        inside = inside_x & inside_y
        any_inside = inside.any(axis=0)
        fraction = any_inside.mean()
        V = fraction * box_area
        if V <= 0:
            return np.inf
        return V

    # Initial positions: Keich-like construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    # Build initial 64x2 positions using separable structure
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    pos0 = np.stack([base_x, base_y], axis=1)

    # Center and scale
    pos0[:, 0] -= pos0[:, 0].mean()
    pos0[:, 1] -= pos0[:, 1].mean()

    best_pos = pos0.copy()
    best_V = compute_volume(best_pos)
    print(f"Initial V: {best_V:.6f}")

    # Simulated Annealing
    rng_sa = np.random.default_rng(123)
    T0 = 0.05
    T_end = 0.001
    n_iterations = 3000
    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break

        if restart == 0:
            current_pos = best_pos.copy()
            current_V = best_V
        else:
            # Perturb best
            current_pos = best_pos + rng_sa.normal(0, 0.1, (N*N, 2))
            current_V = compute_volume(current_pos)

        T = T0
        cooling = (T_end / T0) ** (1.0 / n_iterations)

        for it in range(n_iterations):
            if time.time() > deadline - safety_margin:
                break

            # Propose a move: perturb a random subset of positions
            n_perturb = rng_sa.integers(1, 8)
            idx_perturb = rng_sa.choice(N*N, n_perturb, replace=False)
            proposal = current_pos.copy()
            proposal[idx_perturb] += rng_sa.normal(0, T * 0.5, (n_perturb, 2))

            prop_V = compute_volume(proposal)

            if prop_V < current_V or rng_sa.random() < np.exp(-(prop_V - current_V) / T):
                current_pos = proposal
                current_V = prop_V
                if current_V < best_V:
                    best_V = current_V
                    best_pos = current_pos.copy()
                    print(f"Restart {restart}, Iter {it}: V = {best_V:.6f}")

            T *= cooling

    # Final refinement: local optimization around best
    if time.time() < deadline - safety_margin:
        # Do a few more SA iterations with smaller step
        T0 = 0.01
        T_end = 0.0005
        n_iterations = 2000
        current_pos = best_pos.copy()
        current_V = best_V
        T = T0
        cooling = (T_end / T0) ** (1.0 / n_iterations)

        for it in range(n_iterations):
            if time.time() > deadline - safety_margin:
                break

            n_perturb = rng_sa.integers(1, 5)
            idx_perturb = rng_sa.choice(N*N, n_perturb, replace=False)
            proposal = current_pos.copy()
            proposal[idx_perturb] += rng_sa.normal(0, T * 0.3, (n_perturb, 2))

            prop_V = compute_volume(proposal)

            if prop_V < current_V or rng_sa.random() < np.exp(-(prop_V - current_V) / T):
                current_pos = proposal
                current_V = prop_V
                if current_V < best_V:
                    best_V = current_V
                    best_pos = current_pos.copy()
                    print(f"Refinement Iter {it}: V = {best_V:.6f}")

            T *= cooling

    # Verify final volume with the exact MC estimator
    final_V = final_volume(best_pos)
    if np.isfinite(final_V) and final_V > 0:
        print(f"Final V (exact MC): {final_V:.6f}")
    else:
        # Fallback to initial
        print("Final V invalid, using initial")
        best_pos = pos0

    return best_pos

# EVOLVE-BLOCK-END
