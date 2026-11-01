"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centers):
    """
    Compute the packing radius for given centers.
    Returns the minimum of (min_pairwise_dist / 2, min_wall_dist).
    """
    min_wall = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    return min(min_pair / 2.0, min_wall)


def _sa_run(centers, rng, start_time, max_time, max_iters=500000):
    """
    Run simulated annealing on the given centers.
    Returns (best_centers, best_score).
    """
    centers = centers.copy()
    current_score = _compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score

    T_start = 0.05
    T_end = 1e-12
    alpha = (T_end / T_start) ** (1.0 / max_iters)

    base_step = 0.05
    step_size = base_step
    no_improve = 0
    max_no_improve = 3000

    for it in range(max_iters):
        if it % 2000 == 0:
            if time.time() - start_time > max_time:
                break

        T = T_start * (alpha ** it)

        i = rng.integers(N)
        old_pos = centers[i].copy()

        # Perturb
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)

        new_score = _compute_score(centers)

        if new_score >= current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            if delta / T > -100:
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    no_improve = 0
                else:
                    centers[i] = old_pos
                    no_improve += 1
            else:
                centers[i] = old_pos
                no_improve += 1

        # Adaptive step
        if no_improve > max_no_improve:
            step_size *= 0.5
            no_improve = 0
        elif it % 1000 == 0 and it > 0:
            step_size *= 0.995

        # Periodic reheating
        if it % 30000 == 0 and it > 0:
            step_size = base_step * (0.3 + 0.7 * rng.random())
            no_improve = 0

    return best_centers, best_score


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(12345)
    start_time = time.time()
    max_time = 110.0

    best_centers = None
    best_score = -np.inf

    # Multiple diverse initializations
    inits = []

    # 1. 7x8 grid
    x_t = np.linspace(-0.45, 0.45, 8)
    y_t = np.linspace(-0.45, 0.45, 7)
    c = np.zeros((N, 2))
    idx = 0
    for r in range(7):
        for col in range(8):
            if idx < N:
                c[idx] = [x_t[col], y_t[r]]
                idx += 1
    c += rng.normal(0, 0.005, (N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    inits.append(c)

    # 2. 5x10 grid
    x_t = np.linspace(-0.48, 0.48, 10)
    y_t = np.linspace(-0.48, 0.48, 5)
    c = np.zeros((N, 2))
    idx = 0
    for r in range(5):
        for col in range(10):
            if idx < N:
                c[idx] = [x_t[col], y_t[r]]
                idx += 1
    c += rng.normal(0, 0.005, (N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    inits.append(c)

    # 3. 10x5 grid
    x_t = np.linspace(-0.48, 0.48, 5)
    y_t = np.linspace(-0.48, 0.48, 10)
    c = np.zeros((N, 2))
    idx = 0
    for r in range(10):
        for col in range(5):
            if idx < N:
                c[idx] = [x_t[col], y_t[r]]
                idx += 1
    c += rng.normal(0, 0.005, (N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    inits.append(c)

    # 4. Hexagonal-ish
    c = np.zeros((N, 2))
    idx = 0
    for r in range(8):
        x_off = 0.0 if r % 2 == 0 else 0.055
        for col in range(9):
            if idx < N:
                x = -0.44 + x_off + col * 0.11
                y = -0.44 + r * 0.125
                if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                    c[idx] = [x, y]
                    idx += 1
        if idx >= N:
            break
    c += rng.normal(0, 0.005, (N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    inits.append(c)

    # 5. Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))

    # 6. 6x9 grid
    x_t = np.linspace(-0.46, 0.46, 9)
    y_t = np.linspace(-0.46, 0.46, 6)
    c = np.zeros((N, 2))
    idx = 0
    for r in range(6):
        for col in range(9):
            if idx < N:
                c[idx] = [x_t[col], y_t[r]]
                idx += 1
    c += rng.normal(0, 0.005, (N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    inits.append(c)

    # Run SA from each initialization
    for init_c in inits:
        if time.time() - start_time > max_time:
            break
        bc, bs = _sa_run(init_c, rng, start_time, max_time, max_iters=300000)
        if bs > best_score:
            best_score = bs
            best_centers = bc

    return best_centers
# EVOLVE-BLOCK-END
