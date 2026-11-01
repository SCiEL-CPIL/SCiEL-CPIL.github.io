"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _compute_score(pts):
    """Compute the packing score: min(pairwise_dist/2, wall_dist)."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall = 0.5 - np.abs(pts)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)


def _lj_energy(pts, eps=1.0, sigma=1.0):
    """Lennard-Jones potential energy (vectorized)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    energy = eps * (4.0 * inv_r6 ** 2 - inv_r6)
    return np.sum(energy)


def _lj_forces(pts, eps=1.0, sigma=1.0):
    """LJ forces (vectorized)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    # Force magnitude: 24*eps*(2*inv_r6^2 - inv_r6) / r2
    force_mag = 24.0 * eps * (2.0 * inv_r6 ** 2 - inv_r6) * inv_r2
    np.fill_diagonal(force_mag, 0.0)
    forces = force_mag[:, :, np.newaxis] * diff
    return -np.sum(forces, axis=1)


def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _hex_init():
    """Hexagonal packing initialization."""
    # Hexagonal: rows offset
    n_cols = 8
    n_rows = 7  # 8*7 = 56, take 50
    dx = 1.0 / n_cols
    dy = dx * np.sqrt(3) / 2
    pts = []
    for row in range(n_rows):
        y = -0.5 + dy / 2 + row * dy
        x_offset = 0 if row % 2 == 0 else dx / 2
        for col in range(n_cols):
            x = -0.5 + dx / 2 + col * dx + x_offset
            if x < 0.5 and y < 0.5:
                pts.append([x, y])
    pts = np.array(pts[:N])
    return pts


def _random_init(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _halton_init():
    """Halton sequence for quasi-random points."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        pts.append([x - 0.5, y - 0.5])
    return np.array(pts)


def _optimize(pts, time_budget):
    """Run LJ minimization with annealing."""
    t0 = time.time()
    eps = 1.0
    sigma = 1.0
    n_iter = 0
    max_iter = 50000
    learning_rate = 0.01
    min_lr = 1e-6
    decay = 0.9995
    prev_energy = _lj_energy(pts, eps, sigma)
    
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        forces = _lj_forces(pts, eps, sigma)
        # Add wall repulsion
        wall_force = np.zeros_like(pts)
        for i in range(N):
            for k in range(2):
                d = 0.5 - abs(pts[i, k])
                if d < 0.1:
                    wall_force[i, k] = -np.sign(pts[i, k]) * (1.0 - d / 0.1) ** 2 * 10.0
        forces += wall_force
        
        pts = pts + learning_rate * forces
        # Clamp to square
        pts = np.clip(pts, -0.4999, 0.4999)
        
        energy = _lj_energy(pts, eps, sigma)
        # Simple acceptance (always accept for simplicity, use annealing via LR)
        learning_rate = max(min_lr, learning_rate * decay)
        n_iter += 1
    
    return pts


def construct():
    global _START_TIME
    elapsed = time.time() - _START_TIME
    remaining = _TIME_LIMIT - elapsed
    
    if remaining < 5.0:
        # Not enough time, return grid
        return _grid_init()
    
    # Generate diverse initial configurations
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(42),
        _random_init(123),
        _halton_init(),
        _random_init(777),
        _random_init(999),
    ]
    
    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(0)
    order = list(range(len(inits)))
    rng.shuffle(order)
    
    best_pts = None
    best_score = -np.inf
    
    n_configs = len(inits)
    time_per_config = remaining * 0.8 / n_configs  # leave 20% margin
    
    for idx in order:
        if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
            break
        pts = inits[idx].copy()
        t_cfg_start = time.time()
        cfg_budget = time_per_config - (time.time() - t_cfg_start)
        if cfg_budget < 0.5:
            cfg_budget = 0.5
        pts = _optimize(pts, cfg_budget)
        
        # Clamp
        pts = np.clip(pts, -0.4999, 0.4999)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _grid_init()
    
    return best_pts
# EVOLVE-BLOCK-END
