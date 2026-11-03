"""Optimize packing of 100 equal spheres in a unit cube via simulated annealing
with FCC-inspired initialization and vectorized scoring."""
import time
import numpy as np

N = 100
DEADLINE_FRACTION = 0.88  # return by 150 * 0.88 = 132 s


def _score(centers):
    """Compute the packing radius for given centers."""
    c = centers
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0

    # Wall distances: min distance from each center to any of the 6 faces
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum.reduce([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
        0.5 - c[:, 2], c[:, 2] + 0.5
    ])
    min_wall = np.min(wall_dists)

    # Pair distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)

    r = min(min_pair / 2.0, min_wall)
    return r


def _init_fcc_hybrid(rng):
    """
    Hybrid initialization:
    - Place spheres in a dense FCC-like arrangement.
    - FCC in a cube: layers of close-packed hexagonal arrangements.
    - For 100 spheres, use a 4x5x5 grid with FCC offsets.
    
    Alternative: Use a body-centered cubic (BCC) or simple approach.
    Let's use a 4x4x4 simple cubic = 64 spheres, then add 36 in interstitial sites.
    
    Actually, let's use a more systematic approach:
    - Start with a 5x4x5 = 100 grid (or similar) with slight FCC offsets.
    
    Better: Use the known high-density structure.
    For a cube, the densest packing of equal spheres is FCC or HCP.
    FCC unit cell has 4 spheres. In a cube of side 1, if we tile with FCC:
    - FCC lattice constant a: 4 spheres per cell, radius r = a*sqrt(2)/4 = a/(2*sqrt(2))
    - If cube side = 1, and we fit n cells per side, a = 1/n
    - r = 1/(n * 2 * sqrt(2))
    - Number of spheres ~ 4 * n^3
    - For 100 spheres: n^3 ~ 25, n ~ 2.92 -> use n=3: 4*27=108 spheres (too many)
    - n=2: 4*8=32 (too few)
    
    Let's just use a perturbed grid with FCC-like offsets.
    
    Simple approach: 4x5x5 grid = 100 points, with small random perturbations.
    Or: 5x4x5 grid.
    """
    # Use a 5x4x5 grid = 100 points
    # This gives a reasonably dense starting point
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add small random perturbations to break symmetry
    grid = grid + rng.normal(0, 0.02, grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    
    return grid[:N].copy()


def _init_fcc_layers(rng):
    """
    FCC-inspired initialization using layers.
    FCC structure: ABAB... stacking of close-packed layers.
    Each layer is a hexagonal close-packed arrangement.
    
    For a cube, we can have layers perpendicular to one axis.
    Let's use 5 layers along z, each with 20 spheres.
    """
    centers = []
    
    # 5 layers along z
    n_layers = 5
    n_per_layer = 20
    
    # z positions for layers
    z_positions = np.linspace(-0.4, 0.4, n_layers)
    
    for i, z in enumerate(z_positions):
        # Each layer: arrange 20 spheres in a roughly hexagonal pattern
        # Use a 4x5 grid for the layer, with offsets for FCC
        if i % 2 == 0:
            # Even layer: grid positions
            xs = np.linspace(-0.4, 0.4, 4)
            ys = np.linspace(-0.4, 0.4, 5)
            for x in xs:
                for y in ys:
                    centers.append([x, y, z])
        else:
            # Odd layer: offset positions (FCC shift)
            xs = np.linspace(-0.3, 0.3, 4)
            ys = np.linspace(-0.3, 0.3, 5)
            for x in xs:
                for y in ys:
                    centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    centers = centers + rng.normal(0, 0.015, centers.shape)
    centers = np.clip(centers, -0.45, 0.45)
    
    return centers


def _delta_score(c, idx, new_pos):
    """
    Compute the change in score when moving sphere idx to new_pos.
    This is an approximation for speed.
    """
    old_pos = c[idx]
    
    # Wall constraint for the moved sphere
    old_wall = np.min(np.array([
        0.5 - old_pos[0], old_pos[0] + 0.5,
        0.5 - old_pos[1], old_pos[1] + 0.5,
        0.5 - old_pos[2], old_pos[2] + 0.5
    ]))
    new_wall = np.min(np.array([
        0.5 - new_pos[0], new_pos[0] + 0.5,
        0.5 - new_pos[1], new_pos[1] + 0.5,
        0.5 - new_pos[2], new_pos[2] + 0.5
    ]))
    
    # Check if new position is valid
    if new_wall < 0:
        return -np.inf
    
    # Compute new min pair distance involving sphere idx
    # Only need to check pairs involving idx
    diffs = c - new_pos
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    dists[idx] = np.inf
    new_min_pair_with_idx = np.min(dists)
    
    # The score is min(min_pair/2, min_wall)
    # We need to check if this move improves the bottleneck
    # This is approximate - we'll do full recompute for acceptance
    return new_min(new_wall, new_min_pair_with_idx / 2.0)


def new_min(a, b):
    return a if a < b else b


def construct():
    t0 = time.time()
    deadline = t0 + 150 * DEADLINE_FRACTION

    rng = np.random.default_rng(42)

    # Initialize with FCC-inspired hybrid
    best_x = _init_fcc_layers(rng)
    best_score = _score(best_x)

    # Also try simple cubic grid as alternative init
    alt_x = _init_fcc_hybrid(rng)
    alt_score = _score(alt_x)
    if alt_score > best_score:
        best_x = alt_x
        best_score = alt_score

    # Simulated annealing
    T_start = 0.05
    T_end = 0.001
    n_iterations = 500000

    current_x = best_x.copy()
    current_score = best_score

    # Precompute indices for pair distances
    # For efficiency, we'll use vectorized operations

    for it in range(n_iterations):
        if time.time() > deadline:
            break

        # Annealing schedule
        frac = it / n_iterations
        T = T_start * (T_end / T_start) ** frac

        # Choose a random sphere to move
        idx = rng.integers(0, N)

        # Generate a random new position near current
        step_size = 0.05 * (T / T_start) + 0.005
        new_pos = current_x[idx] + rng.normal(0, step_size, 3)
        new_pos = np.clip(new_pos, -0.49, 0.49)

        # Compute new score (full recompute for correctness)
        trial_x = current_x.copy()
        trial_x[idx] = new_pos
        trial_score = _score(trial_x)

        # Accept or reject
        if trial_score >= current_score:
            current_x = trial_x
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_x = current_x.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = trial_score - current_score
            if delta > 0:
                current_x = trial_x
                current_score = trial_score
            else:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_x = trial_x
                    current_score = trial_score

    # Final polishing: try to improve by small local moves
    # Use a few iterations of local optimization
    for polish_iter in range(50):
        if time.time() > deadline:
            break

        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            # Try small perturbations
            for _ in range(10):
                perturbation = rng.normal(0, 0.01, 3)
                new_pos = best_x[idx] + perturbation
                new_pos = np.clip(new_pos, -0.49, 0.49)
                trial_x = best_x.copy()
                trial_x[idx] = new_pos
                trial_score = _score(trial_x)
                if trial_score > best_score:
                    best_x = trial_x
                    best_score = trial_score
                    improved = True

        if not improved:
            break

    return best_x
