"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(positions):
    """Compute the radius for a given set of positions.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances
    # Use broadcasting for all pairs
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _random_grid_init(n, m):
    """Generate a random grid-like initialization."""
    # Use a stratified approach: divide cube into m^3 cells, place one sphere per cell
    # For n=100, m=5 gives 125 cells, we pick 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if n <= len(grid):
        # Randomly select n positions from the grid
        idx = np.random.choice(len(grid), size=n, replace=False)
        return grid[idx].copy()
    else:
        # If n > m^3, use smaller grid
        m2 = int(np.ceil(n ** (1.0/3.0)))
        ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
        grid2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        if n <= len(grid2):
            idx = np.random.choice(len(grid2), size=n, replace=False)
            return grid2[idx].copy()
        else:
            # Just random
            return np.random.uniform(-0.5, 0.5, size=(n, 3))


def _metropolis_annealing(positions, n, max_iters=200000, t0=0.05, t1=0.0001, step0=0.1):
    """Metropolis-Hastings annealing to optimize sphere packing."""
    current = positions.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        # Linear cooling
        t = t0 * (t1 / t0) ** (it / max_iters)
        # Step size scales with temperature
        step = step0 * (t / t0) ** 0.5
        
        # Random perturbation: move 1-3 random spheres
        n_move = rng.integers(1, 4)
        idx = rng.choice(n, size=n_move, replace=False)
        
        proposed = current.copy()
        delta = rng.normal(0, step, size=(n_move, 3))
        proposed[idx] += delta
        
        # Clamp to valid region
        proposed = np.clip(proposed, -0.5, 0.5)
        
        proposed_score = _score(proposed)
        
        # Accept if better or with probability exp((new-old)/T)
        if proposed_score >= current_score:
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta_e = proposed_score - current_score
            if delta_e > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_e / max(t, 1e-10))
            if rng.random() < accept_prob:
                current = proposed
                current_score = proposed_score
    
    return best, best_score


def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    
    # Start with a good grid initialization
    # Use m=5 grid (125 cells), pick 100
    m = 5
    positions = _random_grid_init(n, m)
    
    # Run multiple annealing runs from different initializations
    best_positions = positions
    best_score = _score(positions)
    
    # Try a few different initial grids
    for trial in range(3):
        # Different grid sizes
        m_trial = 5 + trial
        pos_trial = _random_grid_init(n, m_trial)
        score_trial = _score(pos_trial)
        
        if score_trial > best_score:
            best_positions = pos_trial
            best_score = score_trial
    
    # Run Metropolis annealing
    start_time = time.time()
    time_budget = 100  # Leave 20s buffer
    
    # Run annealing
    pos_annealed, score_annealed = _metropolis_annealing(
        best_positions, n,
        max_iters=150000,
        t0=0.05,
        t1=0.00005,
        step0=0.15
    )
    
    if score_annealed > best_score:
        best_positions = pos_annealed
        best_score = score_annealed
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_budget:
        # Run another round with smaller steps
        pos_annealed2, score_annealed2 = _metropolis_annealing(
            best_positions, n,
            max_iters=100000,
            t0=0.02,
            t1=0.00001,
            step0=0.05
        )
        if score_annealed2 > best_score:
            best_positions = pos_annealed2
            best_score = score_annealed2
    
    # Final local refinement: try small perturbations
    elapsed = time.time() - start_time
    if elapsed < time_budget - 5:
        rng = np.random.default_rng(123)
        for _ in range(5000):
            idx = rng.integers(0, n)
            proposed = best_positions.copy()
            proposed[idx] += rng.normal(0, 0.01, size=3)
            proposed = np.clip(proposed, -0.5, 0.5)
            s = _score(proposed)
            if s > best_score:
                best_positions = proposed
                best_score = s
            if time.time() - start_time > time_budget - 5:
                break
    
    return best_positions
# EVOLVE-BLOCK-END
