"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=0):
    """Run ACO and return the fewest bins found by any ant."""
    n = len(demands)
    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n))
    best_bins = n  # worst case: one item per bin

    for _ in range(n_iters):
        iteration_best = n
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins = 0
            current = -1
            # Open first bin and place a random fitting item
            while not placed.all():
                if current == -1:
                    # Open a new bin: pick a random fitting item
                    available = np.where(~placed)[0]
                    if len(available) == 0:
                        break
                    current = available[rng.integers(len(available))]
                    placed[current] = True
                    bins += 1
                else:
                    # Pick next item with probability proportional to pheromone[current, j] * heuristic[current, j]
                    available = np.where(~placed)[0]
                    if len(available) == 0:
                        break
                    # Check which fit
                    fits = []
                    probs = []
                    for j in available:
                        # We need to know remaining capacity; but we don't track per-bin capacity in this simplified model
                        # Actually the task says "places a random fitting item" then "repeatedly picks a fitting item"
                        # The heuristic matrix is precomputed. "Fitting" means demands[i]+demands[j] <= capacity? 
                        # Let's re-read: "attractiveness for placing item j right after item i in the same bin"
                        # So fitting means demands[current] + demands[j] <= capacity
                        if demands[current] + demands[j] <= capacity:
                            fits.append(j)
                            probs.append(pheromone[current, j] * heuristic[current, j])
                    if len(fits) == 0:
                        # None fits, open new bin
                        current = -1
                        continue
                    probs = np.array(probs)
                    total = probs.sum()
                    if total <= 0:
                        # Uniform random
                        current = fits[rng.integers(len(fits))]
                    else:
                        r = rng.random()
                        cumsum = np.cumsum(probs) / total
                        idx = np.searchsorted(cumsum, r)
                        if idx >= len(fits):
                            idx = len(fits) - 1
                        current = fits[idx]
                    placed[current] = True

        iteration_best = min(iteration_best, bins)
        # Update pheromone
        pheromone *= 0.9
        if iteration_best > 0:
            pheromone += 1.0 / iteration_best

        best_bins = min(best_bins, bins)

    return best_bins


def heuristics(demands, capacity):
    start = time.time()
    deadline = 300.0  # seconds
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    # Precompute fitness matrix: which pairs fit
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity).astype(float)

    # Strategy: Try multiple heuristic designs and pick the one that performs best
    # on a quick ACO simulation with a few iterations.

    heuristic_candidates = []

    # Candidate 1: Original - pair_sum / capacity for fitting pairs
    h1 = np.where(fits > 0, pair_sum / capacity, 1e-9)
    heuristic_candidates.append(h1)

    # Candidate 2: Inverse of remaining space - prefer pairs that fill the bin tightly
    # remaining = capacity - pair_sum; for fitting pairs, remaining is in [0, capacity]
    # We want large heuristic when remaining is small (tight fit)
    remaining = capacity - pair_sum
    h2 = np.where(fits > 0, 1.0 / (remaining + 1e-9), 1e-9)
    heuristic_candidates.append(h2)

    # Candidate 3: Power-law on fill ratio
    fill_ratio = np.where(fits > 0, pair_sum / capacity, 1e-9)
    h3 = np.where(fits > 0, fill_ratio ** 4.0, 1e-9)
    heuristic_candidates.append(h3)

    # Candidate 4: Combination - reward tight fits and large items
    # Large items are harder to place, so placing them early is good
    item_size = demands
    h4 = np.where(fits > 0, (item_size[:, None] + item_size[None, :]) / (2 * capacity) * (1.0 / (remaining + 1e-9)), 1e-9)
    heuristic_candidates.append(h4)

    # Candidate 5: Exponential decay of remaining space
    h5 = np.where(fits > 0, np.exp(-remaining / (capacity * 0.1)), 1e-9)
    heuristic_candidates.append(h5)

    # Evaluate each candidate with a quick ACO simulation
    # Use a small number of iterations to keep it fast
    quick_ants = 5
    quick_iters = 5

    best_heuristic = heuristic_candidates[0]
    best_score = float('inf')

    for idx, h in enumerate(heuristic_candidates):
        if time.time() > deadline - 60:  # Safety margin
            break
        score = _run_aco(demands, capacity, h, n_ants=quick_ants, n_iters=quick_iters, seed=42)
        if score < best_score:
            best_score = score
            best_heuristic = h

    return best_heuristic
# EVOLVE-BLOCK-END
