"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


_primes_list = _primes(N)
_num_primes = len(_primes_list)
_primes_sorted = sorted(_primes_list)
_n_primes = len(_primes_sorted)


def _precompute_indices(primes):
    indices = {}
    for p in primes:
        idx = [np.arange(r, N + 1, p, dtype=np.int32) for r in range(p)]
        indices[p] = idx
    return indices


_indices = _precompute_indices(_primes_list)


def _build_valid_mask(config):
    valid = np.ones(N + 1, dtype=bool)
    for p in _primes_list:
        valid[_indices[p][config[p]]] = False
    return valid


def _compute_score(valid_mask):
    return int(np.sum(valid_mask))


def _greedy_init(seed=None):
    """Greedy initialization: for each prime, pick residue class that removes fewest remaining points."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    active = np.ones(N + 1, dtype=bool)
    config = {}
    
    # Process primes in order of increasing size (smaller primes have more residue classes
    # but each removes fewer points; larger primes remove more points per class)
    # Actually, we want to minimize total removals. Let's process larger primes first
    # since they remove more points per choice, so the choice matters more.
    for p in sorted(_primes_list, reverse=True):
        residue_counts = np.zeros(p, dtype=np.int64)
        for r in range(p):
            idx = _indices[p][r]
            residue_counts[r] = np.sum(active[idx])
        min_count = np.min(residue_counts)
        candidates = np.where(residue_counts == min_count)[0]
        if len(candidates) > 1 and rng.rand() < 0.3:
            chosen_r = int(candidates[rng.randint(len(candidates))])
        else:
            chosen_r = int(candidates[0])
        config[p] = chosen_r
        active[_indices[p][chosen_r]] = False
    
    return config, int(np.sum(active))


def _local_search(config, start_score, deadline, max_iter=50000):
    """
    Local search: try changing one prime's residue at a time.
    Use incremental score updates for speed.
    """
    rng = np.random.RandomState(int(time.time()) % 100000)

    current = config.copy()
    valid_mask = _build_valid_mask(current)
    current_score = _compute_score(valid_mask)

    best_config = current.copy()
    best_score = start_score

    no_improvement = 0
    iter_count = 0

    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        # Try all primes, pick best move
        best_move_score = -1
        best_move_p = -1
        best_move_r = -1

        # Randomly order primes to avoid bias
        prime_order = list(_primes_sorted)
        rng.shuffle(prime_order)

        for p in prime_order:
            old_r = current[p]
            old_idx = _indices[p][old_r]

            for new_r in range(p):
                if new_r == old_r:
                    continue

                new_idx = _indices[p][new_r]
                
                # Compute new score: add back old_idx points that are still valid,
                # remove new_idx points that are currently valid
                # Points in old_idx that are currently False (removed by other primes) won't be added back
                # Points in new_idx that are currently True will be removed
                
                # Count how many in old_idx are currently False (will be re-added)
                old_recovered = np.sum(~valid_mask[old_idx])
                # Count how many in new_idx are currently True (will be removed)
                new_removed = np.sum(valid_mask[new_idx])
                
                new_score = current_score + old_recovered - new_removed

                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_p = p
                    best_move_r = new_r

        if best_move_p == -1 or best_move_score <= current_score:
            # No improving move found, try random perturbation
            num_perturb = rng.randint(1, 4)
            for _ in range(num_perturb):
                p = _primes_sorted[rng.randint(len(_primes_sorted))]
                old_r = current[p]
                new_r = rng.randint(p)
                if new_r != old_r:
                    old_idx = _indices[p][old_r]
                    new_idx = _indices[p][new_r]
                    valid_mask[old_idx] = True
                    valid_mask[new_idx] = False
                    current[p] = new_r
                    current_score = _compute_score(valid_mask)
            no_improvement += 5
        else:
            # Apply best move
            old_r = current[best_move_p]
            old_idx = _indices[best_move_p][old_r]
            new_idx = _indices[best_move_p][best_move_r]
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[best_move_p] = best_move_r
            current_score = best_move_score

            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
                no_improvement = 0
            else:
                no_improvement += 1

        # Occasionally accept worse moves to escape local optima
        if no_improvement > 20 and rng.rand() < 0.15:
            p = _primes_sorted[rng.randint(len(_primes_sorted))]
            old_r = current[p]
            new_r = rng.randint(p)
            if new_r != old_r:
                old_idx = _indices[p][old_r]
                new_idx = _indices[p][new_r]
                valid_mask[old_idx] = True
                valid_mask[new_idx] = False
                current[p] = new_r
                current_score = _compute_score(valid_mask)
                no_improvement = 0

    return best_config, best_score


def _simulated_annealing(config, start_score, deadline, max_iter=100000):
    """
    Simulated annealing: explore the space of residue configurations.
    """
    rng = np.random.RandomState(int(time.time()) % 100000 + 12345)

    current = config.copy()
    valid_mask = _build_valid_mask(current)
    current_score = _compute_score(valid_mask)

    best_config = current.copy()
    best_score = start_score

    temperature = 10.0
    cooling = 0.9999
    min_temperature = 0.1

    iter_count = 0
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        # Pick a random prime and a new residue
        p = _primes_sorted[rng.randint(len(_primes_sorted))]
        old_r = current[p]
        new_r = rng.randint(p)

        if new_r == old_r:
            continue

        old_idx = _indices[p][old_r]
        new_idx = _indices[p][new_r]

        # Compute score delta
        old_recovered = np.sum(~valid_mask[old_idx])
        new_removed = np.sum(valid_mask[new_idx])
        delta = old_recovered - new_removed
        new_score = current_score + delta

        if delta >= 0 or rng.rand() < math.exp(delta / temperature):
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[p] = new_r
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
        else:
            # Revert (no change needed since we didn't modify valid_mask)
            pass

        temperature *= cooling
        if temperature < min_temperature:
            temperature = min_temperature

    return best_config, best_score


def _add_back_search(config, start_score, deadline):
    """
    Try to add back points that were removed by trying different residue assignments.
    For each removed point, check if we can adjust some prime's residue to include it
    without losing too many other points.
    """
    rng = np.random.RandomState(int(time.time()) % 100000 + 99999)

    current = config.copy()
    valid_mask = _build_valid_mask(current)
    current_score = _compute_score(valid_mask)

    best_config = current.copy()
    best_score = start_score

    # Get all removed points
    removed_points = np.where(~valid_mask)[0]
    if len(removed_points) == 0:
        return best_config, best_score

    iter_count = 0
    max_iter = 20000
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1

        # Pick a random removed point
        pt = int(removed_points[rng.randint(len(removed_points))])

        # For each prime, check if we can change its residue to include pt
        best_p = -1
        best_r = -1
        best_delta = -np.inf

        for p in _primes_sorted:
            current_r = current[p]
            # What residue class does pt fall into mod p?
            target_r = pt % p
            
            if target_r == current_r:
                continue

            old_idx = _indices[p][current_r]
            new_idx = _indices[p][target_r]

            old_recovered = np.sum(~valid_mask[old_idx])
            new_removed = np.sum(valid_mask[new_idx])
            delta = old_recovered - new_removed

            if delta > best_delta:
                best_delta = delta
                best_p = p
                best_r = target_r

        if best_p != -1 and best_delta >= 0:
            # Apply the change
            p = best_p
            old_r = current[p]
            old_idx = _indices[p][old_r]
            new_idx = _indices[p][best_r]
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[p] = best_r
            current_score += best_delta

            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
                # Update removed points
                removed_points = np.where(~valid_mask)[0]
                if len(removed_points) == 0:
                    break
        elif best_p != -1 and best_delta > -5 and rng.rand() < 0.3:
            # Accept small negative delta to explore
            p = best_p
            old_r = current[p]
            old_idx = _indices[p][old_r]
            new_idx = _indices[p][best_r]
            valid_mask[old_idx] = True
            valid_mask[new_idx] = False
            current[p] = best_r
            current_score += best_delta

            if current_score > best_score:
                best_score = current_score
                best_config = current.copy()
            removed_points = np.where(~valid_mask)[0]
            if len(removed_points) == 0:
                break

    return best_config, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit

    # Phase 1: Greedy initialization with multiple random seeds
    best_config = None
    best_score = -1

    for seed in range(20):
        if time.time() > deadline:
            break
        config, score = _greedy_init(seed)
        if score > best_score:
            best_score = score
            best_config = config

    # Phase 2: Local search to refine the best configuration
    if time.time() < deadline:
        local_deadline = min(deadline, start_time + 120.0)
        best_config, best_score = _local_search(best_config, best_score, local_deadline)

    # Phase 3: Add-back search
    if time.time() < deadline:
        ab_deadline = min(deadline, start_time + 200.0)
        best_config, best_score = _add_back_search(best_config, best_score, ab_deadline)

    # Phase 4: Simulated annealing for further refinement
    if time.time() < deadline:
        sa_deadline = deadline
        best_config, best_score = _simulated_annealing(best_config, best_score, sa_deadline)

    # Phase 5: Final local search polish
    if time.time() < deadline:
        final_deadline = deadline
        best_config, best_score = _local_search(best_config, best_score, final_deadline)

    # Extract the final list of points
    valid_mask = _build_valid_mask(best_config)
    result = [int(x) for x in np.where(valid_mask)[0]]
    return sorted(result)
# EVOLVE-BLOCK-END
