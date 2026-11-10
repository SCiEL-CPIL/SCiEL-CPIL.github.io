import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill


def _max_sidon_size_exact(arr, time_limit=2.0):
    """
    Exact branch-and-bound: find largest subset with all pair sums unique (incl. diagonals).
    Returns the size. Uses sorted array, incremental sum-set, and strong pruning.
    """
    n = len(arr)
    if n <= 1:
        return n

    arr_sorted = sorted(arr)

    # Quick check: is the whole set Sidon?
    sums = set()
    ok = True
    for i in range(n):
        for j in range(i, n):
            s = arr_sorted[i] + arr_sorted[j]
            if s in sums:
                ok = False
                break
            sums.add(s)
        if not ok:
            break
    if ok:
        return n

    best = 0
    start_time = time.time()

    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best

        # Time check
        if time.time() - start_time > time_limit:
            return

        # Pruning
        if current_size + (n - idx) <= best:
            return

        if current_size > best:
            best = current_size
            if best == n:
                return

        if idx == n:
            return

        # Option 1: Exclude
        dfs(idx + 1, current_size, current_sums, current_elements)

        # Option 2: Include
        x = arr_sorted[idx]
        can_add = True
        new_sums = set()

        for y in current_elements:
            s = x + y
            if s in current_sums:
                can_add = False
                break
            new_sums.add(s)

        if can_add:
            s_diag = 2 * x
            if s_diag in current_sums:
                can_add = False
            else:
                new_sums.add(s_diag)

        if can_add:
            dfs(idx + 1, current_size + 1, current_sums | new_sums, current_elements + [x])

    dfs(0, 0, set(), [])
    return best


def _collision_density(arr):
    """
    Measure how many pair-sum collisions exist in the set.
    Higher = more collisions = harder to find large Sidon subsets.
    """
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    # Count number of sums that appear more than once
    collisions = sum(1 for c in sums.values() if c > 1)
    return collisions


def _construct():
    global _START_TIME, _DEADLINE

    rng = random.Random(12345)

    # Phase 1: Generate diverse candidate sets
    candidates = []

    # Arithmetic progressions (dense collisions)
    for start in range(0, 20):
        for step in [1, 2, 3, 4, 5, 6]:
            arr = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(arr)

    # Clustered sets: two close groups
    for a in range(0, 60):
        for b in range(a + 20, 100):
            arr = [a + i for i in range(6)] + [b + i for i in range(6)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(arr)

    # Random dense sets
    for _ in range(200):
        arr = rng.sample(range(0, 129), 12)
        candidates.append(sorted(arr))

    # Structured: multiples of small numbers
    for m in [1, 2, 3, 4, 5]:
        arr = [i * m for i in range(12)]
        if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
            candidates.append(arr)

    # Quadratic: i^2 mod something
    for base in range(0, 50):
        arr = [(base + i * i) % 129 for i in range(12)]
        if len(set(arr)) == 12 and all(0 <= x <= 128 for x in arr):
            candidates.append(arr)

    # Phase 2: Evaluate candidates, keep top by collision density first, then exact
    # Sort by collision density (higher is better for our goal)
    candidates_with_density = []
    for arr in candidates:
        d = _collision_density(arr)
        candidates_with_density.append((d, arr))

    candidates_with_density.sort(key=lambda x: -x[0])

    # Evaluate top candidates exactly
    best_arr = None
    best_score = 12  # max possible

    # Evaluate top 50 by density
    for d, arr in candidates_with_density[:50]:
        if time.time() > _DEADLINE - 15.0:
            break
        score = _max_sidon_size_exact(arr, time_limit=1.5)
        if score < best_score:
            best_score = score
            best_arr = arr[:]
            if best_score <= 4:
                break

    if best_arr is None:
        best_arr = candidates_with_density[0][1][:]
        best_score = _max_sidon_size_exact(best_arr, time_limit=2.0)

    # Phase 3: Simulated annealing refinement
    if time.time() < _DEADLINE - 10.0:
        current = best_arr[:]
        current_score = best_score
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        steps = 0
        max_steps = 5000

        while T > T_min and steps < max_steps:
            if time.time() > _DEADLINE - 5.0:
                break

            # Pick a random position and new value
            pos = rng.randint(0, 11)
            old_val = current[pos]
            new_val = rng.randint(0, 128)

            if new_val == old_val:
                continue
            if new_val in [current[i] for i in range(12) if i != pos]:
                continue

            # Build new array
            new_arr = current[:]
            new_arr[pos] = new_val

            # Quick collision density check
            new_density = _collision_density(new_arr)
            old_density = _collision_density(current)

            # Accept based on simulated annealing
            if new_density > old_density:
                current = new_arr
                T *= cooling
            else:
                # Accept with probability exp(-(delta)/T) where delta = old - new
                delta = old_density - new_density
                if delta > 0:
                    prob = min(1.0, 2.718281828 ** (-delta / T))
                    if rng.random() < prob:
                        current = new_arr
                T *= cooling

            # Periodically evaluate exact score
            steps += 1
            if steps % 200 == 0:
                exact_score = _max_sidon_size_exact(current, time_limit=1.0)
                if exact_score < current_score:
                    current_score = exact_score
                    best_arr = current[:]
                    best_score = current_score
                    if best_score <= 4:
                        break

        # Final exact evaluation
        final_score = _max_sidon_size_exact(current, time_limit=3.0)
        if final_score < best_score:
            best_arr = current[:]
            best_score = final_score

    # Final validation
    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(0 <= x <= 128 for x in best_arr)

    return sorted(best_arr)


def construct():
    """Entry point: return exactly 12 distinct integers in [0,128]."""
    return _construct()


# EVOLVE-BLOCK-START
# (construct is defined above; this block is for SCiEL evolution markers)
# EVOLVE-BLOCK-END
