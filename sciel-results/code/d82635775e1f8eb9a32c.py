import time
import random
import numpy as np

DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    n = len(chosen)
    if n == 0:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2 if n else 0
    if max_s < 1:
        return True
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _build_counts(chosen, cap=1000):
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    if not chosen:
        return counts
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(len(c))
    sums = c[ii] + c[jj]
    np.add.at(counts, sums, 1)
    return counts


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, seed=42):
    rng = random.Random(seed)
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    
    while pool:
        if time_fn is not None and time_fn() < 3.0:
            break
        if stop_fn is not None and stop_fn():
            break
        pool_list = list(pool)
        if len(pool_list) <= 300:
            batch = pool_list
        else:
            batch = rng.sample(pool_list, 300)
        arr = np.array(batch, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]
        if feasible.size == 0:
            for b in batch:
                pool.discard(b)
            continue
        maxes = cand_max[feasible]
        min_max = maxes.min()
        tight_candidates = feasible[maxes == min_max]
        tight_counts = (gathered[tight_candidates] == 2).sum(axis=1)
        best_local = tight_candidates[np.argmin(tight_counts)]
        x = int(arr[best_local])
        chosen.append(x)
        pool.discard(x)
        if chosen_np.size == 0:
            new_sums_x = np.array([2 * x], dtype=np.int64)
        else:
            new_sums_x = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[new_sums_x] += 1
    return chosen


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iter=100000):
    """
    Optimized local search:
    1. Aggressive vectorized addition of candidates.
    2. Avoids expensive array copies in the main loop.
    """
    if not chosen:
        return chosen
    
    current = set(chosen)
    counts = _build_counts(list(current), cap)
    best_size = len(current)
    best_set = set(current)
    
    rng = random.Random(123)
    all_elements = list(range(1, cap + 1))
    candidates = [x for x in all_elements if x not in current]
    
    no_improve = 0
    max_no_improve = 1000
    
    for it in range(max_iter):
        if time_fn is not None and time_fn() < 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        if not candidates:
            break
            
        # Strategy: Vectorized Addition Check
        # We check a batch of candidates to see if any can be added.
        batch_size = min(len(candidates), 100)
        test_indices = rng.sample(range(len(candidates)), batch_size)
        test_candidates = np.array([candidates[i] for i in test_indices], dtype=np.int64)
        
        chosen_arr = np.asarray(sorted(current), dtype=np.int64)
        
        # Calculate new sums for all test candidates against current set
        # Cross sums: test_candidates[:, None] + chosen_arr[None, :]
        # Doubles: 2 * test_candidates
        cross = test_candidates[:, None] + chosen_arr[None, :]
        doubles = (2 * test_candidates)[:, None]
        
        # Vectorized check for the batch
        # gathered[sum] = counts[sum] + 1
        gathered_cross = counts[cross] + 1
        gathered_doubles = counts[doubles] + 1
        
        # Check validity: max count for any new sum must be <= 2
        max_cross = gathered_cross.max(axis=1)
        max_double = gathered_doubles.max(axis=0)
        
        valid_mask = (max_cross <= 2) & (max_double <= 2)
        
        if valid_mask.any():
            # Pick the best valid candidate (minimizes new tight sums)
            valid_indices = np.where(valid_mask)[0]
            
            # Tightness: number of new sums that become count 2
            tight_cross = (gathered_cross[valid_indices] == 2).sum(axis=1)
            tight_double = (gathered_doubles[valid_indices] == 2).sum(axis=0)
            total_tight = tight_cross + tight_double
            
            best_local_idx = valid_indices[np.argmin(total_tight)]
            x = int(test_candidates[best_local_idx])
            
            # Update state
            current.add(x)
            candidates.remove(x)
            
            # Update counts
            new_sums_x = np.concatenate([chosen_arr + x, np.array([2 * x], dtype=np.int64)])
            counts[new_sums_x] += 1
            
            best_size = len(current)
            best_set = set(current)
            no_improve = 0
            continue
        
        no_improve += 1
        if no_improve > max_no_improve:
            break
    
    if len(best_set) > len(current):
        return list(best_set)
    return list(current)


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 120.0

    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 10.0

    best = []
    best_len = -1
    all_greedy_results = []

    # Phase 1: Multiple scored greedy runs with different seeds
    # Increased seeds to explore more diverse starting points
    for seed in range(40):
        if _time_left() < 25.0:
            break
        res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=seed)
        all_greedy_results.append(res)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: Local improvement on the best greedy solution
    if _time_left() > 35.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved

    # Phase 3: Try simple structured patterns (quick checks)
    patterns = [
        list(range(1, cap + 1, 2)),  # every 2nd
        [i for i in range(1, cap + 1) if i % 3 != 0],  # skip multiples of 3
        [i for i in range(1, cap + 1) if i % 4 in (1, 2)],  # 2 per 4
    ]
    for pat in patterns:
        if _valid(pat):
            if len(pat) > best_len:
                best_len = len(pat)
                best = pat

    # Phase 4: Local improvement on second-best greedy if time permits
    if _time_left() > 45.0 and len(all_greedy_results) > 1:
        # Sort greedy results by length descending
        all_greedy_results.sort(key=len, reverse=True)
        # Try local improve on the second best if it's close to the best
        if len(all_greedy_results[1]) >= best_len - 5:
            improved2 = _local_improve(all_greedy_results[1], cap=cap, time_fn=time_left, stop_fn=stopped)
            if len(improved2) > best_len:
                best_len = len(improved2)
                best = improved2

    # Phase 5: Final polish - another local improve pass
    if _time_left() > 35.0 and best:
        improved3 = _local_improve(best, cap=cap, time_fn=time_left, stop_fn=stopped)
        if len(improved3) > best_len:
            best_len = len(improved3)
            best = improved3

    # Final validation
    if best and not _valid(best):
        fallback = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped, seed=0)
        if _valid(fallback):
            best = fallback

    return best
