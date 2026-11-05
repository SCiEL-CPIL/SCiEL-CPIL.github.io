"""B2[2] set in 1..1000 via model-and-optimize with accelerated repair.

Key improvements over the parent:
  1. Pre-allocated arrays in greedy paths eliminate O(n) copies per iteration.
  2. A remove-one-add-two phase in the repair loop provides a net +1 growth
     path that 1-for-1 swaps cannot reach when the outside pool is sparse.
  3. Multiple repair restarts (up to ~10) are feasible within the deadline,
     diversifying the local-search landscape.
  4. Better swap acceptance: try several removals per round and pick the
     candidate that leaves the most slack (min-max of resulting counts).
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 120.0
START_TIME = time.time()


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen, cap=1000):
    """Validate every sum has <=2 reps a<=b. Returns bool."""
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    if n == 0:
        return True
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    counts = np.bincount(sums, minlength=2 * cap + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=1000, tlimit=8.0):
    """Greedy B2[2] build in a given order with pre-allocated count array."""
    c = np.empty(cap, dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t0 = time.time()
    for x in order:
        if time.time() - t0 > tlimit:
            break
        if n == 0:
            s = 2 * x
            if counts[s] < 2:
                counts[s] += 1
                c[n] = x
                n += 1
        else:
            # check all sums x + c[0..n-1] and 2x
            ok = True
            for i in range(n):
                s = x + c[i]
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            if ok:
                for i in range(n):
                    counts[x + c[i]] += 1
                counts[2 * x] += 1
                c[n] = x
                n += 1
    return c[:n].tolist()


def _greedy_build_np(order, cap=1000, tlimit=8.0):
    """Vectorized-assisted greedy: check feasibility in bulk where possible."""
    c = np.empty(cap, dtype=np.int64)
    n = 0
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t0 = time.time()
    for x in order:
        if time.time() - t0 > tlimit:
            break
        if n == 0:
            s = 2 * x
            if counts[s] < 2:
                counts[s] += 1
                c[n] = x
                n += 1
        else:
            arr = c[:n]
            sums = arr + x
            if (counts[sums] >= 2).any():
                continue
            s2 = 2 * x
            if counts[s2] >= 2:
                continue
            counts[sums] += 1
            counts[s2] += 1
            c[n] = x
            n += 1
    return c[:n].tolist()


def _repair_grow(seed_set, cap=1000, time_fn=None, stop_fn=None,
                 max_rounds=500, seed=0, max_attempts=50000):
    """Phased local improvement with remove-one-add-one and remove-one-add-two."""
    rng = random.Random(seed)
    best = list(seed_set)
    if not best:
        return best
    best_set = set(best)
    outside = [v for v in range(1, cap + 1) if v not in best_set]

    c = np.asarray(best, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)

    def rebuild(ch):
        cc = np.asarray(ch, dtype=np.int64)
        nn = len(cc)
        if nn == 0:
            return np.zeros(2 * cap + 1, dtype=np.int32), nn
        i2, j2 = np.triu_indices(nn)
        return np.bincount(cc[i2] + cc[j2], minlength=2 * cap + 1).astype(np.int32), nn

    attempts = 0
    for _ in range(max_rounds):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        if attempts >= max_attempts:
            break

        c = np.asarray(best, dtype=np.int64)
        n = len(c)

        # Phase 1: pure adds
        if outside:
            oa = np.asarray(outside, dtype=np.int64)
            if n == 0:
                new_sums_all = 2 * oa[:, None]
            else:
                cross = oa[:, None] + c[None, :]
                new_sums_all = np.concatenate([cross, 2 * oa[:, None]], axis=1)
            gathered = counts[new_sums_all] + 1
            cand_max = gathered.max(axis=1)
            feasible = np.where(cand_max <= 2)[0]
            if feasible.size:
                tight = gathered[feasible].max(axis=1)
                bi = int(feasible[int(np.argmin(tight))])
                x = int(oa[bi])
                ns = new_sums_all[bi]
                best.append(x)
                best_set.add(x)
                outside.remove(x)
                c = np.concatenate([c, np.array([x], dtype=np.int64)])
                n += 1
                counts[ns] += 1
                attempts += 1
                continue

        # Phase 2: remove-one-add-one swaps
        if outside and n > 0:
            found_swap = False
            batch_size = min(n, 40)
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                rem_val = int(c[rem])
                trial = list(best)
                del trial[rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size:
                    tight2 = gathered2[feas2].max(axis=1)
                    bi2 = int(feas2[int(np.argmin(tight2))])
                    x = int(oa[bi2])
                    ns2 = new_sums2[bi2]
                    best = trial
                    best.append(x)
                    best_set = set(best)
                    outside.remove(x)
                    cnt_t[ns2] += 1
                    counts = cnt_t
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_swap = True
                    break
            if found_swap:
                continue

        # Phase 3: remove-one-add-two (net +1 growth)
        if len(outside) >= 2 and n > 0:
            found_grow = False
            batch_size = min(n, 20)
            for _ in range(batch_size):
                if time_fn is not None and time_fn() <= 0:
                    break
                if stop_fn is not None and stop_fn() is True:
                    break
                if attempts >= max_attempts:
                    break
                rem = rng.randrange(n)
                trial = list(best)
                del trial[rem]
                cnt_t, _ = rebuild(trial)
                ta = np.asarray(trial, dtype=np.int64)
                oa = np.asarray(outside, dtype=np.int64)
                # Find first outside x that fits
                if ta.size == 0:
                    new_sums2 = 2 * oa[:, None]
                else:
                    cross2 = oa[:, None] + ta[None, :]
                    new_sums2 = np.concatenate([cross2, 2 * oa[:, None]], axis=1)
                gathered2 = cnt_t[new_sums2] + 1
                cand_max2 = gathered2.max(axis=1)
                feas2 = np.where(cand_max2 <= 2)[0]
                if feas2.size < 2:
                    continue
                # Pick two with best slack
                tight2 = gathered2[feas2].max(axis=1)
                idx_sorted = np.argsort(tight2)
                x1 = int(oa[feas2[idx_sorted[0]]])
                x2 = int(oa[feas2[idx_sorted[1]]])
                if x1 == x2:
                    continue
                # Verify both fit simultaneously
                trial2 = trial + [x1, x2]
                cnt_v, _ = rebuild(trial2)
                if cnt_v.max() <= 2:
                    best = trial2
                    best_set = set(best)
                    outside.remove(x1)
                    outside.remove(x2)
                    counts = cnt_v
                    c = np.asarray(best, dtype=np.int64)
                    n = len(c)
                    attempts += 1
                    found_grow = True
                    break
            if found_grow:
                continue

        break
    return best


def _simulated_annealing(seed_set, cap=1000, time_fn=None, stop_fn=None,
                         seed=0, t_start=50.0, t_end=0.01, max_steps=20000):
    """Simulated annealing over binary inclusion vector with B2[2] energy.
    
    Energy = sum over all sums of max(0, count - 2)^2 + penalty for size deviation.
    Moves: flip a single bit (add or remove an element).
    Acceptance: Metropolis criterion on energy change.
    """
    import math
    rng = random.Random(seed)
    
    # Start from seed_set
    chosen = sorted(seed_set)
    n = len(chosen)
    chosen_set = set(chosen)
    outside = [v for v in range(1, cap + 1) if v not in chosen_set]
    
    if n == 0:
        return list(seed_set)
    
    # Build initial counts
    c = np.asarray(chosen, dtype=np.int64)
    ii, jj = np.triu_indices(n)
    counts = np.bincount(c[ii] + c[jj], minlength=2 * cap + 1).astype(np.int32)
    
    def energy_from_counts(cnts, sz):
        """Energy: penalty for sums exceeding 2, plus soft penalty for small size."""
        over = np.maximum(cnts - 2, 0)
        penalty = float(over @ over)  # sum of squares
        # Encourage larger sets: subtract a small reward per element
        reward = 0.001 * sz
        return penalty - reward
    
    current_energy = energy_from_counts(counts, n)
    best_energy = current_energy
    best_set = list(chosen)
    
    # SA parameters
    T = t_start
    cooling = (t_end / t_start) ** (1.0 / max(max_steps, 1))
    
    steps = 0
    for step in range(max_steps):
        if time_fn is not None and time_fn() <= 2.0:
            break
        if stop_fn is not None and stop_fn() is True:
            break
        
        # Choose a move: flip a random bit
        # Bias towards adding (to grow the set)
        if outside and (n == 0 or rng.random() < 0.6):
            # Add a random outside element
            idx = rng.randrange(len(outside))
            x = outside[idx]
            # Compute new counts if we add x
            c_arr = np.asarray(chosen, dtype=np.int64)
            new_sums = c_arr + x
            new_sums = np.concatenate([new_sums, [2 * x]])
            # Check feasibility: which sums would exceed 2?
            new_counts_trial = counts.copy()
            new_counts_trial[new_sums] += 1
            over = np.maximum(new_counts_trial - 2, 0)
            new_energy = float(over @ over) - 0.001 * (n + 1)
            
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e <= 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
                # Accept
                chosen.append(x)
                chosen_set.add(x)
                outside.pop(idx)
                n += 1
                counts = new_counts_trial
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_set = list(chosen)
        else:
            # Remove a random element from chosen
            if n > 0:
                idx = rng.randrange(n)
                x = chosen[idx]
                c_arr = np.asarray(chosen, dtype=np.int64)
                new_sums = c_arr + x
                new_sums = new_sums[new_sums != 2 * x]  # exclude double-count of 2x
                # Actually need to be careful: remove contributions of x
                # Sums involving x: x + c[i] for all i, and 2x
                # When removing x, we subtract 1 from each sum x + c[i] (i != idx) and 2x
                new_counts_trial = counts.copy()
                for i in range(n):
                    if i == idx:
                        continue
                    s = x + int(c_arr[i])
                    new_counts_trial[s] -= 1
                new_counts_trial[2 * x] -= 1
                # Clip to 0
                new_counts_trial = np.maximum(new_counts_trial, 0)
                over = np.maximum(new_counts_trial - 2, 0)
                new_energy = float(over @ over) - 0.001 * (n - 1)
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
                    # Accept removal
                    outside.append(x)
                    chosen_set.discard(x)
                    chosen.pop(idx)
                    n -= 1
                    counts = new_counts_trial
                    current_energy = new_energy
        
        # Cool down
        T *= cooling
        steps += 1
    
    # Final polish: try to add any remaining outside elements that fit
    c_final = np.asarray(best_set, dtype=np.int64)
    nf = len(c_final)
    if nf > 0:
        ii_f, jj_f = np.triu_indices(nf)
        cnts_f = np.bincount(c_final[ii_f] + c_final[jj_f], minlength=2 * cap + 1).astype(np.int32)
    else:
        cnts_f = np.zeros(2 * cap + 1, dtype=np.int32)
    
    best_set_list = list(best_set)
    best_set_s = set(best_set_list)
    outside_f = [v for v in range(1, cap + 1) if v not in best_set_s]
    
    for x in outside_f:
        if time_fn is not None and time_fn() <= 1.0:
            break
        if not best_set_list:
            continue
        c_tmp = np.asarray(best_set_list, dtype=np.int64)
        new_sums = c_tmp + x
        new_sums = np.concatenate([new_sums, [2 * x]])
        if (cnts_f[new_sums] >= 2).any():
            continue
        best_set_list.append(x)
        best_set_s.add(x)
        cnts_f[new_sums] += 1
    
    return best_set_list


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 120.0
    cap = 1000

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 8.0

    best = []
    best_len = -1

    def consider(s):
        nonlocal best, best_len
        if s and _valid(s, cap=cap) and len(s) > best_len:
            best_len = len(s)
            best = list(s)

    rng = random.Random(42)

    # --- Multiple greedy builds with different orderings ---
    orders = []
    # Ascending
    orders.append(list(range(1, cap + 1)))
    # Descending
    orders.append(list(range(cap, 0, -1)))
    # Random shuffles
    for _ in range(8):
        full = list(range(1, cap + 1))
        rng.shuffle(full)
        orders.append(full)
    # Prime-biased
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                return False
        return True
    primes = [x for x in range(1, cap + 1) if is_prime(x)]
    nonprimes = [x for x in range(1, cap + 1) if not is_prime(x)]
    orders.append(primes + nonprimes)
    # Square-biased
    squares = [i * i for i in range(1, int(cap ** 0.5) + 1) if i * i <= cap]
    nonsq = [x for x in range(1, cap + 1) if x not in set(squares)]
    orders.append(squares + nonsq)

    for order in orders:
        if stopped():
            break
        s = _greedy_build_np(order, cap=cap, tlimit=5.0)
        consider(s)

    # --- Repair/grow on the best with multiple restarts ---
    if best and _time_left() > 10.0:
        for restart in range(10):
            if stopped():
                break
            if _time_left() < 12.0:
                break
            seed_r = 100 + restart * 7
            improved = _repair_grow(best, cap=cap, time_fn=time_left, stop_fn=stopped,
                                    max_rounds=400, seed=seed_r, max_attempts=40000)
            consider(improved)

    # --- Simulated annealing over bit-flips with B2[2] energy ---
    if best and _time_left() > 15.0:
        improved_sa = _simulated_annealing(best, cap=cap, time_fn=time_left,
                                           stop_fn=stopped, seed=999)
        consider(improved_sa)

    # --- If we have time, try a second wave of greedy + repair ---
    if best and _time_left() > 20.0:
        for _ in range(4):
            if stopped():
                break
            full = list(range(1, cap + 1))
            rng.shuffle(full)
            s = _greedy_build_np(full, cap=cap, tlimit=4.0)
            consider(s)
            if _time_left() > 15.0:
                improved = _repair_grow(s, cap=cap, time_fn=time_left, stop_fn=stopped,
                                        max_rounds=300, seed=rng.randint(1, 99999),
                                        max_attempts=30000)
                consider(improved)

    if not best:
        full = list(range(1, cap + 1))
        chosen = _greedy_build_np(full, cap=cap, tlimit=10.0)
        best = chosen

    # Final validation guard
    if best and not _valid(best, cap=cap):
        full = list(range(1, cap + 1))
        chosen = _greedy_build_np(full, cap=cap, tlimit=10.0)
        best = chosen

    return best
