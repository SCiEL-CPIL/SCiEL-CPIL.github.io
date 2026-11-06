import random
import time
import numpy as np

def _sum_counts_vec(mask, N):
    """Number of representations a+b (a<=b) for each sum, using a numpy matrix."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.zeros(2 * N + 1, dtype=np.int32)
    A = idx[:, None]
    B = idx[None, :]
    sums = A + B
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    np.add.at(cnt, sums, 1)
    return cnt

def _can_add_vec(mask, cnt, N):
    """Vectorized check of which candidate elements can be added without creating a count>2."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.ones(N + 1, dtype=bool)
    cand = np.nonzero(~mask)[0]
    if cand.size == 0:
        return np.zeros(N + 1, dtype=bool)
    C = cand[:, None]
    sums = C + idx[None, :]
    new = cnt[sums] + 1
    bad = (new > 2).any(axis=1)
    out = np.zeros(N + 1, dtype=bool)
    out[cand[~bad]] = True
    return out

def _add_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] += 1
    if idx.size:
        cnt[idx + x] += 1
    mask[x] = True

def _remove_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] -= 1
    if idx.size:
        other = idx[idx != x]
        cnt[other + x] -= 1
    mask[x] = False

def _saturating_greedy(N, rng, skip_p=0.0):
    mask = np.zeros(N + 1, dtype=bool)
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if mask[x]:
            continue
        # Vectorized feasibility for this x
        idx = np.nonzero(mask)[0]
        ok = True
        if cnt[2 * x] == 2:
            ok = False
        else:
            if idx.size:
                s = idx + x
                if (cnt[s] == 2).any():
                    ok = False
        if ok:
            if rng.random() < skip_p:
                continue  # skip-noise to diversify seeds
            idx = np.nonzero(mask)[0]
            cnt[2 * x] += 1
            if idx.size:
                cnt[idx + x] += 1
            mask[x] = True
    return mask, cnt

def _local_search(mask, cnt, N, rng, time_limit, best_mask, best_size, lambda_pen=8.0):
    """Local search with insert/swap/remove/double-swap moves that can grow the set."""
    cur_size = int(mask.sum())
    best_size_local = best_size
    best_mask_local = best_mask.copy()

    T = 3.0
    T_min = 0.05
    cooling = 0.99998

    def violation_of(cnt):
        return int(np.maximum(0, cnt - 2).sum())

    cur_viol = violation_of(cnt)
    start_time = time.time()

    while time.time() < time_limit:
        r = rng.random()
        if r < 0.40:
            # INSERT: pick a feasible candidate with lowest conflict pressure
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if feas.any():
                cand = np.nonzero(feas)[0]
                # Score candidates by how many sums they'd push to count 2 (lower is better)
                idx = np.nonzero(mask)[0]
                if idx.size > 0:
                    # For each candidate, count how many sums would go from 1->2
                    C = cand[:, None]
                    sums = C + idx[None, :]
                    # Count sums that are currently at 1 (would become 2)
                    pressure = (cnt[sums] == 1).sum(axis=1)
                    # Pick from lowest pressure candidates
                    min_press = pressure.min()
                    low_press_cand = cand[pressure <= min_press + 1]
                    x = int(rng.choice(low_press_cand))
                else:
                    x = int(rng.choice(cand))
                # delta: size +1, viol 0 (feasible)
                _add_vec(mask, cnt, x, N)
                cur_size += 1
                cur_viol = violation_of(cnt)
                if cur_viol == 0 and cur_size > best_size_local:
                    best_size_local = cur_size
                    best_mask_local = mask.copy()
        elif r < 0.65:
            # SWAP: remove a random in-set, add a random not-in-set
            in_idx = np.nonzero(mask)[0]
            if in_idx.size == 0:
                continue
            x = int(rng.choice(in_idx))
            out_idx = np.nonzero(~mask)[0]
            if out_idx.size == 0:
                continue
            y = int(rng.choice(out_idx))
            # try and revert
            _remove_vec(mask, cnt, x, N)
            _add_vec(mask, cnt, y, N)
            new_viol = violation_of(cnt)
            # size unchanged
            delta_E = lambda_pen * (new_viol - cur_viol)
            accept = (delta_E < 0) or (rng.random() < np.exp(-delta_E / max(T, 1e-9)))
            if accept:
                cur_viol = new_viol
                if cur_viol == 0 and cur_size > best_size_local:
                    best_size_local = cur_size
                    best_mask_local = mask.copy()
            else:
                _remove_vec(mask, cnt, y, N)
                _add_vec(mask, cnt, x, N)
                cur_viol = violation_of(cnt)
        elif r < 0.85:
            # DOUBLE SWAP: remove two, add two different ones
            in_idx = np.nonzero(mask)[0]
            if in_idx.size < 2:
                continue
            x1, x2 = rng.sample(list(in_idx), 2)
            out_idx = np.nonzero(~mask)[0]
            if out_idx.size < 2:
                continue
            y1, y2 = rng.sample(list(out_idx), 2)
            # try and revert
            _remove_vec(mask, cnt, x1, N)
            _remove_vec(mask, cnt, x2, N)
            _add_vec(mask, cnt, y1, N)
            _add_vec(mask, cnt, y2, N)
            new_viol = violation_of(cnt)
            # size unchanged
            delta_E = lambda_pen * (new_viol - cur_viol)
            accept = (delta_E < 0) or (rng.random() < np.exp(-delta_E / max(T, 1e-9)))
            if accept:
                cur_viol = new_viol
                if cur_viol == 0 and cur_size > best_size_local:
                    best_size_local = cur_size
                    best_mask_local = mask.copy()
            else:
                _remove_vec(mask, cnt, y1, N)
                _remove_vec(mask, cnt, y2, N)
                _add_vec(mask, cnt, x1, N)
                _add_vec(mask, cnt, x2, N)
                cur_viol = violation_of(cnt)
        else:
            # REMOVE: only useful to free capacity; accept if it reduces violation
            in_idx = np.nonzero(mask)[0]
            if in_idx.size == 0:
                continue
            x = int(rng.choice(in_idx))
            _remove_vec(mask, cnt, x, N)
            new_viol = violation_of(cnt)
            delta_E = -1 + lambda_pen * (new_viol - cur_viol)
            accept = (delta_E < 0) or (rng.random() < np.exp(-delta_E / max(T, 1e-9)))
            if accept:
                cur_size -= 1
                cur_viol = new_viol
            else:
                _add_vec(mask, cnt, x, N)
                cur_viol = violation_of(cnt)
                cur_size += 1
        T *= cooling
        if T < T_min:
            T = T_min
    return best_mask_local, best_size_local

def construct():
    N = 1000
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    rng = random.Random(12345)

    # Multiple randomized greedy seeds (skip-noise) to diversify
    best_mask = None
    best_size = 0
    seed_time = 18.0
    while time.time() < start_time + seed_time:
        skip_p = rng.random() * 0.25
        mask, cnt = _saturating_greedy(N, rng, skip_p=skip_p)
        s = int(mask.sum())
        if s > best_size:
            best_size = s
            best_mask = mask
            best_cnt = cnt

    # Local search to grow beyond greedy
    if best_mask is None:
        best_mask = np.zeros(N + 1, dtype=bool)
        best_cnt = np.zeros(2 * N + 1, dtype=np.int32)
    else:
        best_cnt = _sum_counts_vec(best_mask, N)

    best_mask, best_size = _local_search(
        best_mask.copy(), best_cnt.copy(), N, rng,
        time_limit, best_mask, best_size, lambda_pen=8.0
    )

    # Final polish: a short second local search pass from the best found
    if time.time() < time_limit - 5.0:
        cnt2 = _sum_counts_vec(best_mask, N)
        best_mask, best_size = _local_search(
            best_mask.copy(), cnt2, N, rng,
            time_limit - 2.0, best_mask, best_size, lambda_pen=6.0
        )

    # Final greedy polish: try to add any remaining feasible elements
    if time.time() < time_limit - 1.0:
        cnt3 = _sum_counts_vec(best_mask, N)
        feas = _can_add_vec(best_mask, cnt3, N)
        feas = feas & ~best_mask
        if feas.any():
            cand = np.nonzero(feas)[0]
            # Add in order of increasing value (greedy)
            for x in sorted(cand):
                if time.time() > time_limit - 0.5:
                    break
                _add_vec(best_mask, cnt3, x, N)
            best_size = int(best_mask.sum())

    result = [int(i) for i in np.nonzero(best_mask)[0]]
    return result
