"""
B2[2] Set Construction: Hybrid Greedy + Targeted Swap Local Search.

Strategy:
1. Fast vectorized greedy construction using a smooth soft-max surrogate.
2. Portfolio of diverse randomized initializations.
3. Targeted local search with "add-then-swap" logic to maximize size.
4. Strict deadline guard to ensure valid output within time limit.
"""
import time
import random
import numpy as np

# Configuration
CAP = 1000
DEADLINE_SECONDS = 115.0  # Safety margin from 150s kill limit
START_TIME = time.time()
SOFTMAX_T = 0.25  # Temperature for soft-max scoring


def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)


def _valid(chosen):
    """Validate B2[2]: every sum has at most two representations a<=b."""
    if not chosen:
        return True
    c = np.asarray(chosen, dtype=np.int64)
    n = len(c)
    ii, jj = np.triu_indices(n)
    sums = c[ii] + c[jj]
    max_s = int(c.max()) * 2
    counts = np.bincount(sums, minlength=max_s + 1)
    return bool(counts.max() <= 2)


def _greedy_build(order, cap=CAP, time_fn=None, stop_fn=None):
    """
    Greedy build: iterate candidates in `order`; add if B2[2] preserved.
    Returns the chosen list.
    """
    chosen = []
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    chosen_np = np.empty(0, dtype=np.int64)
    for x in order:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if chosen_np.size == 0:
            new_sums = np.array([2 * x], dtype=np.int64)
        else:
            new_sums = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        new_counts = counts[new_sums] + 1
        if new_counts.max() <= 2:
            chosen.append(x)
            chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
            counts[new_sums] += 1
    return chosen


def _scored_greedy(cap=CAP, time_fn=None, stop_fn=None, rng=None,
                   batch_size=150, softmax_t=SOFTMAX_T):
    """
    Score-driven greedy: picks candidate minimizing soft-max surrogate of max count.
    This smooths the objective to guide the search away from tight constraints.
    """
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)
    t = float(softmax_t) if softmax_t and softmax_t > 0 else 1e-9

    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        
        if len(pool) <= batch_size:
            batch_list = list(pool)
            evaluate_all = True
        else:
            batch_list = rng.sample(list(pool), batch_size)
            evaluate_all = False

        arr = np.array(batch_list, dtype=np.int64)
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)

        gathered = counts[new_sums] + 1
        cand_max = gathered.max(axis=1)
        feasible = np.where(cand_max <= 2)[0]

        if feasible.size == 0:
            if evaluate_all:
                break
            else:
                for b in batch_list:
                    pool.discard(b)
                continue

        # Soft-max score: encourages spreading out counts
        g = gathered[feasible]
        gmax = g.max(axis=1, keepdims=True)
        lse = gmax + np.log(np.sum(np.exp(g - gmax), axis=1, keepdims=True))
        scores = lse[:, 0] / t
        best_local = feasible[int(np.argmin(scores))]
        x = int(arr[best_local])
        chosen.append(x)
        pool.discard(x)

        if chosen_np.size == 0:
            ns = np.array([2 * x], dtype=np.int64)
        else:
            ns = np.concatenate([chosen_np + x, np.array([2 * x], dtype=np.int64)])
        chosen_np = np.concatenate([chosen_np, np.array([x], dtype=np.int64)])
        counts[ns] += 1

    return chosen


def _local_improve(chosen, cap=CAP, time_fn=None, stop_fn=None, rng=None):
    """
    Aggressive local search:
    1. Greedy fill: Add any feasible element.
    2. Single-swap: Remove 1, try to add many.
    3. Double-swap: Remove 2, try to add 3.
    Uses vectorized checks for speed.
    """
    if not chosen:
        return chosen
    
    best = list(chosen)
    best_set = set(best)
    
    # Helper to compute counts
    def get_counts(s):
        if not s:
            return np.zeros(2 * cap + 1, dtype=np.int32)
        c = np.asarray(s, dtype=np.int64)
        n = len(c)
        ii, jj = np.triu_indices(n)
        sums = c[ii] + c[jj]
        return np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)

    # Helper to greedily add feasible candidates to a set
    def greedy_add(s, counts, cand_list, limit=None):
        added = []
        c_np = np.asarray(s, dtype=np.int64) if s else np.empty(0, dtype=np.int64)
        
        if c_np.size > 0:
            ca = np.array(cand_list, dtype=np.int64)
            cross = ca[:, None] + c_np[None, :]
            ns_all = np.concatenate([cross, 2 * ca[:, None]], axis=1)
            gathered = counts[ns_all] + 1
            feasible_mask = gathered.max(axis=1) <= 2
            feasible_cands = [cand_list[i] for i in range(len(ca)) if feasible_mask[i]]
        else:
            feasible_cands = list(cand_list)
            
        for cand in feasible_cands:
            if limit is not None and len(added) >= limit:
                break
            if c_np.size == 0:
                ns = np.array([2 * cand], dtype=np.int64)
            else:
                ns = np.concatenate([c_np + cand, np.array([2 * cand], dtype=np.int64)])
            
            if (counts[ns] + 1).max() <= 2:
                s.append(cand)
                counts[ns] += 1
                c_np = np.concatenate([c_np, np.array([cand], dtype=np.int64)])
                added.append(cand)
                
        return added

    # Phase 1: Greedy fill
    counts = get_counts(best)
    changed = True
    while changed and (time_fn is None or time_fn() > 0) and (stop_fn is None or not stop_fn()):
        changed = False
        remaining = [v for v in range(1, cap + 1) if v not in best_set]
        if not remaining:
            break
        added = greedy_add(best, counts, remaining)
        if added:
            best_set.update(added)
            counts = get_counts(best)
            changed = True

    # Phase 2: Single Swap Hill Climbing
    max_iters = 100
    no_improve_count = 0
    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 5.0:
            break
        if stop_fn is not None and stop_fn():
            break
        if no_improve_count > 30:
            break
            
        if not best:
            break
        idx = rng.randrange(len(best))
        removed = best[idx]
        trial = [best[j] for j in range(len(best)) if j != idx]
        trial_set = set(trial)
        
        trial_counts = get_counts(trial)
        
        remaining = [v for v in range(1, cap + 1) if v not in trial_set]
        added = greedy_add(trial, trial_counts, remaining)
        
        if len(trial) > len(best):
            best = trial
            best_set = set(best)
            counts = get_counts(best)
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Phase 3: Double Swap (Remove 2, Add 3)
    if time_fn is not None and time_fn() > 15.0:
        for _ in range(50):
            if time_fn is not None and time_fn() <= 5.0:
                break
            if stop_fn is not None and stop_fn():
                break
            if len(best) < 2:
                break
                
            idx1 = rng.randrange(len(best))
            idx2 = rng.randrange(len(best) - 1)
            if idx2 >= idx1:
                idx2 += 1
                
            r1 = best[idx1]
            r2 = best[idx2]
            
            trial = [best[j] for j in range(len(best)) if j != idx1 and j != idx2]
            trial_set = set(trial)
            trial_counts = get_counts(trial)
            
            remaining = [v for v in range(1, cap + 1) if v not in trial_set]
            added = greedy_add(trial, trial_counts, remaining, limit=3)
            
            if len(trial) > len(best):
                best = trial
                best_set = set(best)
                counts = get_counts(best)

    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0

    cap = CAP
    full = list(range(1, cap + 1))
    best = []
    best_len = -1
    rng = random.Random(42)

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 5.0

    # Phase 1: Portfolio of diverse greedy constructions
    # We run a mix of random and scored greedy to get good starting points
    while _time_left() > 25.0:
        # Run a few different greedy strategies
        for i in range(4):
            if _time_left() < 25.0:
                break
            if i % 2 == 0:
                order = full[:]
                rng.shuffle(order)
                res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
            else:
                # Vary temperature slightly for diversity
                t = SOFTMAX_T * (0.5 + 0.5 * rng.random())
                res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                     rng=rng, batch_size=150, softmax_t=t)
            
            if len(res) > best_len:
                best_len = len(res)
                best = res

    # Phase 2: Local Improvement
    # Apply local search to the best found so far
    if _time_left() > 10.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    # Phase 3: If time remains, try to improve again or try a second best candidate
    if _time_left() > 10.0 and best:
        # Try local search again on the current best (might find more swaps)
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Final validation
    if best and not _valid(best):
        # Fallback: simple greedy
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple
        else:
            best = []

    return best
