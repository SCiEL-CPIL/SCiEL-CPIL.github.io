"""Stronger B2[2] set search for {1..1000}.

Strategy:
  - Conflict-degree greedy: model as dynamic independent set, pick candidate
    that minimizes the number of other candidates it blocks.
  - Score-driven greedy with soft-max surrogate.
  - Local improvement via targeted single-swap and multi-swap.
  - Portfolio of randomized runs within a strict time budget.
  - Deadline guard with safety margin.
"""
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0
START_TIME = time.time()

# Soft-max temperature for the scored greedy (lower = sharper argmin)
SOFTMAX_T = 0.35
# Weight for tightness (number of sums with count==1) in composite score
TIGHTNESS_W = 0.15


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


def _greedy_build(order, cap=1000, time_fn=None, stop_fn=None):
    """Greedy: iterate candidates in order; add if B2[2] preserved."""
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


def _conflict_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                     batch_size=100, mode="min_conflict"):
    """Conflict-degree greedy: at each step, evaluate a batch of feasible
    candidates and pick the one that blocks the fewest other candidates.
    This models the dynamic independent set problem."""
    if rng is None:
        rng = random.Random()
    pool = set(range(1, cap + 1))
    chosen = []
    chosen_np = np.empty(0, dtype=np.int64)
    counts = np.zeros(2 * cap + 1, dtype=np.int32)

    while pool:
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break

        # Get a batch of candidates to evaluate
        if len(pool) <= batch_size:
            batch_list = list(pool)
        else:
            batch_list = rng.sample(list(pool), batch_size)
        arr = np.array(batch_list, dtype=np.int64)

        # Check feasibility for each candidate
        if chosen_np.size == 0:
            new_sums = 2 * arr[:, None]
            gathered = counts[new_sums] + 1
            cand_max = gathered.max(axis=1)
        else:
            cross = arr[:, None] + chosen_np[None, :]
            new_sums = np.concatenate([cross, 2 * arr[:, None]], axis=1)
            gathered = counts[new_sums] + 1
            cand_max = gathered.max(axis=1)

        feasible_idx = np.where(cand_max <= 2)[0]
        if feasible_idx.size == 0:
            # No feasible candidates in this batch; remove them from pool
            for b in batch_list:
                pool.discard(b)
            continue

        # For each feasible candidate, compute how many OTHER candidates it would block
        # A candidate x blocks candidate z if adding x makes z infeasible
        # i.e., there exists y in chosen such that c[x+y+z] would exceed... 
        # Actually: x blocks z if after adding x, z becomes infeasible.
        # z is infeasible after adding x if for some y in chosen∪{x}, c'[z+y] > 2
        # where c' is counts after adding x.
        # Simpler: x blocks z if they share a "critical" sum.
        # A sum s is critical if c[s] == 2. Adding x makes sum s+x-y critical...
        #
        # Let's use a simpler proxy: for each feasible x, count how many z in the
        # batch would become infeasible. We can check this directly.
        feasible_candidates = arr[feasible_idx]
        
        # For efficiency, just pick the feasible candidate with minimum "impact"
        # Impact = number of sums that go from 0 to 1 (creates new constraints)
        # Actually, let's use: impact = number of sums that go from 1 to 2 (tightens)
        g = gathered[feasible_idx]
        # Sums that were 1 and become 2
        tighten = np.sum((g == 2) & (g - 1 == 1), axis=1)  # sums that become 2
        # Sums that were 0 and become 1
        create = np.sum((g == 1), axis=1)
        
        # We want to minimize tightening (sums going 1->2) and creating (0->1)
        # But creating is necessary; tightening is bad
        # Score: primarily minimize tighten, secondarily minimize create
        scores = tighten * 10 + create
        best_local = feasible_idx[int(np.argmin(scores))]
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


def _scored_greedy(cap=1000, time_fn=None, stop_fn=None, rng=None,
                   batch_size=150, score_mode="min_tightness",
                   softmax_t=SOFTMAX_T):
    """Score-driven greedy: at each step, evaluate a batch of random candidates
    and pick the one that minimizes a composite score combining soft-max of
    max count with a tightness penalty. Lower scores are better."""
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

        # Composite score: soft-max of max count + tightness penalty
        g = gathered[feasible]
        gmax = g.max(axis=1, keepdims=True)
        # Numerically stable log-sum-exp for soft-max
        lse = gmax + np.log(np.sum(np.exp(g - gmax), axis=1, keepdims=True))
        soft_max_score = lse[:, 0] / t

        # Tightness: count of sums that would become count==1 (more flexibility)
        # Actually we want to AVOID creating too many count==1 sums, so penalty
        # is the number of sums that go from 0 to 1
        tightness = np.sum(g == 1, axis=1) * TIGHTNESS_W

        scores = soft_max_score + tightness
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


def _local_improve(chosen, cap=1000, time_fn=None, stop_fn=None, max_iters=100, rng=None):
    """Optimized local improvement:
    1. Greedy fill (add all feasible).
    2. Hill-climbing swaps: remove one, try to add all feasible. 
       Use incremental count updates to avoid O(K^2) recomputation.
    """
    if rng is None:
        rng = random.Random()
    if not chosen:
        return chosen
    
    best = list(chosen)
    best_set = set(best)
    c = np.asarray(best, dtype=np.int64)
    
    # Initial counts
    if c.size > 0:
        ii, jj = np.triu_indices(c.size)
        sums = c[ii] + c[jj]
        counts = np.bincount(sums, minlength=2 * cap + 1).astype(np.int32)
    else:
        counts = np.zeros(2 * cap + 1, dtype=np.int32)

    def _add_all_feasible():
        nonlocal c, counts, best, best_set
        if c.size == 0:
            return
        cand_list = [v for v in range(1, cap + 1) if v not in best_set]
        if not cand_list:
            return
        ca = np.array(cand_list, dtype=np.int64)
        cross = ca[:, None] + c[None, :]
        ns_all = np.concatenate([cross, 2 * ca[:, None]], axis=1)
        gathered = counts[ns_all] + 1
        feasible = np.where(gathered.max(axis=1) <= 2)[0]
        
        if feasible.size == 0:
            return
            
        # Add all feasible
        for fi in feasible:
            cand = int(ca[fi])
            if cand in best_set:
                continue
            ns = np.concatenate([c + cand, np.array([2 * cand], dtype=np.int64)])
            counts[ns] += 1
            c = np.concatenate([c, np.array([cand], dtype=np.int64)])
            best.append(cand)
            best_set.add(cand)

    # Phase 0: Greedy fill
    _add_all_feasible()

    # Phase 1: Hill Climbing Swaps
    for _ in range(max_iters):
        if time_fn is not None and time_fn() <= 0:
            break
        if stop_fn is not None and stop_fn():
            break
        if c.size == 0:
            break
            
        # Pick a random element to remove
        idx = rng.randrange(c.size)
        removed_val = int(c[idx])
        
        # Update counts: subtract contributions of removed_val
        # Sums involving removed_val: removed_val + c[j] for all j != idx, and 2*removed_val
        # Note: c[idx] is removed_val.
        # We need to decrement counts for:
        # 1. 2 * removed_val
        # 2. removed_val + c[j] for all j in 0..size-1, j != idx
        
        # Create a copy of c without the removed element
        # To avoid O(N) copy of c for every swap, we can just compute the sums to decrement
        # Sums to decrement:
        # s1 = 2 * removed_val
        # s2 = removed_val + c[j] for j != idx
        
        # Efficiently get c without idx
        # np.delete is O(N). N is ~400. This is fast.
        c_temp = np.delete(c, idx)
        
        # Decrement counts
        # 2 * removed_val
        counts[2 * removed_val] -= 1
        
        # removed_val + c_temp
        sums_to_dec = removed_val + c_temp
        # Use np.add.at or bincount trick? 
        # Since counts is an array, we can do:
        # np.add.at(counts, sums_to_dec, -1)
        # But np.add.at is slow. 
        # Alternative: np.bincount(sums_to_dec) and subtract.
        dec_counts = np.bincount(sums_to_dec, minlength=2*cap+1)
        counts -= dec_counts
        
        # Save state to revert if needed
        old_c = c
        old_best = best
        old_best_set = best_set
        old_counts = counts.copy() # O(2000) copy, acceptable.
        
        # Remove from best
        best.pop(idx)
        best_set.discard(removed_val)
        c = c_temp
        
        # Try to add all feasible
        _add_all_feasible()
        
        # Check if improved
        if len(best) > len(old_best):
            # Keep improvement
            # c, counts, best, best_set are already updated in _add_all_feasible
            pass
        else:
            # Revert
            c = old_c
            best = old_best
            best_set = old_best_set
            counts = old_counts

    return best


def construct():
    global START_TIME, DEADLINE_SECONDS
    START_TIME = time.time()
    DEADLINE_SECONDS = 115.0

    cap = 1000
    full = list(range(1, cap + 1))
    best = []
    best_len = -1
    rng = random.Random(42)

    def time_left():
        return _time_left()

    def stopped():
        return _time_left() < 5.0

    # Phase 1: portfolio of diverse greedy constructions
    iteration = 0
    while _time_left() > 15.0:
        iteration += 1
        mode = iteration % 7

        if mode == 0:
            # Conflict-degree greedy (primary)
            res = _conflict_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                   rng=rng, batch_size=100)
        elif mode == 1:
            # Conflict-degree greedy with larger batch
            res = _conflict_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                   rng=rng, batch_size=200)
        elif mode == 2:
            # Scored greedy (composite score)
            res = _scored_greedy(cap=cap, time_fn=time_left, stop_fn=stopped,
                                 rng=rng, batch_size=200, softmax_t=SOFTMAX_T)
        elif mode == 3:
            # Random order greedy
            order = full[:]
            rng.shuffle(order)
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 4:
            # Interleaved low/high
            lows = list(range(1, cap // 2 + 1))
            highs = list(range(cap // 2 + 1, cap + 1))
            rng.shuffle(lows)
            rng.shuffle(highs)
            order = []
            li = hi = 0
            while li < len(lows) or hi < len(highs):
                if li < len(lows):
                    order.append(lows[li]); li += 1
                if hi < len(highs):
                    order.append(highs[hi]); hi += 1
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        elif mode == 5:
            # Descending
            order = full[::-1]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)
        else:
            # Ascending
            order = full[:]
            res = _greedy_build(order, cap=cap, time_fn=time_left, stop_fn=stopped)

        if len(res) > best_len:
            best_len = len(res)
            best = res

    # Phase 2: local improvement (single-swap + multi-swap + greedy fill)
    if _time_left() > 10.0 and best:
        improved = _local_improve(best, cap=cap, time_fn=time_left,
                                  stop_fn=stopped, max_iters=50, rng=rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Final validation
    if best and not _valid(best):
        simple = _greedy_build(full, cap=cap)
        if _valid(simple) and len(simple) > 0:
            best = simple

    return best
