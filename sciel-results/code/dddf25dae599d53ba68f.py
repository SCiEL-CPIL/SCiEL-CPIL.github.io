import time
import random
import math

N = 20
MAXV = 10000


def _build_pool():
    """Generate a structured pool: 2,3,5,7,11,13-smooth numbers <= MAXV, plus a
    few dense low-value and geometric-style anchors. Distinct, sorted."""
    primes = [2, 3, 5, 7, 11, 13]
    pool = set()
    # smooth numbers
    def gen(idx, val):
        if val > MAXV:
            return
        if idx == len(primes):
            if 1 <= val <= MAXV:
                pool.add(val)
            return
        p = primes[idx]
        v = 1
        while v <= MAXV:
            gen(idx + 1, val * v)
            v *= p
    gen(0, 1)
    # ensure some small integers and powers present
    for i in range(1, 201):
        pool.add(i)
    p2 = 1
    while p2 <= MAXV:
        pool.add(p2)
        p2 *= 2
    pool = sorted(pool)
    return pool


def _make_scorer():
    """Return (score, apply_move, revert_move) with incremental sum/product tracking.

    State: sum_counts dict, prod_counts dict, and the current set A (list).
    score() returns max(distinct_sums, distinct_prods).
    """
    sum_counts = {}
    prod_counts = {}
    A = []

    def rebuild(newA):
        sum_counts.clear()
        prod_counts.clear()
        for i in range(len(newA)):
            ai = newA[i]
            for j in range(i, len(newA)):
                aj = newA[j]
                s = ai + aj
                sum_counts[s] = sum_counts.get(s, 0) + 1
                pr = ai * aj
                prod_counts[pr] = prod_counts.get(pr, 0) + 1

    def score():
        # distinct counts = number of keys with count>0 (all are >0 by construction)
        return max(len(sum_counts), len(prod_counts))

    def apply_move(old_idx, new_val):
        """Replace A[old_idx] with new_val, updating counts incrementally."""
        old_val = A[old_idx]
        if old_val == new_val:
            return
        # remove contributions of old_val with every element (j>=old_idx handled by
        # iterating over all partners; we must remove each pair exactly once).
        # Pair (old_idx, j) for j in [old_idx, N) contributed old_val + A[j] and old_val*A[j].
        # Also pair (j, old_idx) for j < old_idx contributed A[j]+old_val and A[j]*old_val.
        # To keep it simple and correct, remove all pairs involving old_idx.
        for j in range(N):
            if j == old_idx:
                continue
            s = old_val + A[j]
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            pr = old_val * A[j]
            prod_counts[pr] -= 1
            if prod_counts[pr] == 0:
                del prod_counts[pr]
        # remove the diagonal (old_idx, old_idx)
        s = 2 * old_val
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        pr = old_val * old_val
        prod_counts[pr] -= 1
        if prod_counts[pr] == 0:
            del prod_counts[pr]

        A[old_idx] = new_val

        # add contributions of new_val with every element
        for j in range(N):
            if j == old_idx:
                continue
            s = new_val + A[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            pr = new_val * A[j]
            prod_counts[pr] = prod_counts.get(pr, 0) + 1
        # add diagonal
        s = 2 * new_val
        sum_counts[s] = sum_counts.get(s, 0) + 1
        pr = new_val * new_val
        prod_counts[pr] = prod_counts.get(pr, 0) + 1

    def revert_move(old_idx, old_val):
        """Undo: A[old_idx] currently holds the 'new' value; put old_val back."""
        new_val = A[old_idx]
        if old_val == new_val:
            return
        # remove new_val contributions
        for j in range(N):
            if j == old_idx:
                continue
            s = new_val + A[j]
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            pr = new_val * A[j]
            prod_counts[pr] -= 1
            if prod_counts[pr] == 0:
                del prod_counts[pr]
        s = 2 * new_val
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        pr = new_val * new_val
        prod_counts[pr] -= 1
        if prod_counts[pr] == 0:
            del prod_counts[pr]

        A[old_idx] = old_val

        # add old_val contributions
        for j in range(N):
            if j == old_idx:
                continue
            s = old_val + A[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            pr = old_val * A[j]
            prod_counts[pr] = prod_counts.get(pr, 0) + 1
        s = 2 * old_val
        sum_counts[s] = sum_counts.get(s, 0) + 1
        pr = old_val * old_val
        prod_counts[pr] = prod_counts.get(pr, 0) + 1

    def init(newA):
        A[:] = list(newA)
        rebuild(A)

    return score, apply_move, revert_move, init


def _local_search(start_set, pool, deadline):
    """Best-improvement local search with single-swap and pair-swap moves using
    the incremental scorer. Returns (best_set, best_score)."""
    score, apply_move, revert_move, init = _make_scorer()
    init(start_set)
    current = start_set[:]
    cur_score = score()
    best = start_set[:]
    best_score = cur_score

    pool_set = set(pool)

    def time_left():
        return time.time() < deadline

    # Single-swap best improvement (iterate until no improvement or time up)
    improved = True
    while improved and time_left():
        improved = False
        best_delta = 0
        best_move = None
        cur_score_now = score()
        for idx in range(N):
            old_val = current[idx]
            # candidate new values: sample pool for speed but also include a few
            # targeted collision values.
            # For correctness and speed, evaluate a bounded candidate set.
            cands = []
            # small targeted set: values that create product collisions with existing
            for v in pool:
                if v == old_val:
                    continue
                if v in current:
                    continue
                cands.append(v)
            # limit candidates to keep per-iteration cost bounded
            if len(cands) > 400:
                # deterministic subset: evenly spaced through pool
                step = len(cands) / 400.0
                cands = [cands[int(k * step)] for k in range(400)]
            for nv in cands:
                apply_move(idx, nv)
                sc = score()
                revert_move(idx, old_val)
                delta = cur_score_now - sc
                if delta > best_delta:
                    best_delta = delta
                    best_move = (idx, old_val, nv, sc)
        if best_move is not None:
            idx, old_val, nv, sc = best_move
            apply_move(idx, nv)
            current[idx] = nv
            cur_score = sc
            if cur_score < best_score:
                best_score = cur_score
                best = current[:]
            improved = True

    # Pair-swap phase
    improved = True
    while improved and time_left():
        improved = False
        best_delta = 0
        best_pair = None
        cur_score_now = score()
        for i in range(N):
            for j in range(i + 1, N):
                vi, vj = current[i], current[j]
                # swap
                current[i], current[j] = vj, vi
                sc = score()
                current[i], current[j] = vi, vj
                delta = cur_score_now - sc
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j, sc)
        if best_pair is not None:
            i, j, sc = best_pair
            current[i], current[j] = current[j], current[i]
            cur_score = sc
            if cur_score < best_score:
                best_score = cur_score
                best = current[:]
            improved = True

    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    pool = _build_pool()
    random.seed(12345)

    # Build initial candidates
    candidates = []

    # 1. Arithmetic progression 1..20
    candidates.append(list(range(1, 21)))
    # 2. Odd AP
    candidates.append(list(range(1, 41, 2)))
    # 3. First 20 smooth numbers
    candidates.append(pool[:20])
    # 4. Powers of 2 padded with small ints to reach 20 distinct
    p2 = []
    v = 1
    while len(p2) < 14 and v <= MAXV:
        p2.append(v)
        v *= 2
    base = set(p2)
    fill = [x for x in pool if x not in base]
    while len(base) < 20 and fill:
        base.add(fill.pop(0))
    candidates.append(sorted(base))
    # 5-9. Random smooth
    for _ in range(5):
        candidates.append(sorted(random.sample(pool, 20)))
    # 10-12. Random uniform
    for _ in range(3):
        candidates.append(sorted(random.sample(range(1, MAXV + 1), 20)))

    best_set = None
    best_score = float('inf')

    # Evaluate all candidates first, keep the best as anchor
    for cand in candidates:
        # quick exact score
        sums = set()
        prods = set()
        for i in range(N):
            ai = cand[i]
            for j in range(i, N):
                sums.add(ai + cand[j])
                prods.add(ai * cand[j])
        sc = max(len(sums), len(prods))
        if sc < best_score:
            best_score = sc
            best_set = cand[:]

    # Run local search on each candidate while time permits
    for cand in candidates:
        if time.time() >= deadline:
            break
        res_set, res_score = _local_search(cand, pool, deadline)
        if res_score < best_score:
            best_score = res_score
            best_set = res_set

    # Final polish: a few extra restarts from the current best with randomization
    if best_set is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() >= deadline:
                break
            perturbed = best_set[:]
            for _k in range(3):
                idx = random.randrange(N)
                nv = random.choice(pool)
                if nv not in perturbed:
                    perturbed[idx] = nv
            res_set, res_score = _local_search(perturbed, pool, deadline)
            if res_score < best_score:
                best_score = res_score
                best_set = res_set

    if best_set is None:
        best_set = list(range(1, 21))

    return best_set
