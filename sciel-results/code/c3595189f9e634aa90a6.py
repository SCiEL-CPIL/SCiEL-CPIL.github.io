"""Rewritten program: fast greedy + WalkSAT local search with deadline guard."""
import time
import random
import numpy as np

N = 192
MAX_EL = 48
DEADLINE = 78.0  # safety margin before the 90s kill


def _build_tables():
    # For each element e, the list of (sum, idx) for each 3-combination of OTHER elements.
    # sums[e] = array of 3-sums of distinct others (excluding e itself)
    # idxs[e] = array of (i,j,k) triples
    sums = []
    idxs = []
    for e in range(1, N + 1):
        others = [v for v in range(1, N + 1) if v != e]
        s_list = []
        i_list = []
        m = len(others)
        for i in range(m):
            oi = others[i]
            for j in range(i + 1, m):
                oj = others[j]
                for k in range(j + 1, m):
                    ok = others[k]
                    s_list.append(oi + oj + ok)
                    i_list.append((i, j, k))
        sums.append(np.array(s_list, dtype=np.int64))
        idxs.append(i_list)
    return sums, idxs


def _is_valid_set(selected_set, sums, idxs):
    """Check a set (as a Python set of ints) is valid. O(|S|^3) but fast enough for final checks."""
    arr = sorted(selected_set)
    m = len(arr)
    for a_i in range(m):
        a = arr[a_i]
        others = [v for v in arr if v != a]
        mo = len(others)
        for i in range(mo):
            for j in range(i + 1, mo):
                s2 = others[i] + others[j]
                for k in range(j + 1, mo):
                    if (s2 + others[k]) % a == 0:
                        return False
    return True


def _greedy_build(order, sums, idxs, deadline):
    """Greedy add in given order. Returns selected set."""
    selected = set()
    # bad_count[e] = number of 3-sums of OTHER selected elements divisible by e
    bad = [0] * (N + 1)
    # For fast updates, when we add element x, we need to know for each other e
    # how many 3-sums of (selected + x) excluding e are divisible by e.
    # We maintain: for each e, bad[e] = count over triples from (selected \ {e}) with sum % e == 0
    # Adding x: for each e != x, new triples involving x and 2 from (selected \ {e}) that sum%e==0
    # This is complex; instead maintain per-element counts via precomputed structure.

    # Simpler approach: maintain selected as list, and for validity of adding x,
    # check directly using vectorized sums.
    sel_list = []
    for x in order:
        if len(selected) >= MAX_EL:
            break
        if time.time() > deadline:
            break
        # Check if x can be added: for every a in selected + {x}, no 3-sum of others divisible by a
        # We check two things:
        #  (1) x itself: no 3-sum of (selected) divisible by x  [since others of x = selected]
        #  (2) for each a in selected: no 3-sum of (selected \ {a}) + {x} divisible by a
        #      i.e., triples from selected\{a} (already counted) plus triples involving x
        #
        # Efficient: precompute for each x, the 3-sums of selected (all distinct) mod x
        if sel_list:
            sel_arr = np.array(sel_list, dtype=np.int64)
            # (1) check x: all 3-sums of selected not divisible by x
            m = len(sel_arr)
            ok = True
            # 3-sums of selected
            # vectorized: for each pair, sum + third
            # For m up to 48, O(m^3) is fine but let's be smart
            # Use the precomputed sums table: sums[x-1] gives 3-sums of ALL others (1..192 except x)
            # We need only those triples that are all in selected.
            # Instead, compute directly for small m.
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = sel_arr[i] + sel_arr[j]
                    for k in range(j + 1, m):
                        if (s2 + sel_arr[k]) % x == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # (2) for each a in selected: check triples involving x
            for a in sel_list:
                if a == x:
                    continue
                # triples with x and 2 from (selected \ {a})
                for i in range(m):
                    if sel_arr[i] == a:
                        continue
                    for j in range(i + 1, m):
                        if sel_arr[j] == a:
                            continue
                        if (x + sel_arr[i] + sel_arr[j]) % a == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
        selected.add(x)
        sel_list.append(x)
    return selected


def _local_search(selected, sums, idxs, deadline, rng, max_iters=20000):
    """WalkSAT-style local search: try to grow the set by removing random elements and re-adding."""
    best = set(selected)
    best_size = len(best)
    cur = set(selected)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if len(cur) >= MAX_EL:
            break
        # Try to add a random candidate not in cur
        candidates = [v for v in range(1, N + 1) if v not in cur]
        if not candidates:
            break
        # Try a few random candidates
        improved = False
        for _ in range(5):
            x = rng.choice(candidates)
            # Check if x can be added
            sel_list = list(cur)
            m = len(sel_list)
            ok = True
            # (1) x: 3-sums of cur
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = sel_list[i] + sel_list[j]
                    for k in range(j + 1, m):
                        if (s2 + sel_list[k]) % x == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                # (2) for each a in cur: triples with x
                for a in sel_list:
                    for i in range(m):
                        if sel_list[i] == a:
                            continue
                        for j in range(i + 1, m):
                            if sel_list[j] == a:
                                continue
                            if (x + sel_list[i] + sel_list[j]) % a == 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
            if ok:
                cur.add(x)
                if len(cur) > best_size:
                    best = set(cur)
                    best_size = len(best)
                improved = True
                break
        if not improved:
            # Remove a random element from cur to allow re-adding
            if cur:
                rem = rng.choice(list(cur))
                cur.discard(rem)
    return best


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(42)
    sums, idxs = _build_tables()

    best = set()
    best_size = 0

    # Multiple random-order greedy builds + local search
    while time.time() < deadline:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        # Bias toward larger numbers (they're less likely to divide sums)
        order.sort(key=lambda v: (-v, rng.random()))
        cur = _greedy_build(order, sums, idxs, deadline)
        if len(cur) > best_size:
            best = set(cur)
            best_size = len(best)
        # Local search
        cur2 = _local_search(cur, sums, idxs, deadline, rng, max_iters=5000)
        if len(cur2) > best_size:
            best = set(cur2)
            best_size = len(best)
        if best_size >= MAX_EL:
            break

    # Final validation
    result = sorted(best)
    if result and not _is_valid_set(set(result), sums, idxs):
        # Fallback: rebuild greedily in descending order
        result = []
        for x in range(N, 0, -1):
            cand = set(result + [x])
            if _is_valid_set(cand, sums, idxs):
                result.append(x)
    return result
