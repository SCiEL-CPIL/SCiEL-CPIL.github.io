"""Colorings with three colors in every four-term progression; local-search design."""
import time
import random

# Precompute all nonconstant 4-term progressions within positions 1..128.
# A 4-AP is (a, a+d, a+2d, a+3d) with d>=1, a+3d<=128.
# Each AP is stored as a tuple of 4 0-based indices.
_ALL_APS = []
for d in range(1, 43):           # max d: 1+3d<=128 -> d<=42
    max_a = 128 - 3 * d
    for a in range(1, max_a + 1):
        _ALL_APS.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
_NUM_APS = len(_ALL_APS)

# For each position i, list of AP indices that include i.
_POS_TO_APS = [[] for _ in range(128)]
for _ai, _ap in enumerate(_ALL_APS):
    for _i in _ap:
        _POS_TO_APS[_i].append(_ai)


def _ap_violates(colors, ap):
    # A 4-AP is violated if it uses fewer than 3 distinct colors.
    # Count distinct colors among the 4 positions.
    c0 = colors[ap[0]]
    c1 = colors[ap[1]]
    c2 = colors[ap[2]]
    c3 = colors[ap[3]]
    # distinct count
    distinct = 1
    if c1 != c0:
        distinct += 1
    if c2 != c0 and c2 != c1:
        distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2:
        distinct += 1
    return distinct < 3


def _count_violations(colors):
    v = 0
    for ap in _ALL_APS:
        if _ap_violates(colors, ap):
            v += 1
    return v


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill

    best_colors = None
    best_viol = None
    best_used = None

    rng = random.Random(12345)

    # We will try to use a small number of colors.
    # Strategy: local search minimizing (violations, number_of_colors_used).
    # Start with a random assignment using k colors and hill-climb.

    # Try a range of color counts; smaller is better if feasible.
    # We'll run multiple trials; each trial uses a fixed palette size k.
    # Objective: minimize violations first (must reach 0), then minimize k.

    # We'll search over k from 2 upward, but also allow the search to
    # naturally reduce the number of distinct colors used.

    # Simpler: use a palette of size K, but track how many are actually used.
    # Primary goal: reach 0 violations. Secondary: minimize distinct used.

    # We'll do a multi-start local search. Each start: random coloring with
    # a small palette, then local search to reduce violations, then a
    # "color consolidation" phase to reduce distinct colors used.

    # Precompute for speed: for each AP, we need distinct colors.
    # We'll maintain a violation count incrementally.

    def make_initial(k):
        return [rng.randrange(k) for _ in range(128)]

    def local_search(colors, k, max_steps):
        # Incremental violation tracking.
        viol = [False] * _NUM_APS
        total_viol = 0
        for ai, ap in enumerate(_ALL_APS):
            if _ap_violates(colors, ap):
                viol[ai] = True
                total_viol += 1

        # Local search: pick a random position, try a random color, accept if
        # it reduces total_viol (or ties with some heuristic).
        steps = 0
        stuck = 0
        while total_viol > 0 and steps < max_steps and time.time() < deadline - 5.0:
            steps += 1
            i = rng.randrange(128)
            old = colors[i]
            new = rng.randrange(k)
            if new == old:
                continue
            # Compute delta in violations.
            delta = 0
            for ai in _POS_TO_APS[i]:
                ap = _ALL_APS[ai]
                was_viol = viol[ai]
                # simulate
                colors[i] = new
                now_viol = _ap_violates(colors, ap)
                colors[i] = old
                if was_viol and not now_viol:
                    delta -= 1
                elif (not was_viol) and now_viol:
                    delta += 1
            if delta < 0:
                # accept
                colors[i] = new
                for ai in _POS_TO_APS[i]:
                    ap = _ALL_APS[ai]
                    was_viol = viol[ai]
                    now_viol = _ap_violates(colors, ap)
                    viol[ai] = now_viol
                    if was_viol and not now_viol:
                        total_viol -= 1
                    elif (not was_viol) and now_viol:
                        total_viol += 1
                stuck = 0
            else:
                # random walk to escape local minima
                if stuck > 50 or rng.random() < 0.05:
                    colors[i] = new
                    for ai in _POS_TO_APS[i]:
                        ap = _ALL_APS[ai]
                        was_viol = viol[ai]
                        now_viol = _ap_violates(colors, ap)
                        viol[ai] = now_viol
                        if was_viol and not now_viol:
                            total_viol -= 1
                        elif (not was_viol) and now_viol:
                            total_viol += 1
                    stuck += 1
                else:
                    stuck += 1
        return colors, total_viol

    def consolidate(colors, k, max_steps):
        # Try to reduce the number of distinct colors used.
        # Repeatedly try to map one color to another, accepting if violations
        # don't increase (ideally decrease or stay 0).
        # We'll track violations incrementally.
        viol = [False] * _NUM_APS
        total_viol = 0
        for ai, ap in enumerate(_ALL_APS):
            if _ap_violates(colors, ap):
                viol[ai] = True
                total_viol += 1

        # Find the color that is used the fewest times; try to eliminate it.
        for _round in range(20):
            if time.time() > deadline - 5.0:
                break
            counts = [0] * k
            for c in colors:
                counts[c] += 1
            # pick the least-used color that is still in use
            candidates = [c for c in range(k) if counts[c] > 0]
            if len(candidates) <= 1:
                break
            # sort by count ascending
            candidates.sort(key=lambda c: counts[c])
            src = candidates[0]
            # try to map src -> dst for each dst != src
            improved = False
            # shuffle dst order
            dsts = [c for c in range(k) if c != src]
            rng.shuffle(dsts)
            for dst in dsts:
                # simulate mapping all src -> dst
                # We need to check if this increases violations.
                # Quick check: for each position with color src, changing to dst.
                # We'll compute delta by iterating over affected APs.
                positions_src = [i for i in range(128) if colors[i] == src]
                if not positions_src:
                    continue
                # Compute total delta
                delta = 0
                # We need to consider all APs that include any src position.
                # To avoid double counting, collect affected AP indices.
                affected = set()
                for i in positions_src:
                    for ai in _POS_TO_APS[i]:
                        affected.add(ai)
                # simulate
                for i in positions_src:
                    colors[i] = dst
                for ai in affected:
                    ap = _ALL_APS[ai]
                    was_viol = viol[ai]
                    now_viol = _ap_violates(colors, ap)
                    if was_viol and not now_viol:
                        delta -= 1
                    elif (not was_viol) and now_viol:
                        delta += 1
                # revert
                for i in positions_src:
                    colors[i] = src
                if delta <= 0:
                    # accept
                    for i in positions_src:
                        colors[i] = dst
                    for ai in affected:
                        ap = _ALL_APS[ai]
                        was_viol = viol[ai]
                        now_viol = _ap_violates(colors, ap)
                        viol[ai] = now_viol
                        if was_viol and not now_viol:
                            total_viol -= 1
                        elif (not was_viol) and now_viol:
                            total_viol += 1
                    improved = True
                    break
            if not improved:
                break
        return colors, total_viol

    # Main loop: try different palette sizes and restarts.
    # We'll iterate: for k in [2,3,4,...], do several local-search trials.
    # Keep the best solution (min violations, then min distinct used).

    # Actually, let's be smarter: start with a larger palette to find a
    # 0-violation solution, then consolidate to reduce colors.
    # But we also want to directly find low-color solutions.

    # Let's do: for k in range(2, 8):
    #   for trial in range(trials_per_k):
    #     colors = make_initial(k)
    #     colors, viol = local_search(colors, k, max_steps)
    #     colors, viol = consolidate(colors, k, max_steps)
    #     used = len(set(colors))
    #     if viol == 0 and (best_viol is None or used < best_used or (used==best_used)):
    #        update best
    #   if best_viol == 0 and best_used == k: break  # can't do better with this k

    # But we also need to handle the case where k=2 is infeasible; then k=3, etc.

    # Let's set a reasonable budget.
    trials_per_k = 15
    max_steps_ls = 20000

    for k in range(2, 9):
        if time.time() > deadline - 10.0:
            break
        for trial in range(trials_per_k):
            if time.time() > deadline - 5.0:
                break
            colors = make_initial(k)
            colors, viol = local_search(colors, k, max_steps_ls)
            colors, viol = consolidate(colors, k, 1000)
            used = len(set(colors))
            # Score: (viol, used)
            if best_viol is None or viol < best_viol or (viol == best_viol and used < best_used):
                best_viol = viol
                best_used = used
                best_colors = colors[:]
                # If we found a 0-violation solution with k colors, we can stop
                # trying larger k (but smaller k might have been missed; we go
                # in order so smaller k already tried).
                if viol == 0 and used == k:
                    # Can't improve with larger k (would use >=k colors)
                    break
        if best_viol == 0 and best_used == k:
            break

    # If we never found a 0-violation solution, return the best we have.
    # But the task requires a valid coloring. The evaluator scores 0 for invalid.
    # So we must ensure we return a valid coloring if possible.
    # If best_viol > 0, we need to do more work. But given time, let's return
    # the best; if invalid, score is 0. To be safe, we should try harder.

    # Actually, let's add a fallback: if best_viol > 0, do a more aggressive
    # search with a larger palette to guarantee validity.
    if best_viol is None or best_viol > 0:
        # Try with k=4, more trials, more steps
        for k in range(3, 8):
            if time.time() > deadline - 5.0:
                break
            for trial in range(30):
                if time.time() > deadline - 2.0:
                    break
                colors = make_initial(k)
                colors, viol = local_search(colors, k, 50000)
                if viol == 0:
                    used = len(set(colors))
                    if best_viol is None or viol < best_viol or (viol == best_viol and used < best_used):
                        best_viol = viol
                        best_used = used
                        best_colors = colors[:]
                    break
            if best_viol == 0:
                break

    # Final: ensure we return something. If still invalid, return as-is (score 0).
    if best_colors is None:
        # Fallback: use the greedy seed approach as a last resort
        colors = []
        for x in range(1, 129):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 3 * d < 1:
                        break
                    idx = x - 3 * d - 1
                    if idx >= 0 and idx < len(colors):
                        s = {colors[idx], colors[idx + d], colors[idx + 2 * d], color}
                        if len(s) < 3:
                            ok = False
                            break
                if ok:
                    colors.append(color)
                    break
        return colors

    return best_colors
