import random
import time

N = 256


def _can_add(x, sel, sums):
    """Check if x can be added to sel (Sidon property preserved)."""
    # x must not be in sel
    for a in sel:
        if a == x:
            return False
    # Check all pairs (x, a) and (x, x)
    for a in sel:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True


def _compute_sums(sel):
    """Compute set of all pairwise sums a+b with a<=b."""
    sums = set()
    n = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i, n):
            sums.add(a + sel[j])
    return sums


def _is_maximal(sel, sums):
    """Check if sel is inclusion-maximal in [1, N]."""
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        ok = True
        for a in sel:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            return False
    return True


def _greedy_construct(order):
    """Greedy Sidon construction following given order."""
    sel = []
    sums = set()
    for x in order:
        if _can_add(x, sel, sums):
            sel.append(x)
            for a in sel:
                sums.add(x + a)
            sums.add(2 * x)
    return sel, sums


def _try_remove(sel, sums):
    """Try to remove an element while maintaining maximality. Returns (new_sel, new_sums, removed) or None."""
    n = len(sel)
    sel_set = set(sel)
    for idx in range(n):
        x = sel[idx]
        # Try removing x
        new_sel = [a for i, a in enumerate(sel) if i != idx]
        new_sel_set = set(new_sel)
        new_sums = _compute_sums(new_sel)
        # Check maximality of new_sel
        maximal = True
        for y in range(1, N + 1):
            if y in new_sel_set:
                continue
            # Check if y can be added to new_sel
            can = True
            for a in new_sel:
                if (y + a) in new_sums:
                    can = False
                    break
            if can and (2 * y) in new_sums:
                can = False
            if can:
                maximal = False
                break
        if maximal:
            return new_sel, new_sums, x
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    best_sel = None
    best_size = N + 1

    # Phase 1: Multiple greedy constructions with different orderings
    for trial in range(2000):
        if time.time() > deadline:
            break

        if trial == 0:
            order = list(range(1, N + 1))
        elif trial == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)

        sel, sums = _greedy_construct(order)

        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            best_sums = sums

        # Phase 2: Local search - try to remove elements
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for attempt in range(50):
                    if time.time() > deadline:
                        break
                    result = _try_remove(sel, sums)
                    if result is not None:
                        sel, sums, _ = result
                        improved = True
                        break

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    # Also try: greedy with random skips (sometimes skipping a candidate leads to smaller maximal set)
    for trial in range(3000):
        if time.time() > deadline:
            break

        order = list(range(1, N + 1))
        random.shuffle(order)

        sel = []
        sums = set()
        for x in order:
            # Randomly skip with small probability
            if random.random() < 0.05:
                continue
            if _can_add(x, sel, sums):
                sel.append(x)
                for a in sel:
                    sums.add(x + a)
                sums.add(2 * x)

        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]
            best_sums = sums

        # Local search
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for attempt in range(30):
                    if time.time() > deadline:
                        break
                    result = _try_remove(sel, sums)
                    if result is not None:
                        sel, sums, _ = result
                        improved = True
                        break

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    if best_sel is None:
        # Fallback: simple ascending greedy
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(x, sel, sums):
                sel.append(x)
                for a in sel:
                    sums.add(x + a)
                sums.add(2 * x)
        best_sel = sel

    return best_sel
