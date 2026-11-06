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


def _backtracking_construct():
    """
    Backtracking/greedy construction that at each step picks the available
    element that blocks the most other candidates.  This forces early
    saturation and tends to yield smaller maximal sets.
    """
    sel = []
    sums = set()
    sel_set = set()
    # candidates: set of elements not yet decided
    candidates = set(range(1, N + 1))

    while candidates:
        # For each candidate, count how many OTHER candidates it would block
        # if added.  Pick the one that blocks the most.
        best_x = None
        best_blocks = -1
        for x in candidates:
            # Check if x can be added
            can = True
            for a in sel:
                if (x + a) in sums:
                    can = False
                    break
            if not can:
                continue
            if (2 * x) in sums:
                continue
            # Count how many other candidates y would be blocked by adding x
            # y is blocked if (x+y) is already in sums OR (x+y) equals some
            # existing sum, OR (2*y) is in sums after adding x (i.e., 2*y == x+y => y==x, skip)
            # More precisely: after adding x, new sums include x+a for all a in sel∪{x}.
            # y is blocked if for some a in sel∪{x}, (y+a) is in new_sums, or 2*y in new_sums.
            # new_sums = sums ∪ {x+a : a in sel} ∪ {2*x}
            blocks = 0
            for y in candidates:
                if y == x:
                    continue
                # Check if y would be blocked after adding x
                blocked = False
                for a in sel:
                    if (y + a) in sums:
                        blocked = True
                        break
                    # New sum x+a: check if y+a == x+a => y==x (already excluded)
                    # New sum x+a: check if x+a is in sums? No, we check if (y+a) equals any new sum
                    # Actually: y is blocked if (y+a) is in new_sums for some a in sel∪{x}
                    # (y+a) in sums → already checked
                    # (y+a) == x+a → y==x, excluded
                    # (y+a) == 2*x → y+a = 2x → y = 2x-a
                    # (y+x) in sums → check
                    # (y+x) == x+a → y==a, check if a in sel
                    # (y+x) == 2*x → y==x, excluded
                    if (y + a) == 2 * x:
                        blocked = True
                        break
                if not blocked:
                    # Check (y + x) against existing sums
                    if (y + x) in sums:
                        blocked = True
                    elif (y + x) == 2 * x:
                        blocked = True  # y == x, already excluded
                    # Check 2*y against new sums
                    if not blocked:
                        if (2 * y) in sums:
                            blocked = True
                        elif (2 * y) == 2 * x:
                            blocked = True  # y == x, excluded
                        elif any((2 * y) == x + a for a in sel):
                            blocked = True
                if blocked:
                    blocks += 1

            if blocks > best_blocks:
                best_blocks = blocks
                best_x = x

        if best_x is None:
            # No candidate can be added; we're done
            break

        # Add best_x
        sel.append(best_x)
        sel_set.add(best_x)
        candidates.discard(best_x)
        for a in sel:
            sums.add(best_x + a)
        sums.add(2 * best_x)

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

    # Phase 0: Backtracking construction with blocking priority
    for trial in range(50):
        if time.time() > deadline:
            break
        sel, sums = _backtracking_construct()
        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = sel[:]

        # Local search: try to remove elements
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                result = _try_remove(sel, sums)
                if result is not None:
                    sel, sums, _ = result
                    improved = True

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]

    # Phase 1: Multiple greedy constructions with different orderings
    for trial in range(1000):
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

        # Local search - try to remove elements
        if len(sel) < best_size:
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                result = _try_remove(sel, sums)
                if result is not None:
                    sel, sums, _ = result
                    improved = True

            if len(sel) < best_size:
                best_size = len(sel)
                best_sel = sel[:]
                best_sums = sums

    # Phase 2: Greedy with random skips
    for trial in range(2000):
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
                result = _try_remove(sel, sums)
                if result is not None:
                    sel, sums, _ = result
                    improved = True

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
