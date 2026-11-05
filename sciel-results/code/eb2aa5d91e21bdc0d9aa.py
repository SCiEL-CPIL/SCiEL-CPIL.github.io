"""Difference triangle set (4,7): local search to minimize the largest entry."""
import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135.0  # safety margin under the 150s wall-clock
    rng = random.Random(12345)

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])

    best = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]

    def rebuild_used():
        u = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    u.add(row[j] - row[i])
        return u

    def try_swap(r, pos, newval):
        row = rows[r]
        old = row[pos]
        if newval == old:
            return False
        if pos > 0 and newval <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and newval >= row[pos + 1]:
            return False
        # differences that involve position pos
        remove = set()
        add = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                remove.add(old - row[i])
                add.add(newval - row[i])
            else:
                remove.add(row[i] - old)
                add.add(row[i] - newval)
        # must not collide with any currently-used difference not being removed
        if (add & (used - remove)):
            return False
        # apply
        row[pos] = newval
        used -= remove
        used |= add
        return True

    # Phase 1: try to reduce the max entry by swapping out the current max mark
    while time.time() < deadline - 10.0:
        progress = False
        # find positions of the global max
        maxval = max(max(r) for r in rows)
        targets = []
        for r in range(n_rows):
            row = rows[r]
            for pos in range(1, len(row)):
                if row[pos] == maxval:
                    targets.append((r, pos))
        rng.shuffle(targets)
        for (r, pos) in targets:
            if time.time() > deadline - 10.0:
                break
            row = rows[r]
            old = row[pos]
            # candidate new values: smaller than old, respect ordering
            lo = row[pos - 1] + 1 if pos > 0 else 1
            hi = (row[pos + 1] - 1) if pos < len(row) - 1 else old - 1
            # try to find a newval in [lo, hi] that reduces max
            # prefer values that keep the new max small; just scan downward from hi
            found = False
            for newval in range(hi, lo - 1, -1):
                if try_swap(r, pos, newval):
                    found = True
                    progress = True
                    break
            if found:
                m = max(max(rr) for rr in rows)
                if m < best:
                    best = m
                    best_rows = [rr[:] for rr in rows]
        if not progress:
            # no max-reducing swap found; try to reduce a non-max large mark to open space
            # pick the largest non-max mark and try to lower it
            candidates = []
            for r in range(n_rows):
                row = rows[r]
                for pos in range(1, len(row)):
                    if row[pos] < maxval:
                        candidates.append((row[pos], r, pos))
            candidates.sort(reverse=True)
            tried = 0
            for (v, r, pos) in candidates:
                if tried >= 200 or time.time() > deadline - 10.0:
                    break
                tried += 1
                row = rows[r]
                old = row[pos]
                lo = row[pos - 1] + 1 if pos > 0 else 1
                hi = (row[pos + 1] - 1) if pos < len(row) - 1 else old - 1
                # try a random lower value
                if hi >= lo:
                    newval = rng.randint(lo, hi)
                    if try_swap(r, pos, newval):
                        m = max(max(rr) for rr in rows)
                        if m < best:
                            best = m
                            best_rows = [rr[:] for rr in rows]
                        # revert to keep structure stable? keep it (neutral move)
            # occasionally rebuild a row to escape
            if rng.random() < 0.1 and time.time() < deadline - 15.0:
                r = rng.randrange(n_rows)
                # remove row r from used, rebuild it greedily
                row = rows[r]
                remove = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        remove.add(row[j] - row[i])
                used -= remove
                newrow = [0]
                x = 0
                while len(newrow) < k + 1:
                    x += 1
                    nd = {x - a for a in newrow}
                    if not (nd & used):
                        newrow.append(x)
                        used |= nd
                rows[r] = newrow
                m = max(max(rr) for rr in rows)
                if m < best:
                    best = m
                    best_rows = [rr[:] for rr in rows]

    return best_rows
