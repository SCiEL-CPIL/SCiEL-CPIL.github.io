import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    # 1. Greedy baseline
    best = greedy_construct(n_rows, k)
    if not best:
        return []
    best_scope = max(max(row) for row in best)

    # Counting lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs

    if best_scope <= lower_bound:
        return best

    # 2. Iterative scope reduction via strong backtracking
    target = best_scope - 1
    while target >= lower_bound and time.time() < deadline:
        # Try several randomized restarts at this target
        found = None
        # Budget per target: stop trying this target if too slow
        target_start = time.time()
        restarts = 0
        while time.time() < deadline and (target_start + 1.0) * 0 + time.time() - target_start < 6.0:
            found = _search(n_rows, k, target, deadline, random.seed())
            restarts += 1
            if found is not None:
                break
        if found is not None:
            s = max(max(row) for row in found)
            if s < best_scope:
                best = found
                best_scope = s
                if best_scope <= lower_bound:
                    break
            target = best_scope - 1
        else:
            target -= 1

    # 3. Polish: try to compact the best solution found
    if time.time() < deadline - 5.0:
        polished = _polish(best, n_rows, k, deadline)
        if polished is not None:
            s = max(max(row) for row in polished)
            if s < best_scope:
                best = polished
                best_scope = s

    return best


def greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            mask = 0
            for a in row:
                mask |= (1 << (x - a))
            if mask & used:
                continue
            row.append(x)
            used |= mask
        rows.append(row)
    return rows


def _search(n_rows, k, max_scope, deadline, seed):
    """Strong backtracking: build n_rows rows, all diffs distinct, max entry <= max_scope.
    Uses bitset for used diffs. Forward-checking: before committing a candidate, verify
    that a completion of the current row is still possible (greedy feasibility)."""
    rng = random.Random(seed)
    used = 0
    rows = []

    def build_row(pos, last, row):
        # pos = number of elements already in row (row[0]=0 always)
        if pos == k + 1:
            return row
        # Try to complete the rest of the row greedily to check feasibility
        # (this is a necessary condition, used for pruning)
        candidates = []
        for x in range(last + 1, max_scope + 1):
            mask = 0
            for a in row:
                mask |= (1 << (x - a))
            if mask & used:
                continue
            candidates.append((x, mask))
            if len(candidates) > 60:
                break
        if not candidates:
            return None

        # Order: try smallest first (compactness), but with some randomization
        # among the first ~12 to escape local minima.
        if len(candidates) > 12:
            head = candidates[:12]
            rng.shuffle(head)
            ordered = head + candidates[12:]
        else:
            ordered = candidates
            rng.shuffle(ordered)

        for x, mask in ordered:
            used |= mask
            row.append(x)
            res = build_row(pos + 1, x, row)
            if res is not None:
                return res
            row.pop()
            used ^= mask
        return None

    for _ in range(n_rows):
        if time.time() > deadline:
            return None
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
    return rows


def _polish(solution, n_rows, k, deadline):
    """Try to reduce the scope of a valid solution by locally lowering entries."""
    best = [list(r) for r in solution]
    best_scope = max(max(r) for r in best)
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for ri in range(n_rows):
            for pi in range(k, 0, -1):  # from largest index backwards
                if time.time() > deadline - 3.0:
                    break
                cur = best[ri][pi]
                for newv in range(best[ri][pi - 1] + 1, cur):
                    # Check if replacing cur with newv keeps all diffs distinct
                    row = best[ri][:pi] + [newv] + best[ri][pi + 1:]
                    if _valid_row_in_context(row, best, ri):
                        best[ri][pi] = newv
                        improved = True
                        break
    s = max(max(r) for r in best)
    if s < best_scope:
        return best
    return None


def _valid_row_in_context(row, all_rows, row_idx):
    """Check that `row` (with all_rows[row_idx] replaced by row) keeps global diff uniqueness."""
    seen = 0
    for i, r in enumerate(all_rows):
        rr = row if i == row_idx else r
        for j in range(len(rr)):
            for jj in range(j + 1, len(rr)):
                d = rr[jj] - rr[j]
                bit = 1 << d
                if seen & bit:
                    return False
                seen |= bit
    return True
