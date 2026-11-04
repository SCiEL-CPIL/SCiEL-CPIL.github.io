import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Candidate pool size for the greedy selection. Larger = more lookahead,
    # but each candidate costs O(k) diff checks.
    c = 16

    seed = 0
    while time.time() < deadline - 5.0:
        rows = _greedy_degree(n_rows, k, seed, c)
        if rows is None:
            seed += 1
            continue

        # Polish: try to reduce the global max by re-seating single elements.
        rows, scope = _polish(rows, k, deadline - 2.0)

        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            if best_scope == lower_bound:
                break

        seed += 1

    if best_rows is None:
        # Deterministic fallback: pure smallest-first greedy.
        best_rows = _simple_greedy(n_rows, k)

    return best_rows


def _greedy_degree(n_rows, k, seed, c):
    """Greedy row construction where each next element is chosen by a
    'degree' score: prefer candidates that leave the most remaining valid
    difference slots for the rest of the search (min-conflicts surrogate).
    A small amount of randomness (seeded) breaks ties / adds diversity."""
    rng = random.Random(seed)
    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            # Dynamic search window: enough room to find c valid candidates.
            limit = 600
            scored = []  # (score, x)
            x = last_x + 1
            while len(scored) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # Degree: how many of these new differences are "small"
                    # (small diffs are the scarce resource near the bottom of
                    # the difference range, so consuming them is costly).
                    # We penalize candidates that use small diffs more.
                    score = sum(1.0 / d for d in new_diffs)
                    # Add a tiny random jitter so ties are broken by seed.
                    score += rng.random() * 0.01
                    scored.append((score, x))
                x += 1

            if not scored:
                # Expand window (rare).
                x = last_x + 1
                while not scored:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        score = sum(1.0 / d for d in new_diffs)
                        scored.append((score, x))
                    x += 1

            # Pick the candidate with the smallest score (leaves most slots).
            scored.sort(key=lambda t: t[0])
            chosen_x = scored[0][1]

            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)

        rows.append(row)

    return rows


def _polish(rows, k, deadline):
    """Local search: repeatedly try to move a single element to a smaller
    value that keeps the whole set valid, accepting only moves that reduce
    the global max. This is the refinement machinery from the inspiration."""
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)

    def _valid_diffs(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in all_d:
                        return None
                    all_d.add(d)
        return all_d

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find the row holding the current max.
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        row = best[max_row_idx]
        # Try to move elements from the back (largest positions) to smaller values.
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline:
                return best, best_scope
            old_val = row[pos]
            for new_val in range(row[pos - 1] + 1, old_val):
                row[pos] = new_val
                # Quick validity check.
                all_d = set()
                ok = True
                for r in best:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in all_d:
                                ok = False
                                break
                            all_d.add(d)
                    if not ok:
                        break
                if ok:
                    new_scope = max(max(r) for r in best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                row[pos] = old_val
            if improved:
                break

    return best, best_scope


def _simple_greedy(n_rows, k):
    """Deterministic smallest-first greedy (fallback)."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
        rows.append(row)
    return rows
