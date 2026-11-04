"""Difference triangle set (7,5): construct n_rows rows of k+1 increasing ints from 0
with all positive intra-row differences distinct across rows; minimize the max entry."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s wall-clock

    # Fast greedy construction (same as seed) to get an initial valid solution.
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in best_rows)

    # Local search: try to reduce the largest entry by repositioning elements.
    # Strategy: for the row with the largest max, try to move its last element
    # to a smaller position, or swap elements between rows to free up space.

    rng = random.Random(42)

    def all_diffs(rows):
        """Return the set of all positive intra-row differences."""
        d = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d.add(row[j] - row[i])
        return d

    def is_valid(rows):
        """Check that all positive intra-row differences are distinct."""
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def row_diffs(row):
        """Return set of positive differences within a single row."""
        n = len(row)
        s = set()
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    # Iterative improvement loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False

        # Find the row with the largest max entry
        max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        max_val = max(rows[max_row_idx])

        # Try to reduce the last element of the worst row
        row = rows[max_row_idx]
        last = row[-1]
        # Try moving last to a smaller value
        for new_last in range(last - 1, 0, -1):
            if new_last <= row[-2]:
                break
            candidate = row[:-1] + [new_last]
            # Check strict increase
            if any(candidate[i] >= candidate[i+1] for i in range(len(candidate)-1)):
                continue
            # Check differences
            cand_diffs = row_diffs(candidate)
            # Check against all other rows' differences
            ok = True
            for i in range(n_rows):
                if i == max_row_idx:
                    continue
                if cand_diffs & row_diffs(rows[i]):
                    ok = False
                    break
            if ok:
                rows[max_row_idx] = candidate
                best_rows = [r[:] for r in rows]
                best_max = max(max(r) for r in best_rows)
                improved = True
                break

        if improved:
            continue

        # Try repositioning interior elements of the worst row
        for pos in range(1, len(row) - 1):
            old_val = row[pos]
            for new_val in range(old_val - 1, row[pos - 1], -1):
                if new_val <= row[pos - 1]:
                    break
                candidate = row[:pos] + [new_val] + row[pos+1:]
                if any(candidate[i] >= candidate[i+1] for i in range(len(candidate)-1)):
                    continue
                cand_diffs = row_diffs(candidate)
                ok = True
                for i in range(n_rows):
                    if i == max_row_idx:
                        continue
                    if cand_diffs & row_diffs(rows[i]):
                        ok = False
                        break
                if ok:
                    rows[max_row_idx] = candidate
                    best_rows = [r[:] for r in rows]
                    best_max = max(max(r) for r in best_rows)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # Cross-row swaps: try swapping a non-zero element between two rows
        # to potentially allow a smaller max elsewhere
        for i in range(n_rows):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, n_rows):
                for pi in range(1, len(rows[i])):
                    for pj in range(1, len(rows[j])):
                        vi = rows[i][pi]
                        vj = rows[j][pj]
                        if vi == vj:
                            continue
                        # Swap
                        rows[i][pi], rows[j][pj] = vj, vi
                        # Check both rows still strictly increasing
                        if any(rows[i][a] >= rows[i][a+1] for a in range(len(rows[i])-1)):
                            rows[i][pi], rows[j][pj] = vi, vj
                            continue
                        if any(rows[j][a] >= rows[j][a+1] for a in range(len(rows[j])-1)):
                            rows[i][pi], rows[j][pj] = vi, vj
                            continue
                        # Check validity
                        di = row_diffs(rows[i])
                        dj = row_diffs(rows[j])
                        if di & dj:
                            rows[i][pi], rows[j][pj] = vi, vj
                            continue
                        ok = True
                        for r in range(n_rows):
                            if r == i or r == j:
                                continue
                            dr = row_diffs(rows[r])
                            if (di & dr) or (dj & dr):
                                ok = False
                                break
                        if not ok:
                            rows[i][pi], rows[j][pj] = vi, vj
                            continue
                        # Check if this reduces the overall max
                        new_max = max(max(rows[r]) for r in range(n_rows))
                        if new_max < best_max:
                            best_rows = [r[:] for r in rows]
                            best_max = new_max
                            improved = True
                            break
                        else:
                            rows[i][pi], rows[j][pj] = vi, vj
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_rows

# EVOLVE-BLOCK-END
