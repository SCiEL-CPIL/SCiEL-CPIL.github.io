"""Unified simulated-annealing search for difference-triangle sets.

Instead of building rows independently with greedy heuristics and then compacting,
we treat the whole collection of rows as a single energy landscape. The energy
function penalises duplicate differences (the hard constraint) and also penalises
the scope (the optimisation objective). Moves are random perturbations of
individual elements (shift by ±1, ±2, or swap two elements within a row).
Simulated annealing allows the search to escape local minima that per-row
greedy + local compaction cannot.
"""

import time
import random
import sys
import math


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    rng = random.Random(42)

    # ---- Build initial solution with plain greedy ----
    def greedy_init():
        u = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    def compute_energy(rows):
        """Return (num_duplicates, scope)."""
        diff_count = {}
        for row in rows:
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        num_dup = sum(c - 1 for c in diff_count.values() if c > 1)
        scope = max(max(r) for r in rows)
        return num_dup, scope

    def make_move(rows):
        """Apply a random move. Returns (rows, old_val, pos, r_idx) for undo, or None if no valid move."""
        r_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, m)  # never move the first element (stays 0)
        row = rows[r_idx]

        move_type = rng.random()
        if move_type < 0.6:
            # Shift by small amount
            delta = rng.choice([-2, -1, 1, 2])
            new_val = row[pos] + delta
            # Check bounds: must be > row[pos-1] and < row[pos+1] if pos+1 < m
            if pos > 0 and new_val <= row[pos - 1]:
                return None
            if pos + 1 < m and new_val >= row[pos + 1]:
                return None
            if new_val <= 0:
                return None
            rows[r_idx][pos] = new_val
            return (r_idx, pos, row[pos], delta)
        elif move_type < 0.8:
            # Swap two elements within the row (positions 1..m-1)
            pos2 = rng.randrange(1, m)
            if pos2 == pos:
                return None
            # Ensure order is preserved after swap
            # We need to check that after swap, the row is still strictly increasing
            # This is complex; skip for now
            return None
        else:
            # Shift by larger amount
            delta = rng.choice([-3, -4, -5, 3, 4, 5])
            new_val = row[pos] + delta
            if pos > 0 and new_val <= row[pos - 1]:
                return None
            if pos + 1 < m and new_val >= row[pos + 1]:
                return None
            if new_val <= 0:
                return None
            rows[r_idx][pos] = new_val
            return (r_idx, pos, row[pos], delta)

    def undo_move(rows, move_info):
        r_idx, pos, old_val, delta = move_info
        rows[r_idx][pos] = old_val

    # ---- Local compaction (fast, deterministic) ----
    def compact(rows):
        """Try to reduce each element from right to left by 1 if feasible."""
        improved = True
        while improved:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for pos in range(m - 1, 0, -1):
                    if row[pos] <= row[pos - 1] + 1:
                        continue
                    new_val = row[pos] - 1
                    # Check feasibility: compute all diffs before and after
                    # Quick check: just verify no duplicates introduced
                    # Build diff set excluding this element's contributions
                    all_diffs = set()
                    for i in range(n_rows):
                        r = rows[i]
                        for a in range(m):
                            for b in range(a + 1, m):
                                if i == r_idx and (a == pos or b == pos):
                                    continue
                                all_diffs.add(r[b] - r[a])
                    # Check new diffs
                    ok = True
                    for a in range(m):
                        if a == pos:
                            continue
                        d = new_val - row[a] if a < pos else row[a] - new_val
                        if d in all_diffs:
                            ok = False
                            break
                        all_diffs.add(d)
                    if ok:
                        row[pos] = new_val
                        improved = True
        return rows

    # ---- Initial solution ----
    best_rows = greedy_init()
    best_rows = compact(best_rows)
    best_dup, best_scope = compute_energy(best_rows)
    best_energy = best_dup * 10000 + best_scope

    # ---- Simulated annealing ----
    t_start = time.time()
    T = 50.0
    T_min = 0.1
    cooling = 0.9997  # slow cooling

    current_rows = [list(r) for r in best_rows]
    cur_dup, cur_scope = compute_energy(current_rows)
    cur_energy = cur_dup * 10000 + cur_scope

    steps = 0
    stuck_counter = 0

    while time.time() < deadline:
        move = make_move(current_rows)
        if move is None:
            stuck_counter += 1
            if stuck_counter > 200:
                # Restart from best
                current_rows = [list(r) for r in best_rows]
                cur_dup, cur_scope = compute_energy(current_rows)
                cur_energy = cur_dup * 10000 + cur_scope
                stuck_counter = 0
                T = max(T, 10.0)
            continue
        stuck_counter = 0

        new_dup, new_scope = compute_energy(current_rows)
        new_energy = new_dup * 10000 + new_scope
        delta_e = new_energy - cur_energy

        if delta_e <= 0 or (T > T_min and rng.random() < math.exp(-delta_e / T)):
            cur_dup, cur_scope = new_dup, new_scope
            cur_energy = new_energy
            # Accept
            if cur_energy < best_energy:
                best_dup, best_scope = cur_dup, cur_scope
                best_energy = cur_energy
                best_rows = [list(r) for r in current_rows]
                # Try compaction on the new best
                compacted = [list(r) for r in best_rows]
                compact(compacted)
                c_dup, c_scope = compute_energy(compacted)
                c_energy = c_dup * 10000 + c_scope
                if c_energy < best_energy:
                    best_dup, best_scope = c_dup, c_scope
                    best_energy = c_energy
                    best_rows = compacted
                    current_rows = [list(r) for r in best_rows]
                    cur_dup, cur_scope = c_dup, c_scope
                    cur_energy = c_energy
        else:
            undo_move(current_rows, move)

        T *= cooling
        if T < T_min:
            T = T_min

        steps += 1
        if steps % 5000 == 0:
            # Periodic compaction of current state
            compact(current_rows)
            cur_dup, cur_scope = compute_energy(current_rows)
            cur_energy = cur_dup * 10000 + cur_scope
            if cur_energy < best_energy:
                best_dup, best_scope = cur_dup, cur_scope
                best_energy = cur_energy
                best_rows = [list(r) for r in current_rows]

    # Final compaction
    final_rows = [list(r) for r in best_rows]
    compact(final_rows)
    f_dup, f_scope = compute_energy(final_rows)
    f_energy = f_dup * 10000 + f_scope
    if f_energy < best_energy:
        best_rows = final_rows
        best_scope = f_scope

    # Validate: if still duplicates, fall back to greedy
    if best_dup > 0:
        # Try a few more greedy restarts
        for seed in range(10):
            rng2 = random.Random(seed)
            u = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < m:
                    x += 1
                    if rng2.random() < 0.1:
                        continue
                    ok = True
                    for a in row:
                        if (x - a) in u:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            u.add(x - a)
                        row.append(x)
                rows.append(row)
            rows = compact(rows)
            d, s = compute_energy(rows)
            if d == 0 and s < best_scope:
                best_rows = rows
                best_scope = s

    # Final validation
    def validate(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not validate(best_rows):
        # Absolute fallback
        u = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
