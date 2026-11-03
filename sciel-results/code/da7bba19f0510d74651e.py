import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Bitset-based backtracking to find a solution with scope near lower_bound
    #    Try scopes from lower_bound upward
    found_bt = False
    for target_scope in range(lower_bound, best_scope + 1):
        if time.time() > deadline - 10.0:
            break
        solution = _backtracking_bitset(n_rows, k, target_scope, deadline)
        if solution:
            new_scope = max(max(row) for row in solution)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = solution
                found_bt = True
            break  # Found a solution at this scope; move to local search

    # 3. Simulated-annealing local search to reduce scope further
    best_solution = _simulated_annealing(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. If SA didn't reach lower bound, try more targeted backtracking
    while time.time() < deadline - 3.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        solution = _backtracking_bitset(n_rows, k, target_scope, deadline)
        if solution:
            new_scope = max(max(row) for row in solution)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = solution
                # Polish with SA
                best_solution = _simulated_annealing(best_solution, n_rows, k, deadline)
                best_scope = max(max(row) for row in best_solution)
            else:
                break
        else:
            # Can't find at this scope; give up further backtracking
            break

    return best_solution

def greedy_construct(n_rows, k):
    """Greedy construction using bitset for speed."""
    max_possible_scope = n_rows * k * (k + 1) // 2 + 50
    used_bits = 0  # bit i set means difference (i+1) is used
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Check all differences x - a for a in row
            conflict = False
            for a in row:
                d = x - a
                if d > 0 and (used_bits >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                # Mark all new differences
                for a in row:
                    d = x - a
                    if d > 0:
                        used_bits |= (1 << d)
                row.append(x)
        rows.append(row)
    return rows

def _backtracking_bitset(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Bitset-based backtracking. Uses a single Python int as a bitset for used differences.
    Difference d (1-indexed) is tracked by bit d. This gives O(1) conflict checks.
    """
    rng = random.Random(seed_offset * 7919 + 12345)
    total_positions = k + 1
    used_bits = 0
    rows = []

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == total_positions:
            return list(row_so_far)

        # Pruning: remaining positions need at least 'remaining' more values
        remaining = total_positions - pos - 1
        if current_mark + remaining > max_scope:
            return None

        # Find valid candidates: x in (current_mark, max_scope - remaining]
        # such that for all a in row_so_far, (x - a) is not in used_bits
        lo = current_mark + 1
        hi = max_scope - remaining
        if lo > hi:
            return None

        # Collect valid candidates
        valid = []
        for x in range(lo, hi + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > 0 and (used_bits >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid.append(x)

        if not valid:
            return None

        # Heuristic: try smaller values first (tighter scope), with some randomization
        n_try = min(12, len(valid))
        candidates = valid[:n_try]
        # Occasionally include some larger candidates for diversity
        if len(valid) > n_try and rng.random() < 0.3:
            extra = rng.sample(valid[n_try:], min(3, len(valid) - n_try))
            candidates += extra
        rng.shuffle(candidates)

        for x in candidates:
            # Compute new bits to set
            new_bits = 0
            for a in row_so_far:
                d = x - a
                if d > 0:
                    new_bits |= (1 << d)

            old_bits = used_bits
            used_bits |= new_bits
            res = build_row(pos + 1, x, row_so_far + [x])
            used_bits = old_bits

            if res is not None:
                return res

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all diffs for this row
        for ii in range(len(row)):
            for jj in range(ii + 1, len(row)):
                d = row[jj] - row[ii]
                used_bits |= (1 << d)

    return rows

def _simulated_annealing(solution, n_rows, k, deadline):
    """
    Simulated annealing to reduce scope.
    State: list of rows. Move: pick a random element (not first), shift it by a small random amount.
    Energy: current scope (max element). Accept worse moves with probability exp(-dE/T).
    """
    if not solution:
        return solution

    rng = random.Random(42)
    current = [list(row) for row in solution]
    best = [list(row) for row in current]
    best_scope = max(max(row) for row in best)

    # Build bitset of used differences
    def build_used_bits(rows):
        bits = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    bits |= (1 << d)
        return bits

    used_bits = build_used_bits(current)
    current_scope = best_scope

    # Annealing schedule
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 50000

    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        # Pick a random row and position (not 0)
        row_idx = rng.randrange(n_rows)
        row = current[row_idx]
        pos = rng.randrange(1, len(row))

        # Propose a shift: delta in [-3, 3], non-zero
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        new_val = row[pos] + delta

        # Validate: must be > row[pos-1] and < row[pos+1] (if exists)
        if new_val <= row[pos - 1]:
            continue
        if pos + 1 < len(row) and new_val >= row[pos + 1]:
            continue

        # Compute old diffs involving position pos
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(row[pos] - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - row[pos])

        # Compute new diffs
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)

        # Check: new_diffs must not conflict with (used - old_diffs)
        # And new_diffs must not have internal conflicts
        if len(new_diffs) != len(new_diffs):  # redundant, sets are unique
            continue

        # Check conflicts with other used diffs
        conflict = False
        for d in new_diffs:
            if d in old_diffs:
                continue
            if (used_bits >> d) & 1:
                conflict = True
                break
        if conflict:
            continue

        # Check internal conflicts in new_diffs (shouldn't happen since they're a set)
        # But we need to make sure no two new_diffs are the same (they can't be in a set)

        # Apply move
        old_bits = used_bits
        # Remove old diffs
        for d in old_diffs:
            used_bits &= ~(1 << d)
        # Add new diffs
        for d in new_diffs:
            used_bits |= (1 << d)

        row[pos] = new_val
        new_scope = max(max(r) for r in current)
        dE = new_scope - current_scope
        current_scope = new_scope

        # Accept or reject
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            # Accept
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        else:
            # Reject: revert
            row[pos] = row[pos] - delta  # undo
            used_bits = old_bits
            current_scope = new_scope - dE

        # Cool down
        T = max(T_min, T * cooling)

        # If T is very low, do a greedy descent pass
        if T <= T_min * 1.1:
            # Greedy: try to pull each element down
            for ri in range(n_rows):
                r = current[ri]
                for p in range(len(r) - 1, 0, -1):
                    orig = r[p]
                    for nv in range(r[p-1]+1, orig):
                        old_d = set()
                        for i in range(p):
                            old_d.add(orig - r[i])
                        for j in range(p+1, len(r)):
                            old_d.add(r[j] - orig)
                        new_d = set()
                        for i in range(p):
                            new_d.add(nv - r[i])
                        for j in range(p+1, len(r)):
                            new_d.add(r[j] - nv)
                        conflict = False
                        for d in new_d:
                            if d in old_d:
                                continue
                            if (used_bits >> d) & 1:
                                conflict = True
                                break
                        if not conflict:
                            for d in old_d:
                                used_bits &= ~(1 << d)
                            for d in new_d:
                                used_bits |= (1 << d)
                            r[p] = nv
                            break

            # Restart annealing if we made progress
            if max(max(r) for r in current) < best_scope:
                best_scope = max(max(r) for r in current)
                best = [list(r) for r in current]
                T = 5.0  # reheat

    return best
