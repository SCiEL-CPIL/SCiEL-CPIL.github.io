"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across rows.
    Minimize the largest entry.

    Structural approach:
    - Pre-compute a candidate pool of low-scope rows
    - Use difference-frequency table for O(1) conflict updates
    - Row-level replacement search (swap entire rows)
    - Simulated annealing over row compositions
    """
    deadline = time.time() + 145

    def get_scope(rows):
        return max(max(row) for row in rows)

    def row_diffs(row):
        """Return multiset of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def build_freq_table(rows):
        """Build difference frequency Counter."""
        freq = Counter()
        for row in rows:
            for d in row_diffs(row):
                freq[d] += 1
        return freq

    def count_conflicts(freq):
        """Count total conflicts from frequency table."""
        return sum(c - 1 for c in freq.values() if c > 1)

    def remove_row_from_freq(freq, row):
        for d in row_diffs(row):
            freq[d] -= 1
            if freq[d] == 0:
                del freq[d]

    def add_row_to_freq(freq, row):
        for d in row_diffs(row):
            freq[d] += 1

    # --- Greedy initialization ---
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = get_scope(best_rows)
    freq = build_freq_table(best_rows)
    best_conflicts = count_conflicts(freq)

    if best_conflicts == 0:
        return best_rows

    # --- Build candidate pool ---
    # Generate many candidate rows with scope up to best_scope + margin
    # Focus on rows that are "compatible" with low conflict

    max_scope_try = best_scope + 20
    candidate_pool = []
    seen_rows = set()

    # Add the greedy rows as candidates
    for r in rows:
        key = tuple(r)
        if key not in seen_rows:
            seen_rows.add(key)
            candidate_pool.append(list(r))

    # Generate random candidate rows with bounded scope
    n_candidates_target = min(500, n_rows * 30)
    t_start = time.time()
    while len(candidate_pool) < n_candidates_try if False else len(candidate_pool) < n_candidates_target:
        if time.time() - t_start > 10:
            break
        # Random row: pick k+1 values from [0, max_scope_try]
        # Use a biased approach: start with 0, then random gaps
        row = [0]
        for i in range(k):
            remaining_slots = k - i
            remaining_range = max_scope_try - row[-1]
            if remaining_range < remaining_slots:
                break
            # Choose next value
            min_next = row[-1] + 1
            max_next = max_scope_try - (remaining_slots - 1)
            if min_next > max_next:
                break
            # Biased toward smaller values (lower scope)
            val = random.randint(min_next, max_next)
            row.append(val)
        if len(row) == k + 1:
            key = tuple(row)
            if key not in seen_rows:
                seen_rows.add(key)
                candidate_pool.append(list(r)) if False else candidate_pool.append(list(row))

    # Pre-compute diffs for each candidate
    candidate_diffs = [row_diffs(r) for r in candidate_pool]

    # --- Row-level simulated annealing ---
    current_rows = [list(r) for r in best_rows]
    current_freq = build_freq_table(current_rows)
    current_conflicts = count_conflicts(current_freq)
    current_scope = get_scope(current_rows)

    # Track which candidate index each row came from (for efficient swap)
    # We'll just use the rows directly

    T = 50.0
    cooling = 0.9995
    min_T = 0.5
    step = 0

    while time.time() < deadline - 5 and T > min_T:
        step += 1

        # Pick a random row to replace
        r_idx = random.randint(0, n_rows - 1)
        old_row = current_rows[r_idx]

        # Pick a random candidate from the pool
        c_idx = random.randint(0, len(candidate_pool) - 1)
        new_row = candidate_pool[c_idx]

        if new_row == old_row:
            continue

        # Compute delta conflicts efficiently
        # Remove old row from freq
        remove_row_from_freq(current_freq, old_row)
        # Add new row to freq
        add_row_from_freq_diffs = candidate_diffs[c_idx]
        for d in add_row_from_freq_diffs:
            current_freq[d] += 1

        new_conflicts = count_conflicts(current_freq)
        new_scope = get_scope(current_rows)
        # Need to update scope: new_scope = max of all rows' max values
        # Actually let's compute it properly
        # But that's O(n_rows * k). For speed, track max per row.
        # For now, approximate: new_scope >= max(old_scope_without_r_idx, max(new_row))
        # Let's just compute it
        # Actually, let's revert and compute delta more carefully

        # Revert
        for d in add_row_from_freq_diffs:
            current_freq[d] -= 1
            if current_freq[d] == 0:
                del current_freq[d]
        for d in row_diffs(old_row):
            current_freq[d] += 1

        # Compute delta: conflicts when old row removed vs new row added
        # old_conflicts_contribution: conflicts involving old_row's diffs
        # This is complex; let's just do full recompute for correctness
        # but use the freq table for speed

        # Actually, let's just apply and check
        remove_row_from_freq(current_freq, old_row)
        for d in candidate_diffs[c_idx]:
            current_freq[d] += 1

        new_conflicts = count_conflicts(current_freq)

        # Compute new scope
        new_scope = 0
        for i in range(n_rows):
            if i == r_idx:
                new_scope = max(new_scope, max(new_row))
            else:
                new_scope = max(new_scope, max(current_rows[i]))

        delta_conflicts = new_conflicts - current_conflicts
        delta_scope = new_scope - current_scope

        # Acceptance
        if delta_conflicts < 0:
            accept = True
        elif delta_conflicts == 0 and delta_scope < 0:
            accept = True
        else:
            delta_energy = delta_conflicts * 50 + delta_scope
            if delta_energy < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_energy / T)

        if accept:
            current_rows[r_idx] = list(new_row)
            current_conflicts = new_conflicts
            current_scope = new_scope

            if current_conflicts < best_conflicts or \
               (current_conflicts == best_conflicts and current_scope < best_scope):
                best_conflicts = current_conflicts
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            for d in candidate_diffs[c_idx]:
                current_freq[d] -= 1
                if current_freq[d] == 0:
                    del current_freq[d]
            for d in row_diffs(old_row):
                current_freq[d] += 1

        T *= cooling

    # --- Local polish: try to reduce scope of the binding row ---
    remaining = deadline - time.time()
    if remaining > 3:
        polish_time = min(remaining - 3, 30)
        polish_start = time.time()

        for _ in range(50):
            if time.time() - polish_start > polish_time:
                break
            # Find the row with the largest max value
            max_idx = max(range(n_rows), key=lambda i: max(best_rows[i]))
            row = best_rows[max_idx]
            old_last = row[-1]

            # Try to reduce the last element
            for new_last in range(old_last - 1, row[-2], -1):
                row[-1] = new_last
                # Check validity
                freq_check = build_freq_table(best_rows)
                conf = count_conflicts(freq_check)
                if conf == 0:
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        break
                else:
                    row[-1] = old_last
                    break
            else:
                row[-1] = old_last

    return best_rows
# EVOLVE-BLOCK-END
