import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    k1 = k + 1
    num_diffs = n_rows * k1 * (k1 - 1) // 2

    # Bitset-based difference tracking for speed
    # Max scope is bounded by ~200 for n_rows=4, k=7 (36 diffs per row, 144 total)
    MAX_SCOPE = 250  # Safe upper bound for bitset

    def compute_diffs_bitset(row):
        """Compute all positive differences in a row as a bitset."""
        mask = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d <= MAX_SCOPE:
                    mask |= (1 << d)
        return mask

    def greedy_construct():
        """Fast greedy: smallest valid x each step."""
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = 0
                    for a in row:
                        new_diffs |= (1 << (x - a))
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    def generate_row_random(used_diffs, k_val, rng, p_small=0.75, window=4, max_scope=200):
        """Generate a random row given used diffs. Returns None if stuck."""
        row = [0]
        row_diffs = 0
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used_diffs | row_diffs
            while x < max_scope:
                new_diffs = 0
                for a in row:
                    new_diffs |= (1 << (x - a))
                if not (new_diffs & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if rng.random() < p_small:
                nxt = candidates[0]
            else:
                nxt = rng.choice(candidates)
            row.append(nxt)
            row_diffs |= (1 << (nxt - row[-2])) if len(row) >= 2 else 0
            # Recompute row_diffs properly
            row_diffs = compute_diffs_bitset(row)
        return row

    def local_compact(rows, used_mask, max_rounds=100):
        """Deterministic compaction: try to shift each element left."""
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for ri in range(n_rows):
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    old = row[pos]
                    if old <= 1:
                        continue
                    # Remove diffs
                    new_mask = compute_diffs_bitset(row)
                    # We need to remove the old element's contributions
                    # This is tricky with bitsets, so let's use a different approach
                    # Rebuild the mask without the element at pos
                    temp_row = row[:pos] + row[pos+1:]
                    mask_without = compute_diffs_bitset(temp_row)
                    # Also need to remove diffs involving old element with other elements
                    # Actually, let's just recompute the full used mask minus this row's old diffs
                    old_row_mask = compute_diffs_bitset(row)
                    # Remove old row mask from used
                    used_mask &= ~old_row_mask
                    
                    row.pop(pos)
                    min_v = row[pos - 1] + 1 if pos > 0 else 1
                    found = False
                    for nv in range(min_v, old):
                        test_row = row[:pos] + [nv] + row[pos:]
                        # Check if strictly increasing
                        if pos > 0 and nv <= row[pos-1]:
                            continue
                        if pos < len(row) and nv >= row[pos]:
                            continue
                        new_mask = compute_diffs_bitset(test_row)
                        if not (new_mask & used_mask):
                            row = test_row
                            used_mask |= new_mask
                            found = True
                            break
                    if found:
                        ns = max(max(r) for r in rows)
                        if ns < scope:
                            scope = ns
                            improved = True
                    else:
                        # Restore old
                        row.insert(pos, old)
                        used_mask |= compute_diffs_bitset(row)
        return rows, scope

    def local_shift(rows, used_mask, rng, max_iters=500):
        """Random local improvement: shift random elements to smaller values."""
        rows = [list(r) for r in rows]
        scope = max(max(r) for r in rows)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            ri = rng.randint(0, n_rows - 1)
            row = rows[ri]
            ei = rng.randint(1, len(row) - 1)
            old = row[ei]
            if old <= 1:
                continue
            
            old_row_mask = compute_diffs_bitset(row)
            used_mask &= ~old_row_mask
            
            row.pop(ei)
            min_v = row[ei - 1] + 1 if ei > 0 else 1
            found = False
            for nv in range(min_v, old):
                if ei > 0 and nv <= row[ei-1]:
                    continue
                if ei < len(row) and nv >= row[ei]:
                    continue
                test_row = row[:ei] + [nv] + row[ei:]
                new_mask = compute_diffs_bitset(test_row)
                if not (new_mask & used_mask):
                    row = test_row
                    used_mask |= new_mask
                    found = True
                    break
            if found:
                ns = max(max(r) for r in rows)
                if ns < scope:
                    scope = ns
            else:
                row.insert(ei, old)
                used_mask |= compute_diffs_bitset(row)
        return rows, scope

    # Phase 1: Greedy baseline
    rows = greedy_construct()
    if not rows:
        return [[0] + list(range(1, k1)) for _ in range(n_rows)]

    used = 0
    for row in rows:
        used |= compute_diffs_bitset(row)

    best_scope = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    # Phase 2: Deterministic compaction
    rows, scope = local_compact(rows, used, max_rounds=200)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
        used = 0
        for row in best_rows:
            used |= compute_diffs_bitset(row)

    # Phase 3: Simulated Annealing with row regeneration
    rng = random.Random(42)
    T = 8.0
    cooling = 0.997
    T_min = 0.15
    sa_deadline = deadline - 25.0  # Leave 25s for refinement

    current_rows = [list(r) for r in best_rows]
    current_used = used
    current_scope = best_scope

    while time.time() < sa_deadline:
        ri = rng.randint(0, n_rows - 1)
        old_row = current_rows[ri]
        old_diffs = compute_diffs_bitset(old_row)
        current_used &= ~old_diffs

        # Regenerate row with randomization
        new_row = generate_row_random(current_used, k, rng, p_small=0.8, window=3, max_scope=best_scope + 20)

        if new_row is None:
            current_used |= old_diffs
            continue

        new_diffs = compute_diffs_bitset(new_row)
        new_scope = max(max(current_rows[i]) for i in range(n_rows) if i != ri)
        new_scope = max(new_scope, max(new_row))

        delta = new_scope - current_scope

        if delta <= 0:
            # Accept
            current_rows[ri] = new_row
            current_used |= new_diffs
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_rows[ri] = new_row
                current_used |= new_diffs
                current_scope = new_scope
            else:
                current_used |= old_diffs

        T = max(T_min, T * cooling)

    # Phase 4: Refinement on best solution
    rows = [list(r) for r in best_rows]
    used = 0
    for row in rows:
        used |= compute_diffs_bitset(row)
    scope = best_scope

    # Local compaction
    rows, scope = local_compact(rows, used, max_rounds=150)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
        used = 0
        for row in best_rows:
            used |= compute_diffs_bitset(row)

    # Local shift
    rows, scope = local_shift(rows, used, rng, max_iters=400)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
        used = 0
        for row in best_rows:
            used |= compute_diffs_bitset(row)

    # Phase 5: More SA with tighter parameters if time remains
    if time.time() < deadline - 10:
        rng2 = random.Random(12345)
        T = 3.0
        cooling = 0.998
        T_min = 0.1
        current_rows = [list(r) for r in best_rows]
        current_used = 0
        for row in current_rows:
            current_used |= compute_diffs_bitset(row)
        current_scope = best_scope

        while time.time() < deadline - 5:
            ri = rng2.randint(0, n_rows - 1)
            old_row = current_rows[ri]
            old_diffs = compute_diffs_bitset(old_row)
            current_used &= ~old_diffs

            new_row = generate_row_random(current_used, k, rng2, p_small=0.85, window=3, max_scope=best_scope + 10)

            if new_row is None:
                current_used |= old_diffs
                continue

            new_diffs = compute_diffs_bitset(new_row)
            other_max = max(max(current_rows[i]) for i in range(n_rows) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope

            if delta <= 0:
                current_rows[ri] = new_row
                current_used |= new_diffs
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                prob = math.exp(-delta / T)
                if rng2.random() < prob:
                    current_rows[ri] = new_row
                    current_used |= new_diffs
                    current_scope = new_scope
                else:
                    current_used |= old_diffs

            T = max(T_min, T * cooling)

    # Final compaction
    if time.time() < deadline - 2:
        rows = [list(r) for r in best_rows]
        used = 0
        for row in rows:
            used |= compute_diffs_bitset(row)
        rows, scope = local_compact(rows, used, max_rounds=100)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
