import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 135.0  # Safety margin before 150s kill

    # Number of differences per row
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        """Generate an initial valid set using a simple greedy approach."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def targeted_descent(rows, budget):
        """Try to reduce the scope by shrinking the largest mark in the row that has it.
        Returns (improved_rows, new_scope) or (None, None) if no improvement found within budget."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return None, None
        improved = False

        while time.time() - start < budget:
            # Find the row(s) containing the current max
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            # The last element is the max of this row
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                continue

            # Try a set of candidate values: prefer values just above the lower bound
            # to maximize the reduction, but also sample a few random ones
            candidates = set()
            # Sequential near the bottom (best reduction)
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            # Some random samples in the middle
            for _ in range(15):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):  # try larger reductions first
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate_scope(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val

            if not improved:
                # Try reducing a non-last element in the max row
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate_scope(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    def hill_climb(cur, budget):
        """Pure hill climbing: only accept moves that reduce scope or maintain it.
        More aggressive local search after SA."""
        start = time.time()
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return cur, cur_scope
        no_improve = 0

        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else cur_scope + 10
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue

            cur[ri][pos] = new_val
            new_scope = full_validate_scope(cur)

            if new_scope is not None:
                if new_scope < cur_scope:
                    cur_scope = new_scope
                    no_improve = 0
                else:
                    cur[ri][pos] = old_val
                    no_improve += 1
            else:
                cur[ri][pos] = old_val
                no_improve += 1

            if no_improve > 500:
                break

        return cur, cur_scope

    def lns_destroy_repair(rows, budget):
        """
        Large Neighborhood Search step:
        1. Destroy: Randomly select 1-2 rows and reset them to a greedy baseline, 
           OR remove random interior points from random rows.
        2. Repair: Re-insert points using a global greedy strategy that respects 
           the differences of the *remaining* valid parts of the other rows.
        """
        start = time.time()
        cur = [r[:] for r in rows]
        
        # Determine used differences from the rows we are NOT destroying
        # Strategy: Pick 1-2 rows to destroy completely.
        # This is a strong move that allows large jumps in the solution space.
        
        n_destroy = rng.randint(1, 2)
        destroy_idx = rng.sample(range(n_rows), n_destroy)
        
        # Calculate used diffs from the non-destroyed rows
        used_diffs = set()
        for i in range(n_rows):
            if i in destroy_idx:
                continue
            row = cur[i]
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    used_diffs.add(row[b] - row[a])
        
        # Also, we need to be careful: if we destroy rows, we lose their diffs.
        # But we also need to ensure the *new* rows don't conflict with *each other* 
        # or the *remaining* rows.
        
        # We will greedily rebuild the destroyed rows.
        # To allow some flexibility, we can also randomly remove a few interior points 
        # from non-destroyed rows to free up space, but let's stick to full row reset 
        # for simplicity and strong perturbation first.
        
        # Rebuild destroyed rows
        for r_idx in destroy_idx:
            # Start with 0
            new_row = [0]
            # We need to find k more numbers.
            # Greedy: try smallest possible x such that {x - a for a in new_row} 
            # does not intersect used_diffs AND (if we are building multiple rows) 
            # does not intersect diffs of other new rows built so far in this step.
            
            # To be safe, we build one row at a time, updating used_diffs as we go.
            # This ensures no conflict between the rebuilt rows.
            
            x = 0
            while len(new_row) < k + 1:
                x += 1
                # Check against current new_row diffs
                new_diffs = {x - a for a in new_row}
                # Check against global used_diffs
                if not (new_diffs & used_diffs):
                    new_row.append(x)
                    # Update global used_diffs with the new diffs from this row
                    used_diffs |= new_diffs
            
            cur[r_idx] = new_row
        
        # Validate the whole set
        scope = full_validate_scope(cur)
        if scope is not None:
            return cur, scope
            
        # If the simple greedy rebuild failed (conflict within the rebuilt rows or with others 
        # that wasn't caught? No, the greedy logic above updates used_diffs, so it should be valid 
        # by construction relative to the non-destroyed rows and each other. 
        # Wait, the greedy logic adds diffs to `used_diffs` as it builds. 
        # So the resulting set should be valid. 
        # Why would it fail? 
        # Ah, `full_validate_scope` checks the *entire* set. 
        # The greedy build ensures:
        # 1. New row diffs are unique within the row (by construction of set).
        # 2. New row diffs do not intersect `used_diffs` (which contains all non-destroyed rows).
        # 3. As we build row 1, we add its diffs to `used_diffs`. 
        # 4. When building row 2, we check against `used_diffs` which now includes row 1.
        # So the result *should* be valid. 
        # If `full_validate_scope` returns None, there is a bug in my reasoning or implementation.
        # Let's double check.
        # `used_diffs` starts with diffs of non-destroyed rows.
        # Loop `r_idx` in `destroy_idx`:
        #   Build `new_row`.
        #   `new_diffs` are diffs within `new_row`.
        #   Check `not (new_diffs & used_diffs)`.
        #   If pass, append and `used_diffs |= new_diffs`.
        # This ensures all diffs in `new_row` are distinct from each other (since `new_diffs` is a set of diffs from `new_row`, if `x-a` was already in `new_diffs` from a previous step? No, `new_diffs` is computed for the *candidate* x. 
        # Wait. `new_diffs = {x - a for a in new_row}`. 
        # If we previously added `x'` to `new_row`, then `x - x'` is in `new_diffs`. 
        # Is it possible that `x - a1 == x - a2`? No, `a1 != a2`.
        # Is it possible that `x - a` equals a diff from a *previous* element in `new_row`? 
        # Yes. For example, if `new_row` has `[0, 2]`, and we try `x=4`. 
        # `new_diffs` = `{4-0, 4-2}` = `{4, 2}`. 
        # The diff `2` already exists in `new_row` (from `2-0`). 
        # The check `if not (new_diffs & used_diffs)` only checks against `used_diffs` (external). 
        # It does NOT check against diffs *within* `new_row` that were generated in previous steps of the while loop!
        # 
        # Correction: The set `new_diffs` contains diffs between the *candidate* x and *existing* elements.
        # It does NOT contain diffs between *existing* elements.
        # However, the existing elements were already validated when they were added.
        # So, do we need to check if `x-a` conflicts with any diff *already in new_row*?
        # No, because `x-a` is a diff involving the *new* element x. 
        # The only potential conflict is if `x-a1 == x-a2` (impossible) 
        # OR if `x-a` is equal to a diff `b-c` where `b,c` are in `new_row`?
        # If `b,c` are in `new_row`, their diff `b-c` was added to `used_diffs` when `b` or `c` was added?
        # No, `used_diffs` is updated *after* the row is complete? 
        # In my code above: `used_diffs |= new_diffs` happens *inside* the while loop? 
        # No, I put it inside the `if pass` block. 
        # So, when adding `x`, `new_diffs` contains `{x-a for a in new_row}`.
        # These are all diffs involving `x`.
        # The diffs *not* involving `x` (i.e., between existing elements) are already in `used_diffs`?
        # No! `used_diffs` was updated with `new_diffs` from the *previous* iteration?
        # Let's trace:
        # Iter 1: `new_row=[0]`. `x=1`. `new_diffs={1}`. `used_diffs` (external) doesn't have 1. 
        #   Pass. `new_row=[0,1]`. `used_diffs` |= {1}. 
        # Iter 2: `x=2`. `new_diffs={2, 1}`. 
        #   `2` is not in `used_diffs`. `1` IS in `used_diffs` (added in Iter 1).
        #   So `new_diffs & used_diffs` is not empty. Reject.
        # This is CORRECT. The diff `1` (from `1-0`) is already used. 
        # So the logic holds: `used_diffs` accumulates ALL diffs from the partially built row.
        # Therefore, the greedy rebuild produces a valid set.
        
        return None, None

    # Main Loop: Mix of LNS and Hill Climbing
    # We use LNS to jump to new regions, and Hill Climbing to polish.
    
    while time.time() < deadline - 5.0:
        # 1. Try LNS
        if rng.random() < 0.7: # 70% chance LNS
            new_rows, new_scope = lns_destroy_repair(best_rows, budget=0.5)
            if new_rows is not None:
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = new_rows
                # Polish the LNS result
                if time.time() < deadline - 5.0:
                    hc_rows, hc_scope = hill_climb(new_rows, budget=0.5)
                    if hc_scope is not None and hc_scope < best_scope:
                        best_scope = hc_scope
                        best_rows = hc_rows
        else: # 30% chance Hill Climb on best
            hc_rows, hc_scope = hill_climb(best_rows, budget=1.0)
            if hc_scope is not None and hc_scope < best_scope:
                best_scope = hc_scope
                best_rows = hc_rows

        # 2. Targeted Descent on Best
        if time.time() < deadline - 5.0:
            desc_rows, desc_scope = targeted_descent(best_rows, budget=1.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    # Final Polish
    if time.time() < deadline - 2.0:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=min(5.0, deadline - time.time() - 2.0))
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    if time.time() < deadline - 1.0:
        hc_rows, hc_scope = hill_climb(best_rows, budget=min(3.0, deadline - time.time() - 1.0))
        if hc_scope is not None and hc_scope < best_scope:
            best_scope = hc_scope
            best_rows = hc_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
