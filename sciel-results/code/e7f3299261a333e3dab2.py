import time
import random
import math

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

# ---------------------------------------------------------------------------
# Global difference-slot assignment model
#
# For a row a = [a_0=0, a_1, ..., a_k] the k*(k+1)/2 pairwise differences
# a_j - a_i (i<j) must be distinct globally across all rows.  We model the
# problem as assigning each "slot" (row r, pair (i,j)) a distinct integer
# value from 1..M such that within each row the assigned values are
# consistent with some increasing sequence starting at 0.
#
# Consistency within a row: if we fix a_0=0 and assign values to all
# (i,j) slots, the row is valid iff for every i<j:
#   a_j - a_i = assigned_value(r, i, j)
# This is a system of linear equations.  We can solve it by setting
# a_0 = 0, then a_j = assigned_value(r, 0, j), and checking consistency.
# ---------------------------------------------------------------------------

def _build_slots(n_rows, k):
    """Return list of (row_idx, i, j) for all difference slots."""
    slots = []
    for r in range(n_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                slots.append((r, i, j))
    return slots

def _slots_from_rows(rows, k):
    """Extract slot values from rows."""
    n_rows = len(rows)
    vals = {}
    for r in range(n_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                vals[(r, i, j)] = rows[r][j] - rows[r][i]
    return vals

def _reconstruct_rows_from_slots(slot_vals, n_rows, k):
    """Given slot values, reconstruct rows if consistent. Return None if not."""
    rows = []
    for r in range(n_rows):
        # a_0 = 0, a_j = slot_val((r, 0, j))
        a = [0] * (k + 1)
        for j in range(1, k + 1):
            a[j] = slot_vals.get((r, 0, j))
            if a[j] is None or a[j] <= a[j-1]:
                return None
        # Verify all other slots
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                expected = a[j] - a[i]
                actual = slot_vals.get((r, i, j))
                if actual is None or actual != expected:
                    return None
        rows.append(a)
    return rows

def _slot_consistency_cost(slot_vals, n_rows, k):
    """Count how many slots are inconsistent with the row reconstructed from (0,j) slots."""
    cost = 0
    for r in range(n_rows):
        a = [0] * (k + 1)
        for j in range(1, k + 1):
            a[j] = slot_vals.get((r, 0, j), 0)
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if i == 0:
                    continue
                expected = a[j] - a[i]
                actual = slot_vals.get((r, i, j), 0)
                if actual != expected:
                    cost += 1
    return cost

def _greedy_slot_assign(n_rows, k, rng, max_scope=150):
    """Greedy: assign slots in order, pick smallest available value that keeps consistency."""
    slots = _build_slots(n_rows, k)
    slot_vals = {}
    used = set()
    
    # Sort slots: (r,0,j) first to establish row structure, then others
    def slot_priority(s):
        r, i, j = s
        if i == 0:
            return (0, r, j)
        return (1, r, i, j)
    slots.sort(key=slot_priority)
    
    for s in slots:
        r, i, j = s
        # Determine what value this slot should have based on already-assigned slots
        # For (r, 0, j): a_j, must be > a_{j-1}
        # For (r, i, j) with i>0: a_j - a_i
        if i == 0:
            # a_j = value, must be > a_{j-1}
            min_val = slot_vals.get((r, 0, j-1), 0) + 1 if j > 1 else 1
            for v in range(min_val, max_scope + 1):
                if v not in used:
                    slot_vals[s] = v
                    used.add(v)
                    break
            else:
                # Fallback: just pick something
                v = min_val + len(used)
                slot_vals[s] = v
                used.add(v)
        else:
            # Need a_j - a_i. a_j and a_i may or may not be set yet.
            # a_j = slot_vals[(r,0,j)], a_i = slot_vals[(r,0,i)]
            aj = slot_vals.get((r, 0, j))
            ai = slot_vals.get((r, 0, i))
            if aj is not None and ai is not None:
                v = aj - ai
                if v <= 0 or v in used:
                    # Inconsistent; try to adjust by picking a_j
                    # For now, skip (will cause inconsistency cost)
                    pass
                else:
                    slot_vals[s] = v
                    used.add(v)
                    continue
            # Fallback: pick smallest unused
            v = 1
            while v in used:
                v += 1
            slot_vals[s] = v
            used.add(v)
    
    rows = _reconstruct_rows_from_slots(slot_vals, n_rows, k)
    if rows is None:
        return None
    return rows

class SlotSAState:
    """SA state over slot assignments."""
    def __init__(self, slot_vals, n_rows, k, max_scope):
        self.slot_vals = dict(slot_vals)
        self.n_rows = n_rows
        self.k = k
        self.max_scope = max_scope
        self.slots = _build_slots(n_rows, k)
        self.slot_idx = {s: i for i, s in enumerate(self.slots)}
        self.used_count = [0] * (max_scope + 1)
        self.collisions = 0
        self.inconsistency = 0
        self.max_assigned = 0
        self._rebuild()
    
    def _rebuild(self):
        self.used_count = [0] * (self.max_scope + 1)
        self.collisions = 0
        self.max_assigned = 0
        for s, v in self.slot_vals.items():
            if 1 <= v <= self.max_scope:
                self.used_count[v] += 1
                if v > self.max_assigned:
                    self.max_assigned = v
        for v in range(1, self.max_scope + 1):
            if self.used_count[v] > 1:
                self.collisions += self.used_count[v] - 1
        self.inconsistency = _slot_consistency_cost(self.slot_vals, self.n_rows, self.k)
    
    def cost(self):
        return self.max_assigned + 100 * self.inconsistency + 50 * self.collisions
    
    def try_assign(self, slot, new_val):
        """Try to assign new_val to slot. Return (delta_cost, success)."""
        old_val = self.slot_vals.get(slot)
        if old_val == new_val:
            return 0, True
        if new_val < 1 or new_val > self.max_scope:
            return 0, False
        
        # Remove old
        if old_val is not None and 1 <= old_val <= self.max_scope:
            self.used_count[old_val] -= 1
            if self.used_count[old_val] >= 1:
                self.collisions -= 1
        # Add new
        self.used_count[new_val] += 1
        if self.used_count[new_val] > 1:
            self.collisions += 1
        
        old_max = self.max_assigned
        if new_val > self.max_assigned:
            self.max_assigned = new_val
        elif old_val == self.max_assigned and old_val is not None and new_val < old_val:
            # Recompute max
            self.max_assigned = 0
            for v, c in enumerate(self.used_count):
                if c > 0:
                    self.max_assigned = v
        
        self.slot_vals[slot] = new_val
        delta = (self.max_assigned - old_max) + 50 * (self.collisions - self.collisions)  # collisions already updated
        # Recompute inconsistency is expensive; approximate delta as 0 for now
        return delta, True
    
    def revert_assign(self, slot, old_val, new_val):
        """Revert slot from new_val back to old_val."""
        # Remove new
        self.used_count[new_val] -= 1
        if self.used_count[new_val] >= 1:
            self.collisions -= 1
        # Add old
        if old_val is not None and 1 <= old_val <= self.max_scope:
            self.used_count[old_val] += 1
            if self.used_count[old_val] > 1:
                self.collisions += 1
        
        old_max = self.max_assigned
        if new_val == self.max_assigned:
            self.max_assigned = 0
            for v, c in enumerate(self.used_count):
                if c > 0:
                    self.max_assigned = v
        if old_val is not None and old_val > self.max_assigned:
            self.max_assigned = old_val
        
        if old_val is None:
            del self.slot_vals[slot]
        else:
            self.slot_vals[slot] = old_val

def _slot_sa_optimize(slot_vals, n_rows, k, deadline, rng, T0=5.0, cooling=0.999, min_T=0.01):
    max_scope = max(slot_vals.values()) + 30
    state = SlotSAState(slot_vals, n_rows, k, max_scope)
    
    best_cost = state.cost()
    best_slot_vals = dict(state.slot_vals)
    
    T = T0
    slots = state.slots
    n_slots = len(slots)
    
    while time.time() < deadline:
        # Pick a random slot
        s = slots[rng.randint(0, n_slots - 1)]
        old_val = state.slot_vals.get(s)
        
        # Propose new value
        if old_val is None:
            new_val = rng.randint(1, state.max_scope)
        else:
            # Biased toward smaller values
            if rng.random() < 0.6 and old_val > 1:
                drop = rng.randint(1, max(1, old_val // 4))
                new_val = old_val - drop
            else:
                delta = int(rng.gauss(0, max(1, state.max_scope * 0.1)))
                new_val = old_val + delta
            new_val = max(1, min(state.max_scope, new_val))
        
        if new_val == old_val:
            T = max(T * cooling, min_T)
            continue
        
        delta_cost, success = state.try_assign(s, new_val)
        if not success:
            T = max(T * cooling, min_T)
            continue
        
        # Check if we can reconstruct valid rows
        rows = _reconstruct_rows_from_slots(state.slot_vals, n_rows, k)
        if rows is not None and _valid(rows, k):
            sc = _scope(rows)
            if sc < best_cost:
                best_cost = sc
                best_slot_vals = dict(state.slot_vals)
        elif delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            pass  # keep move
        else:
            state.revert_assign(s, old_val, new_val)
        
        T = max(T * cooling, min_T)
    
    # Try to reconstruct final rows
    rows = _reconstruct_rows_from_slots(best_slot_vals, n_rows, k)
    if rows is not None and _valid(rows, k):
        return _scope(rows), rows
    
    # Fallback: return best valid rows we can find
    return float('inf'), None

def _greedy_construct(n_rows, k, rng, bias=1.0):
    """Original per-row greedy as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 80
            while x <= max_x and len(candidates) < 15:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
            if not candidates:
                x = last + 1
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        rows.append(row)
    return rows

def _local_search_rows(rows, k, deadline, rng):
    """Row-element local search (fast, for polishing)."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Build used diffs
    used = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(rows)):
            row = rows[ri]
            for j in range(k, 0, -1):  # from right to left
                # Try to decrease row[j]
                lo = row[j-1] + 1
                hi = row[j] - 1
                if hi < lo:
                    continue
                # Find smallest valid value
                for new_val in range(lo, row[j]):
                    # Check new diffs
                    ok = True
                    new_diffs = set()
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = abs(row[t] - new_val)
                        if d in used and (ri, t, j) not in used:  # rough check
                            # Need more careful check
                            pass
                    # Simplified: just try and validate
                    old_val = row[j]
                    row[j] = new_val
                    # Quick check
                    valid = True
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = row[j] - row[t] if row[j] > row[t] else row[t] - row[j]
                        if d <= 0:
                            valid = False
                            break
                    if valid:
                        improved = True
                        break
                    row[j] = old_val
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions via greedy
    init_deadline = start_time + 8.0
    candidates = []
    
    for trial in range(20):
        if time.time() > init_deadline:
            break
        bias = rng.uniform(0.3, 2.5)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates:
        rows = _greedy_construct(n_rows, k, rng, bias=1.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            # Last resort: simple sequential
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                for j in range(1, k+1):
                    x = len(used) + j + 1
                    while any((x - row[i]) in used for i in range(len(row))):
                        x += 1
                    for i in range(len(row)):
                        used.add(x - row[i])
                    row.append(x)
                rows.append(row)
            return rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Slot-based SA on top candidates
    remaining = deadline - time.time()
    top_n = min(4, len(candidates))
    time_per = remaining / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 10:
            break
        s, rows = candidates[idx]
        slot_vals = _slots_from_rows(rows, k)
        sa_deadline = time.time() + time_per * 0.85
        sa_scope, sa_rows = _slot_sa_optimize(slot_vals, n_rows, k, sa_deadline, rng)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                # Add to candidates for further refinement
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Polish with row-element local search
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(10, deadline - time.time() - 5)
        # Simple iterative improvement
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            improved = False
            for ri in range(n_rows):
                row = best_rows[ri]
                for j in range(k, 0, -1):
                    lo = row[j-1] + 1
                    for new_val in range(lo, row[j]):
                        # Try
                        old = row[j]
                        row[j] = new_val
                        if _valid(best_rows, k):
                            ns = _scope(best_rows)
                            if ns < best_scope:
                                best_scope = ns
                                improved = True
                            else:
                                row[j] = old
                            break
                        else:
                            row[j] = old
            if not improved:
                break
    
    return best_rows
