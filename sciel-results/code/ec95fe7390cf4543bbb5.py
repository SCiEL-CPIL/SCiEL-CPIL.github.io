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

def _greedy_row(used, k, max_scope, rng, bias=1.0):
    """Build one row greedily."""
    row = [0]
    last = 0
    for j in range(1, k + 1):
        best_val = None
        best_score = float('inf')
        # Try values from last+1 up to some limit
        limit = min(last + 60, max_scope)
        for x in range(last + 1, limit + 1):
            # Check all differences
            ok = True
            new_diffs = []
            for i in range(len(row)):
                d = x - row[i]
                if d in used:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                # Score: prefer smaller values, but also consider "spread"
                score = x
                if bias > 0:
                    score = x ** bias
                if score < best_score:
                    best_score = score
                    best_val = x
                    best_diffs = new_diffs
                # Don't stop immediately; collect a few options
                if best_score < last + 5:
                    break
        if best_val is None:
            # Fallback: find any valid value
            x = last + 1
            while x <= max_scope + 50:
                ok = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if ok:
                    best_val = x
                    best_diffs = new_diffs
                    break
                x += 1
            if best_val is None:
                return None
        
        row.append(best_val)
        last = best_val
        used.update(best_diffs)
    
    return row

def _greedy_construct(n_rows, k, rng, max_scope=130, bias=1.0):
    """Construct all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_scope, rng, bias)
        if row is None:
            return None
        rows.append(row)
    return rows

def _backtrack_construct(n_rows, k, rng, max_scope=120):
    """Backtracking construction with forward checking."""
    used = set()
    rows = []
    
    def backtrack(row_idx, pos, current_row, last_val):
        if pos == k + 1:
            rows.append(list(current_row))
            return True
        
        # Collect candidates with some randomization
        candidates = []
        for v in range(last_val + 1, max_scope + 1):
            valid = True
            new_diffs = []
            for i in range(len(current_row)):
                d = v - current_row[i]
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                candidates.append((v, new_diffs))
                if len(candidates) > 15:
                    break
        
        if not candidates:
            return False
        
        # Randomize order with some preference for smaller values
        if rng.random() < 0.6:
            rng.shuffle(candidates)
        
        for v, new_diffs in candidates:
            current_row.append(v)
            used.update(new_diffs)
            
            if backtrack(row_idx, pos + 1, current_row, v):
                return True
            
            current_row.pop()
            used.difference_update(new_diffs)
        
        return False
    
    for r in range(n_rows):
        if not backtrack(r, 1, [0], 0):
            return None
    return rows

class SAState:
    """Simulated annealing state with efficient diff tracking."""
    def __init__(self, rows, k, rng):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.rng = rng
        self.used_diffs = set()
        self._rebuild()
    
    def _rebuild(self):
        self.used_diffs = set()
        for row in self.rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    self.used_diffs.add(row[j] - row[i])
    
    def scope(self):
        return max(a for r in self.rows for a in r[1:])
    
    def try_move(self, row_idx, pos, new_val):
        """Try to change rows[row_idx][pos] to new_val. Return (delta_scope, success)."""
        row = self.rows[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return 0, True
        if new_val <= row[pos-1] or (pos < len(row)-1 and new_val >= row[pos+1]):
            return 0, False
        
        # Remove old diffs from used set
        removed = []
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(row[pos] - row[t])
            removed.append(d)
            self.used_diffs.discard(d)
        
        # Check if new_val creates conflicts
        conflicts = 0
        new_diffs = []
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(new_val - row[t])
            if d in self.used_diffs:
                conflicts += 1
            new_diffs.append(d)
        
        if conflicts > 0:
            # Restore old diffs
            for d in removed:
                self.used_diffs.add(d)
            return 0, False
        
        # Apply move
        row[pos] = new_val
        old_scope = self.scope()
        
        # Add new diffs
        for d in new_diffs:
            self.used_diffs.add(d)
        
        new_scope = self.scope()
        return new_scope - old_scope, True
    
    def revert_move(self, row_idx, pos, old_val):
        """Revert rows[row_idx][pos] to old_val."""
        row = self.rows[row_idx]
        new_val = row[pos]
        
        # Remove new diffs
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(new_val - row[t])
            self.used_diffs.discard(d)
        
        # Restore old diffs
        row[pos] = old_val
        for t in range(len(row)):
            if t == pos:
                continue
            d = abs(old_val - row[t])
            self.used_diffs.add(d)

def _sa_optimize(rows, k, deadline, rng, T0=8.0, cooling=0.9997, min_T=0.3):
    """Simulated annealing optimization on rows."""
    state = SAState(rows, k, rng)
    best_rows = [list(r) for r in rows]
    best_scope = state.scope()
    
    T = T0
    no_improve = 0
    restart_thresh = 3000
    
    while time.time() < deadline:
        # Pick random row and position
        row_idx = rng.randint(0, state.n_rows - 1)
        pos = rng.randint(1, state.k)
        row = state.rows[row_idx]
        old_val = row[pos]
        
        # Propose new value: bias toward decreasing
        if rng.random() < 0.8:
            # Decrease
            max_drop = old_val - row[pos-1] - 1
            if max_drop <= 0:
                T = max(T * cooling, min_T)
                continue
            drop = rng.randint(1, max_drop)
            new_val = old_val - drop
        else:
            # Increase slightly
            delta = rng.randint(1, 4)
            new_val = old_val + delta
        
        # Bound check
        lo = row[pos-1] + 1
        hi = row[pos+1] - 1 if pos < state.k else 200
        new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            T = max(T * cooling, min_T)
            continue
        
        delta, success = state.try_move(row_idx, pos, new_val)
        if not success:
            T = max(T * cooling, min_T)
            continue
        
        if delta <= 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
            # Accept
            if state.scope() < best_scope:
                best_scope = state.scope()
                best_rows = [list(r) for r in state.rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert
            state.revert_move(row_idx, pos, old_val)
            no_improve += 1
        
        T = max(T * cooling, min_T)
        
        if no_improve > restart_thresh:
            # Restart from best
            state.rows = [list(r) for r in best_rows]
            state._rebuild()
            T = T0
            no_improve = 0
    
    return best_scope, best_rows

def _local_search(rows, k, deadline, rng):
    """Simple local search: try to decrease each element."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Build used diffs
    used = set()
    for r in best_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used.add(r[j] - r[i])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(best_rows)):
            row = best_rows[ri]
            for j in range(k, 0, -1):
                lo = row[j-1] + 1
                for new_val in range(lo, row[j]):
                    # Try the move
                    old_val = row[j]
                    
                    # Remove old diffs
                    removed = []
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = abs(row[j] - row[t])
                        removed.append(d)
                        used.discard(d)
                    
                    # Check new
                    ok = True
                    new_diffs = []
                    for t in range(k+1):
                        if t == j:
                            continue
                        d = abs(new_val - row[t])
                        if d in used:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if ok:
                        row[j] = new_val
                        used.update(new_diffs)
                        ns = _scope(best_rows)
                        if ns < best_scope:
                            best_scope = ns
                            improved = True
                        else:
                            # Revert
                            row[j] = old_val
                            used.difference_update(new_diffs)
                            used.update(removed)
                        break
                    else:
                        # Restore old diffs
                        for d in removed:
                            used.add(d)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    init_deadline = start_time + 20.0
    candidates = []
    
    # Try different max_scope values
    for max_s in [110, 115, 120, 125, 130]:
        if time.time() > init_deadline:
            break
        
        # Backtracking
        for trial in range(3):
            if time.time() > init_deadline:
                break
            rows = _backtrack_construct(n_rows, k, rng, max_scope=max_s)
            if rows is not None and _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
        
        # Greedy with different biases
        for bias in [0.5, 1.0, 1.5, 2.0]:
            if time.time() > init_deadline:
                break
            rows = _greedy_construct(n_rows, k, rng, max_scope=max_s, bias=bias)
            if rows is not None and _valid(rows, k):
                s = _scope(rows)
                candidates.append((s, rows))
                if s < best_scope:
                    best_scope = s
                    best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Last resort: simple construction
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
    
    # Phase 2: SA optimization on top candidates
    remaining = deadline - time.time()
    top_n = min(4, len(candidates))
    time_per = remaining * 0.7 / max(1, top_n)
    
    for idx in range(top_n):
        if time.time() > deadline - 10:
            break
        s, rows = candidates[idx]
        sa_deadline = time.time() + time_per
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if sa_rows is not None and _valid(sa_rows, k):
            actual = _scope(sa_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = sa_rows
                candidates.append((actual, sa_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Final polish with local search
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(8, deadline - time.time() - 5)
        _, polished = _local_search(best_rows, k, polish_deadline, rng)
        if _valid(polished, k):
            ps = _scope(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = polished
    
    return best_rows
