import time
import random
import math

class SAState:
    def __init__(self, rows, k, max_scope_bound=200):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.max_scope_bound = max_scope_bound
        # used_diffs[d] = count of times difference d appears
        self.used_diffs = [0] * (max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        
        self._rebuild_state()

    def _rebuild_state(self):
        self.used_diffs = [0] * (self.max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        
        for r in self.rows:
            if r[-1] > self.scope:
                self.scope = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= self.max_scope_bound:
                        self.used_diffs[d] += 1
                    # If d > max_scope_bound, it's effectively unique for our collision 
                    # purposes if max_scope_bound is high enough (e.g., 200 for scope 112).
                    # If d exceeds bound, we ignore it in collision count, assuming 
                    # we are trying to keep scope small.
        
        # Recalculate collisions accurately
        self.collisions = 0
        for d in range(1, self.max_scope_bound + 1):
            if self.used_diffs[d] > 1:
                self.collisions += self.used_diffs[d] - 1

    def get_cost(self):
        return self.scope + 1000 * self.collisions

    def try_move(self, i, j, new_val):
        """
        Attempts to change rows[i][j] to new_val.
        Returns (delta_cost, success)
        If success is False, state is unchanged.
        If success is True, state is updated.
        """
        row = self.rows[i]
        old_val = row[j]
        
        if new_val == old_val:
            return 0, True
        
        # Check strict increase
        if j > 0 and new_val <= row[j-1]:
            return 0, False
        if j < self.k and new_val >= row[j+1]:
            return 0, False
            
        # 1. Remove old diffs involving index j
        delta_coll = 0
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                self.used_diffs[d] -= 1
                # If count drops from 2 to 1, collision removed (-1)
                # If count drops from 1 to 0, no collision change
                if self.used_diffs[d] == 1:
                    delta_coll -= 1
                # If it was >2, it remains a collision, so no change in count?
                # Wait. Collisions = sum(max(0, count-1)).
                # If count was 2, coll contribution 1. New count 1, coll contribution 0. Delta -1.
                # If count was 3, coll contribution 2. New count 2, coll contribution 1. Delta -1.
                # So delta is -1 if new_count >= 1 and old_count > new_count?
                # Actually:
                # old_coll_contrib = max(0, old_count - 1)
                # new_coll_contrib = max(0, new_count - 1)
                # We decremented count.
                # If new_count >= 1: delta = new_coll_contrib - old_coll_contrib = (new_count - 1) - (new_count - 1 + 1) = -1.
                # If new_count == 0: delta = 0 - 0 = 0.
                # So: if self.used_diffs[d] > 0: delta_coll -= 1
                
        # Let's restart the collision logic properly.
        # We already modified used_diffs in the loop above. Let's revert that logic in code below.
        
        # Re-implementing try_move with clear steps:
        
        # 1. Calculate delta collisions for removal
        delta_coll = 0
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                # We are about to decrement used_diffs[d]
                if self.used_diffs[d] > 1:
                    # It was a collision (count >= 2). After decrement, count >= 1.
                    # If count was 2, it becomes 1 (no collision). Delta -1.
                    # If count was 3, it becomes 2 (still collision, but one less). Delta -1.
                    delta_coll -= 1
                # If count was 1, it becomes 0. No collision before or after. Delta 0.
                
        # 2. Calculate delta collisions for addition
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                # We are about to increment used_diffs[d]
                if self.used_diffs[d] > 0:
                    # It exists (count >= 1). After increment, count >= 2.
                    # If count was 1, it becomes 2 (collision added). Delta +1.
                    # If count was 2, it becomes 3 (collision increased). Delta +1.
                    delta_coll += 1
                # If count was 0, it becomes 1. No collision. Delta 0.

        # 3. Apply changes to used_diffs
        for t in range(self.k + 1):
            if t == j:
                continue
            d_old = abs(row[t] - old_val)
            if d_old <= self.max_scope_bound:
                self.used_diffs[d_old] -= 1
                
            d_new = abs(row[t] - new_val)
            if d_new <= self.max_scope_bound:
                self.used_diffs[d_new] += 1
                
        self.collisions += delta_coll
        
        # 4. Update Scope
        old_scope = self.scope
        # Only row i's last element or the moved element (if it was the max) affects scope?
        # Scope is max of all elements.
        # If new_val > old_scope, scope increases.
        # If old_val == old_scope and new_val < old_val, scope might decrease.
        
        if new_val > self.scope:
            self.scope = new_val
        elif old_val == self.scope and new_val < old_val:
            # Recalculate scope
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s
        # Else scope unchanged
        
        row[j] = new_val
        
        delta_cost = (self.scope - old_scope) + 1000 * delta_coll
        return delta_cost, True

    def revert_move(self, i, j, new_val, old_val):
        """
        Reverts a move that was applied.
        """
        row = self.rows[i]
        
        delta_coll = 0
        # 1. Remove new diffs (decrement)
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
                self.used_diffs[d] -= 1

        # 2. Add old diffs (increment)
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1
                self.used_diffs[d] += 1

        self.collisions += delta_coll
        
        # Update Scope
        old_scope = self.scope
        if old_val > self.scope:
            self.scope = old_val
        elif new_val == self.scope and old_val < new_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s
            
        row[j] = old_val

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

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _greedy_random(n_rows, k, rng, aggressiveness=0.5):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            # Look for valid candidates
            candidates = []
            x = last + 1
            # Limit search window to keep it fast, but allow expansion if needed
            max_search = last + 50
            while x <= max_search:
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
                    if len(candidates) >= 5:
                        break
                x += 1
            
            if not candidates:
                # Fallback: find the first valid one regardless of window
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
            
            # Select candidate
            # Bias towards smaller values, but allow randomness
            # Weight = 1 / (value - last + 1)^power
            power = 1.0 + aggressiveness
            weights = [1.0 / (c[0] - last + 1) ** power for c in candidates]
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

def _sa_optimize(rows, k, deadline, rng):
    state = SAState(rows, k, 200)
    best_rows = [list(r) for r in state.rows]
    best_cost = state.get_cost()
    best_scope = state.scope
    
    T = 5.0
    cooling = 0.9999
    min_T = 0.01
    n_rows = len(rows)
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = state.rows[i][j]
        prev_val = state.rows[i][j-1]
        next_val = state.rows[i][j+1] if j < k else state.scope + 100
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
            
        # Propose new value
        if rng.random() < 0.6:
            # Try to move down (reduce scope)
            if old_val > lo:
                # Random step down
                step = rng.randint(1, max(1, (old_val - lo) // 2))
                new_val = old_val - step
            else:
                new_val = old_val
        else:
            # Random perturbation
            span = hi - lo
            if span == 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.3)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
            
        if new_val == old_val:
            continue
            
        old_cost = state.get_cost()
        delta_cost, success = state.try_move(i, j, new_val)
        
        if not success:
            continue
            
        new_cost = old_cost + delta_cost
        
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, 0.01)):
            # Accept
            if state.collisions == 0:
                if state.scope < best_cost: # best_cost is actually best scope if collisions 0
                    best_cost = state.scope
                    best_rows = [list(r) for r in state.rows]
        else:
            # Revert
            state.revert_move(i, j, new_val, old_val)
            
        T = max(T * cooling, min_T)

    return best_cost, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    # Use a fraction of time for initialization
    init_deadline = start_time + 10.0
    
    candidates = []
    while time.time() < init_deadline:
        aggr = rng.uniform(0.2, 1.0)
        rows = _greedy_random(n_rows, k, rng, aggressiveness=aggr)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    if not candidates:
        # Fallback to simple greedy if random failed
        rows = _greedy_random(n_rows, k, rng, aggressiveness=0.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            # Should not happen, but return something
            return [[0, 1, 2, 3, 4, 5]] * n_rows

    # Phase 2: Optimize the best few candidates
    # Sort by scope
    candidates.sort(key=lambda x: x[0])
    
    # Take top 3 distinct structures if possible
    top_candidates = candidates[:3]
    
    remaining_time = deadline - time.time()
    time_per_sa = remaining_time / len(top_candidates)
    
    for i, (s, rows) in enumerate(top_candidates):
        if time.time() >= deadline - 5:
            break
            
        sa_deadline = time.time() + time_per_sa * 0.9
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            actual_scope = _scope(sa_rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = sa_rows

    return best_rows
