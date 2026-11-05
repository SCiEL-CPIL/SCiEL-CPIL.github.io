import time
import random
import heapq

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin from 150s limit
    
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used
    
    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def _bitset_from_set(s):
        b = 0
        for x in s:
            b |= (1 << x)
        return b
    
    def _greedy_initial(n_rows, k, seed=42):
        """Fast greedy to get a valid starting point for SA."""
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                # Try to find a valid x starting from last+1
                # We limit search a bit to keep it fast, but usually it's tight
                x = row[-1] + 1
                while True:
                    # Check conflicts
                    conflict = False
                    new_diffs_bits = 0
                    for a in row:
                        d = x - a
                        if d <= 0: continue
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                        new_diffs_bits |= (1 << d)
                    
                    if not conflict:
                        row.append(x)
                        used_bits |= new_diffs_bits
                        found = True
                        break
                    x += 1
                    if x > 200: # Safety break, shouldn't happen for small k
                        return None
            rows.append(row)
        return rows

    def _sa_optimize(rows, n_rows, k, deadline):
        """Simulated Annealing on the solution structure."""
        if not rows:
            return rows
        
        best_rows = [r[:] for r in rows]
        best_scope = _scope(best_rows)
        
        # Precompute difference masks for each row position? No, rows change.
        # We maintain `used_bits` for the current state.
        used_bits = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    used_bits |= (1 << d)
        
        current_scope = best_scope
        T = 10.0
        cooling = 0.9999
        
        rng = random.Random(123)
        
        while time.time() < deadline:
            if T < 0.01:
                break
                
            # Pick a random row and position to move
            ri = rng.randint(0, n_rows - 1)
            pi = rng.randint(1, k) # Position 1..k (0 is fixed at 0)
            
            old_val = rows[ri][pi]
            
            # Determine valid range for new value
            low = rows[ri][pi-1] + 1
            high = rows[ri][pi+1] - 1 if pi < k else current_scope
            
            if low > high:
                continue
            
            # Pick a new value
            new_val = rng.randint(low, high)
            if new_val == old_val:
                continue
            
            # Calculate change in used_bits
            # Remove old diffs involving pi
            # Add new diffs involving pi
            
            # Optimization: Instead of recalculating all diffs, just check conflicts for the new value
            # and update the bitset incrementally if valid.
            
            # 1. Check if new_val creates any conflict with existing used_bits (excluding old_val's contribution)
            # We need to remove old_val's contribution from used_bits first to check properly,
            # or just check if (new_diffs & (used_bits ^ old_diffs)) == 0
            
            old_diffs_bits = 0
            for i in range(pi):
                d = old_val - rows[ri][i]
                old_diffs_bits |= (1 << d)
            for i in range(pi+1, len(rows[ri])):
                d = rows[ri][i] - old_val
                old_diffs_bits |= (1 << d)
                
            new_diffs_bits = 0
            for i in range(pi):
                d = new_val - rows[ri][i]
                new_diffs_bits |= (1 << d)
            for i in range(pi+1, len(rows[ri])):
                d = rows[ri][i] - new_val
                new_diffs_bits |= (1 << d)
                
            # Check conflict: new_diffs_bits should not overlap with (used_bits without old_diffs_bits)
            # used_bits_without_old = used_bits & ~old_diffs_bits  (Bitwise NOT is tricky with infinite bits, but Python handles it)
            # Actually, simpler: 
            # Conflict if (new_diffs_bits & (used_bits ^ old_diffs_bits)) != 0
            # Note: XOR works if old_diffs_bits is a subset of used_bits, which it is.
            
            conflict = new_diffs_bits & (used_bits ^ old_diffs_bits)
            
            if conflict:
                continue
            
            # Valid move
            # Apply move
            rows[ri][pi] = new_val
            used_bits = (used_bits ^ old_diffs_bits) | new_diffs_bits
            
            new_scope = _scope(rows)
            
            # Acceptance criterion
            delta_scope = new_scope - current_scope
            
            if delta_scope < 0:
                # Improvement
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
            else:
                # Worse or equal
                # Accept with probability exp(-delta / T)
                # If delta is large, prob is small.
                if rng.random() < pow(2.718281828, -delta_scope / T):
                    current_scope = new_scope
                else:
                    # Reject: revert
                    rows[ri][pi] = old_val
                    used_bits = (used_bits ^ new_diffs_bits) | old_diffs_bits
                    current_scope = _scope(best_rows) # Revert scope too? No, current_scope tracks current state.
                    # Wait, if we reject, current_scope should go back to previous value.
                    # But we didn't store previous current_scope explicitly, it was derived from rows.
                    # Let's just recalculate or store it.
                    # To avoid expensive _scope, we can track it.
                    # But _scope is O(N*K^2) which is small.
                    pass # current_scope is already correct because we didn't update it on rejection? 
                    # No, we only update current_scope on acceptance. 
                    # So on rejection, current_scope remains the value BEFORE the move was considered?
                    # No, current_scope was updated in the previous step.
                    # If we reject, the state is reverted, so current_scope should be the value of the reverted state.
                    # Since we didn't change rows, current_scope is still the value of the current rows.
                    # So no action needed for current_scope on rejection.
            
            T *= cooling
        
        return best_rows

    # 1. Generate initial solution
    best_rows = _greedy_initial(n_rows, k, seed=42)
    if not best_rows:
        # Fallback to simple greedy if initial fails
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            best_rows.append(row)
    
    best_scope = _scope(best_rows)
    
    # 2. Run Simulated Annealing
    # We can run multiple SA runs with different seeds if time permits, but one long run is usually better.
    # Let's run SA for the remaining time.
    if time.time() < deadline - 1:
        polished = _sa_optimize(best_rows, n_rows, k, deadline - 1)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
            
    return best_rows
