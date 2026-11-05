"""Difference triangle set (7,5): minimize the largest entry.
Constraint-satisfaction solver with bitset difference tracking,
forward checking, and systematic backtracking with MRV heuristics."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs_set(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _solve_constraint(n_rows, k, max_val, deadline, rng):
    """
    Solve the constraint satisfaction problem using backtracking with
    forward checking and MRV heuristics.
    
    We build rows one at a time. For each row, we choose k+1 values starting at 0,
    strictly increasing, all <= max_val, such that all differences are globally unique.
    
    Uses bitset (Python int) for fast difference tracking.
    """
    # Bitset of used differences: bit d is set if difference d is used
    used_bits = 0
    max_diff = max_val * (max_val + 1) // 2  # theoretical max diff
    # Actually max diff is at most max_val (since values are 0..max_val)
    # So we need bits 0..max_val
    used_bits = 0  # bit i set means difference i is used
    
    rows = []
    
    def _can_add_diff(d):
        """Check if difference d is not yet used."""
        return (used_bits >> d) & 1 == 0
    
    def _mark_diff(d):
        """Mark difference d as used."""
        nonlocal used_bits
        used_bits |= (1 << d)
    
    def _unmark_diff(d):
        """Unmark difference d."""
        nonlocal used_bits
        used_bits &= ~(1 << d)
    
    def _build_row(row_idx, deadline):
        """
        Build a single row of k+1 elements starting at 0.
        Uses backtracking with MRV-like heuristics.
        Returns row or None if infeasible.
        """
        if time.time() > deadline:
            return None
        
        row = [0]
        # Track which differences this row uses (for backtracking)
        row_diffs = []  # list of sets of diffs added at each step
        
        def backtrack(pos):
            """Build the row from position pos onwards (row[0]=0 already set)."""
            if time.time() > deadline:
                return False
            
            if pos == k + 1:
                return True
            
            # The next value must be > row[-1] and <= max_val
            # Also, for each candidate value v, all diffs (v - row[i] for i < pos) must be unused
            low = row[-1] + 1
            high = max_val
            
            # Additional constraint: we need to be able to fill remaining positions
            # remaining positions: k + 1 - pos - 1 = k - pos
            # Each needs at least 1 more than previous, so minimum final value is low + (k - pos)
            # This gives us an upper bound on how large low can be
            min_final = low + (k - pos)
            if min_final > max_val:
                return False
            
            # Generate candidate values
            # For efficiency, we'll try values in a smart order
            # Prefer values that create "rare" differences (less likely to conflict later)
            
            # First, find all valid candidates
            valid_candidates = []
            for v in range(low, high + 1):
                # Check if all new diffs are available
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = v - row[i]
                    if d > max_val:
                        ok = False
                        break
                    if (used_bits >> d) & 1:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    valid_candidates.append((v, new_diffs))
            
            if not valid_candidates:
                return False
            
            # Order candidates: try those that leave more flexibility
            # Heuristic: prefer values whose new differences are "spread out" 
            # or prefer smaller values (to keep scope small)
            # For now, try smaller values first (minimizes scope)
            valid_candidates.sort(key=lambda x: x[0])
            
            for v, new_diffs in valid_candidates:
                if time.time() > deadline:
                    return False
                
                # Mark the new diffs
                for d in new_diffs:
                    _mark_diff(d)
                
                row.append(v)
                row_diffs.append(new_diffs)
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                row.pop()
                row_diffs.pop()
                for d in new_diffs:
                    _unmark_diff(d)
            
            return False
        
        if backtrack(1):
            return row
        return None
    
    # Build all rows
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = _build_row(r, deadline)
        if row is None:
            return None
        
        # Mark all differences in this row as used
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                _mark_diff(d)
        
        rows.append(row)
    
    return rows


def _solve_with_randomization(n_rows, k, max_val, deadline, rng, n_attempts=5):
    """
    Try multiple randomized attempts to solve the CSP.
    Randomization: shuffle the order in which rows are built,
    and use randomized candidate ordering.
    """
    best_rows = None
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Randomize row order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        # We'll build rows in the shuffled order
        # But the constraint is symmetric, so this should be fine
        
        used_bits = 0
        rows_built = {}
        
        def _can_add_diff(d):
            return (used_bits >> d) & 1 == 0
        
        def _mark_diff(d):
            nonlocal used_bits
            used_bits |= (1 << d)
        
        def _unmark_diff(d):
            nonlocal used_bits
            used_bits &= ~(1 << d)
        
        def _build_row_randomized(row_idx, deadline):
            """Build a row with randomized candidate ordering."""
            if time.time() > deadline:
                return None
            
            row = [0]
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                
                if pos == k + 1:
                    return True
                
                low = row[-1] + 1
                high = max_val
                
                min_final = low + (k - pos)
                if min_final > max_val:
                    return False
                
                valid_candidates = []
                for v in range(low, high + 1):
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = v - row[i]
                        if d > max_val:
                            ok = False
                            break
                        if (used_bits >> d) & 1:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        valid_candidates.append((v, new_diffs))
                
                if not valid_candidates:
                    return False
                
                # Randomize order with bias toward smaller values
                # Use a weighted shuffle: smaller values get higher priority
                rng.shuffle(valid_candidates)
                # But also try smaller values first with some probability
                if rng.random() < 0.7:
                    valid_candidates.sort(key=lambda x: x[0])
                
                for v, new_diffs in valid_candidates:
                    if time.time() > deadline:
                        return False
                    
                    for d in new_diffs:
                        _mark_diff(d)
                    
                    row.append(v)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    row.pop()
                    for d in new_diffs:
                        _unmark_diff(d)
                
                return False
            
            if backtrack(1):
                return row
            return None
        
        # Build rows in randomized order
        success = True
        for r in row_order:
            if time.time() > deadline:
                success = False
                break
            
            row = _build_row_randomized(r, deadline)
            if row is None:
                success = False
                break
            
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    _mark_diff(d)
            
            rows_built[r] = row
        
        if success and len(rows_built) == n_rows:
            # Reorder rows to original order
            result = [rows_built[i] for i in range(n_rows)]
            if best_rows is None or _scope(result) < _scope(best_rows):
                best_rows = result
    
    return best_rows


def _local_search_reduce(rows, k, deadline, rng):
    """
    Local search to reduce the maximum entry.
    For each element that contributes to the max, try to reduce it.
    """
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    used = set()
    for row in rows:
        used |= _row_diffs_set(row)
    
    current_scope = best_scope
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline and T > 0.01:
        # Find elements at or near the maximum
        max_elements = []
        for ri, row in enumerate(rows):
            for pos in range(1, len(row)):
                if row[pos] >= current_scope - 3:
                    max_elements.append((ri, pos))
        
        if not max_elements:
            break
        
        # Pick a random max element
        ri, pos = rng.choice(max_elements)
        row = rows[ri]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = old_val - 1
        if pos + 1 < len(row):
            high = min(high, row[pos + 1] - 1)
        
        if low > high:
            T *= cooling
            continue
        
        # Try candidate values, preferring smaller
        n_candidates = min(15, high - low + 1)
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        candidates = candidates[:n_candidates]
        candidates.sort()  # try smallest first
        
        moved = False
        for new_val in candidates:
            if new_val >= old_val:
                continue
            
            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            # Check for conflicts
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            # Accept
            row[pos] = new_val
            used -= old_diffs
            used |= new_diffs
            
            new_scope = _scope(rows)
            if new_scope < current_scope:
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            
            moved = True
            break
        
        if not moved:
            T *= cooling
    
    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    rng = random.Random(12345)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try to solve at progressively lower max_val bounds
    # Start from a reasonable upper bound and go down
    # The counting bound is 105, best known is 112
    # Try: 112, 111, 110, 109, 108, 107, 106, 105
    
    # First, get a quick solution using a higher bound
    for max_val in [115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105]:
        if time.time() > deadline - 20.0:
            break
        
        # Try the systematic solver first
        rows = _solve_constraint(n_rows, k, max_val, deadline - 20.0, rng)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if max_val <= 105:
                    break  # Hit counting bound
        
        # If that failed or found a worse solution, try randomized
        if time.time() < deadline - 20.0:
            rows2 = _solve_with_randomization(
                n_rows, k, max_val, deadline - 20.0, rng, n_attempts=3
            )
            if rows2 is not None:
                scope2 = _scope(rows2)
                if scope2 < best_scope:
                    best_scope = scope2
                    best_rows = rows2
                    if max_val <= 105:
                        break
    
    # Phase 2: Local search to reduce the scope further
    if best_rows is not None and time.time() < deadline - 10.0:
        improved = _local_search_reduce(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 3: More local search iterations
    if best_rows is not None and time.time() < deadline - 5.0:
        improved2 = _local_search_reduce(best_rows, k, deadline - 5.0, rng)
        new_scope2 = _scope(improved2)
        if new_scope2 < best_scope:
            best_rows = improved2
            best_scope = new_scope2
    
    # Final validation
    if best_rows is not None:
        if not _valid(best_rows):
            # Should not happen, but fallback
            # Generate a simple valid solution
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    row.append(x)
                    used |= new_diffs
                best_rows.append(row)
    
    return best_rows

# EVOLVE-BLOCK-END
