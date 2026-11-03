"""Constraint-based backtracking search with forward checking for difference triangle sets."""
import sys
import time
from itertools import combinations

class BacktrackingSolver:
    """Solves difference triangle set problem using backtracking with constraint propagation."""
    
    def __init__(self, n_rows, k, max_scope, time_limit=110.0):
        self.n_rows = n_rows
        self.k = k
        self.max_scope = max_scope
        self.time_limit = time_limit
        self.start_time = time.time()
        self.best_rows = None
        self.best_score = max_scope + 1
        self.nodes_explored = 0
        
    def _check_timeout(self):
        if time.time() - self.start_time > self.time_limit:
            return True
        return False
    
    def _row_diffs(self, row):
        """Compute all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def _is_row_valid(self, row, used_diffs):
        """Check if a row is valid (strictly increasing, starts at 0, no internal diff conflicts)."""
        if not row or row[0] != 0:
            return False, None
        if len(row) != self.k + 1:
            return False, None
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False, None
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False, None
                if d in used_diffs:
                    return False, None
                diffs.add(d)
        return True, diffs
    
    def _can_extend_row(self, partial_row, used_diffs):
        """Check if partial row can be extended to a full valid row."""
        if len(partial_row) >= self.k + 1:
            return False
        last_val = partial_row[-1] if partial_row else 0
        # Minimum possible next value
        min_next = last_val + 1
        # Maximum possible value
        max_val = self.max_scope
        if min_next > max_val:
            return False
        # Check if we can still fit remaining elements
        remaining = self.k + 1 - len(partial_row)
        if max_val - min_next + 1 < remaining:
            return False
        return True
    
    def _generate_candidates_for_position(self, partial_row, pos, used_diffs):
        """Generate candidate values for position pos in partial_row, ordered by heuristic."""
        if pos == 0:
            return [0]
        prev_val = partial_row[-1] if partial_row else 0
        min_val = prev_val + 1
        max_val = self.max_scope
        
        candidates = []
        for val in range(min_val, max_val + 1):
            # Quick check: does this create a diff conflict with existing elements?
            new_diffs = set()
            conflict = False
            for i, a in enumerate(partial_row):
                d = val - a
                if d <= 0:
                    conflict = True
                    break
                if d in new_diffs or d in used_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            if not conflict:
                candidates.append(val)
        
        # Heuristic: prefer values that leave more room for future elements
        # Sort by how many valid extensions are possible (approximate)
        if candidates:
            # Prefer values closer to the middle to leave room on both sides
            mid = (min_val + max_val) // 2
            candidates.sort(key=lambda v: abs(v - mid))
        
        return candidates
    
    def _solve(self, row_idx, partial_row, used_diffs, rows):
        """
        Backtracking solver.
        row_idx: which row we're currently building
        partial_row: the partial row being built
        used_diffs: set of all differences used so far
        rows: list of completed rows
        """
        if self._check_timeout():
            return False
        
        self.nodes_explored += 1
        
        # If we've completed all rows, check if this is better
        if row_idx == self.n_rows:
            score = max(max(r) for r in rows)
            if score < self.best_score:
                self.best_score = score
                self.best_rows = [r[:] for r in rows]
            return True
        
        # If partial_row is complete, move to next row
        if len(partial_row) == self.k + 1:
            return self._solve(row_idx + 1, [], used_diffs, rows)
        
        # Try to extend the current row
        if not self._can_extend_row(partial_row, used_diffs):
            return False
        
        pos = len(partial_row)
        candidates = self._generate_candidates_for_position(partial_row, pos, used_diffs)
        
        for val in candidates:
            if self._check_timeout():
                return False
            
            new_partial = partial_row + [val]
            valid, new_diffs = self._is_row_valid(new_partial, used_diffs)
            
            if valid:
                new_used = used_diffs | new_diffs
                if self._solve(row_idx, new_partial, new_used, rows):
                    # Continue searching for better solutions
                    if self.best_score == self.max_scope:
                        # Found a solution at the limit, try to improve
                        pass
                    else:
                        # Found a better solution, keep searching
                        pass
            
            # If we found a solution and it's optimal (can't be better), stop
            if self.best_score <= self.max_scope:
                # Check if we can prove no better solution exists
                # For now, continue searching
                pass
        
        return False
    
    def solve(self):
        """Main solve method."""
        self.start_time = time.time()
        rows = []
        used_diffs = set()
        self._solve(0, [], used_diffs, rows)
        return self.best_rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all pairwise differences
    within rows are distinct across all rows.
    
    Uses constraint-based backtracking search with forward checking.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * n_rows] if k == 0 else []
    
    # Total number of differences needed
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope
    counting_bound = total_diffs
    
    # Try increasing scopes until we find a valid solution
    # Start from a reasonable lower bound
    # For (7,5): counting bound = 7*15 = 105, known best = 112
    start_scope = max(counting_bound, k * (k + 1) // 2 * n_rows)
    
    # Use a heuristic upper bound
    # Greedy construction gives an upper bound
    used = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        greedy_rows.append(row)
    
    greedy_scope = max(max(r) for r in greedy_rows)
    
    # Try scopes from lower bound to greedy scope
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Binary search or linear search for minimum scope
    # Try a few scopes near the lower bound
    scopes_to_try = []
    for s in range(start_scope, min(greedy_scope + 1, start_scope + 20)):
        scopes_to_try.append(s)
    
    # Also try the greedy scope as fallback
    if greedy_scope not in scopes_to_try:
        scopes_to_try.append(greedy_scope)
    
    for scope in scopes_to_try:
        if scope >= best_scope:
            break
        
        solver = BacktrackingSolver(n_rows, k, scope, time_limit=100.0)
        result = solver.solve()
        
        if result is not None:
            result_scope = max(max(r) for r in result)
            if result_scope < best_scope:
                best_scope = result_scope
                best_rows = result
            
            # If we found a solution at this scope, we can stop
            # (since we're trying scopes in increasing order)
            break
    
    return best_rows
