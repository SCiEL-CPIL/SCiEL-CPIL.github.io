"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on scope: need total_diffs distinct positive integers
    # The smallest total_diffs distinct positive integers are 1, 2, ..., total_diffs
    # But they must be representable as differences within rows.
    # A simple counting bound: if max entry is M, the differences in one row
    # are at most M. We need total_diffs distinct differences, each <= M.
    # So M >= total_diffs. But that's not tight.
    # Better: the differences in a row with max M are subsets of {1,...,M}.
    # We need total_diffs distinct values from {1,...,M}, so M >= total_diffs.
    # Actually, the counting bound given is 105 for (7,5).
    # total_diffs = 7 * 5 * 6 / 2 = 105. So M >= 105.

    # We'll use a randomized local search / greedy hybrid approach.
    # Strategy:
    # 1. Start with a greedy construction to get a valid solution.
    # 2. Try to improve (reduce scope) using local search:
    #    - Randomly perturb rows and try to fix conflicts while reducing max entry.

    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            if len(row_diffs) != len(row):  # Should be k*(k+1)//2
                pass
            if all_diffs & row_diffs:
                return False
            all_diffs |= row_diffs
        return True

    def scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)

    # Greedy construction
    def greedy_construct(n_rows, k, max_scope=None):
        """
        Greedy: build each row by picking the smallest next value
        that doesn't create a difference conflict.
        If max_scope is given, only consider values up to max_scope.
        Returns None if no valid row can be constructed within max_scope.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                if max_scope is None:
                    candidate_max = 200  # Arbitrary large
                else:
                    candidate_max = max_scope
                for x in range(row[-1] + 1, candidate_max + 1):
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # First, get a baseline greedy solution
    best_rows = greedy_construct(n_rows, k)
    if best_rows is None:
        # Should not happen for reasonable inputs
        # Fallback: return a trivial valid solution with large scope
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set(x - a for a in row)
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_scope = scope(best_rows)

    # Try to improve using local search
    # We'll use a simulated annealing-like approach with random restarts.

    def random_row(k, max_val):
        """Generate a random strictly increasing row starting at 0, with max entry <= max_val."""
        if max_val < k:
            return None
        # Choose k values from 1..max_val
        vals = sorted(random.sample(range(1, max_val + 1), k))
        return [0] + vals

    def repair_and_improve(rows, max_scope, time_limit):
        """
        Try to repair rows to be valid within max_scope, then try to reduce scope.
        Returns (valid_rows, scope) or (None, None).
        """
        rows = [list(r) for r in rows]
        current_scope = max_scope
        used = set()
        
        # First, try to make all rows valid within current_scope
        for _ in range(1000):  # Iteration limit
            if time.time() > time_limit:
                return None, None
            valid = True
            for i in range(n_rows):
                row_diffs = get_diffs(rows[i])
                if row_diffs & used:
                    valid = False
                    # Try to regenerate this row
                    new_row = random_row(k, current_scope)
                    if new_row is not None:
                        rows[i] = new_row
                    break
            if valid:
                # Check all differences are distinct
                all_diffs = set()
                for row in rows:
                    rd = get_diffs(row)
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs |= rd
                if valid:
                    return rows, scope(rows)
        
        return None, None

    # Local search: try to reduce scope
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try to find a solution with scope = best_scope - 1, best_scope - 2, etc.
        target_scope = best_scope - 1
        if target_scope < total_diffs:
            break
        
        found = False
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Generate random rows within target_scope
            rows = []
            valid = True
            used = set()
            for i in range(n_rows):
                row = random_row(k, target_scope)
                if row is None:
                    valid = False
                    break
                rd = get_diffs(row)
                if rd & used:
                    valid = False
                    break
                used |= rd
                rows.append(row)
            
            if valid:
                # Verify
                all_diffs = set()
                for row in rows:
                    rd = get_diffs(row)
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs |= rd
                if valid:
                    new_scope = scope(rows)
                    if new_scope < best_scope:
                        best_rows = rows
                        best_scope = new_scope
                    found = True
                    break
        
        if not found:
            # Try a different approach: start from best_rows and perturb
            rows = [list(r) for r in best_rows]
            for trial in range(100):
                if time.time() > deadline:
                    break
                
                # Pick a random row and perturb it
                i = random.randrange(n_rows)
                old_row = rows[i]
                new_row = random_row(k, target_scope)
                if new_row is None:
                    continue
                rows[i] = new_row
                
                # Check validity
                all_diffs = set()
                valid = True
                for j in range(n_rows):
                    rd = get_diffs(rows[j])
                    if rd & all_diffs:
                        valid = False
                        break
                    all_diffs |= rd
                
                if valid:
                    new_scope = scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        target_scope = best_scope - 1
                        if target_scope < total_diffs:
                            break
                        break
    
    # Final verification
    all_diffs = set()
    for row in best_rows:
        rd = get_diffs(row)
        if rd & all_diffs:
            # Should not happen, but fallback
            break
        all_diffs |= rd
    
    return best_rows

# EVOLVE-BLOCK-END
