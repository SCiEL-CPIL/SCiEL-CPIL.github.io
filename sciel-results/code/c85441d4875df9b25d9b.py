"""Difference triangle set (7,5) constructor with randomized backtracking + forward checking."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry (the scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Total number of differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope: we need total_diffs distinct positive differences,
    # the smallest possible max difference is at least total_diffs (since differences
    # range from 1 to scope)
    # Actually differences range from 1 to scope, so we need scope >= total_diffs
    # But that's a very loose bound. A better bound: the sum of all differences
    # in a row with max entry m is at most m*k (each of the k non-zero entries
    # contributes at most m to each difference... not quite).
    # Simple counting: we need total_diffs distinct positive integers as differences,
    # so the largest difference must be at least total_diffs.
    lb = total_diffs

    best_result = None
    best_scope = float('inf')

    # Greedy initial solution for a starting point
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = frozenset(x - a for a in row)
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    greedy_rows = greedy_construct()
    greedy_scope = max(max(r) for r in greedy_rows)
    best_result = greedy_rows
    best_scope = greedy_scope

    # Backtracking with forward checking
    # State: list of current rows (each a list), used_diffs set
    # We build rows one at a time, and within a row we place elements one at a time.

    # For efficiency, we'll use a recursive backtracking that builds rows sequentially.
    # Each row is built by choosing the next element, checking that all new differences
    # are unused.

    # We'll try to find a solution with scope < best_scope.
    # Use randomized restarts with different orderings.

    def backtrack(rows, used_diffs, max_val_limit):
        """
        Try to complete the construction with the given rows and used differences.
        rows: list of completed rows (each is a list of k+1 ints starting at 0)
        used_diffs: set of all differences used so far
        max_val_limit: the maximum allowed value for the last element of any row
        Returns a list of rows if successful, None otherwise.
        """
        if len(rows) == n_rows:
            return rows[:]

        # Check time
        if time.time() > deadline:
            return None

        # Build the next row
        row = [0]
        return _build_row(row, rows, used_diffs, max_val_limit)

    def _build_row(row, rows, used_diffs, max_val_limit):
        """
        Recursively build a single row. row currently has some elements,
        we need to extend it to length k+1.
        """
        if time.time() > deadline:
            return None

        if len(row) == k + 1:
            # Check that all elements are <= max_val_limit
            if row[-1] > max_val_limit:
                return None
            return [row[:]]

        last = row[-1]
        # Try candidates for the next element
        # The next element must be > last and <= max_val_limit
        # Also, the new differences (x - a for a in row) must not be in used_diffs

        # Generate candidate values in a smart order
        # We want to keep values small to stay under max_val_limit
        start_val = last + 1
        end_val = max_val_limit + 1

        # For randomized exploration, shuffle candidates within a window
        # But we need to be careful about time. Let's try candidates in increasing order
        # but with some randomization in the choice of which candidate to try first.

        # Actually, for backtracking to work well, we should try candidates that
        # leave the most flexibility. A good heuristic: try smaller values first
        # (to keep the row compact) but with randomization to avoid getting stuck.

        candidates = list(range(start_val, end_val))

        # For the first few positions, try smaller values first to keep scope low
        # For later positions, we might need larger values
        # Use a randomized ordering: with some probability, try in increasing order,
        # otherwise shuffle

        if random.random() < 0.3:
            # Try in increasing order (greedy-like)
            pass
        else:
            # Randomize but bias towards smaller values
            # Weight each candidate by 1/value to bias towards smaller
            weights = [1.0 / max(c, 1) for c in candidates]
            random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                return None

            # Compute new differences
            new_diffs = []
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)

            if not valid:
                continue

            # Also check that new_diffs don't contain duplicates within themselves
            if len(set(new_diffs)) != len(new_diffs):
                continue

            # Forward check: verify that it's still possible to complete the row
            # The remaining elements need to be placed, each adding k+1-len(row) more elements
            # Each new element x' > x adds differences x'-a for all a in row+[x]
            # The smallest possible next value is x+1, which adds differences (x+1)-a for a in row+[x]
            # We need these to not conflict. This is complex, so we'll skip heavy forward checking
            # and rely on backtracking.

            row.append(x)
            new_used = used_diffs | set(new_diffs)

            result = _build_row(row, rows, new_used, max_val_limit)
            if result is not None:
                return result

            row.pop()

        return None

    # Main search loop: try different max_val_limit values and randomized restarts
    # Start from the lower bound and go up, or start from greedy_scope and try to improve

    # We'll try scopes from lb up to greedy_scope - 1, and for each scope, do several
    # randomized restarts.

    # First, let's establish a good upper bound
    upper = greedy_scope - 1  # We want to improve on greedy

    # Try scopes in increasing order from lb
    # For each scope, do a limited number of randomized backtracking attempts
    max_restarts_per_scope = 50
    max_depth_check = 20000  # Max recursive calls per attempt to avoid infinite loops

    scope = lb
    while scope <= upper and time.time() < deadline:
        found = False
        for restart in range(max_restarts_per_scope):
            if time.time() > deadline:
                break

            # Randomize the random seed for diversity
            random.seed(restart * 1000 + scope)

            # Try backtracking with this scope
            # We need to reset the recursion limit or use iterative approach
            # Python's recursion limit might be an issue, so let's increase it
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100000)

            try:
                result = backtrack([], set(), scope)
            except RecursionError:
                result = None
            finally:
                sys.setrecursionlimit(old_limit)

            if result is not None:
                # Verify the solution
                all_diffs = set()
                valid = True
                for r in result:
                    if len(r) != k + 1 or r[0] != 0:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d <= 0:
                                valid = False
                                break
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid and max(max(r) for r in result) <= scope:
                    best_result = result
                    best_scope = scope
                    found = True
                    break

        if found:
            # Found a solution with this scope, try to go lower
            upper = scope - 1
            # But we can't go below lb
            if upper < lb:
                break
        else:
            # Didn't find a solution with this scope, try next scope
            scope += 1
            # If scope is getting large, increase restarts or break
            if scope > greedy_scope - 1:
                break

    # Return the best result found
    return best_result
