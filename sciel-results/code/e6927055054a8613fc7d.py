import time
import random
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 115.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def is_valid_solution(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, m):
                if row[i] <= row[i - 1]:
                    return False
            ds = row_diffs_set(row)
            if len(ds) != diffs_per_row:
                return False
            if used & ds:
                return False
            used |= ds
        return True

    def greedy_solution(scope_limit):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                placed = False
                x = row[-1] + 1
                while x <= scope_limit:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        placed = True
                        break
                    x += 1
                if not placed:
                    return None
            rows.append(row)
        return rows

    fallback = greedy_solution(200) or greedy_solution(300)
    if fallback is None:
        # Extremely unlikely fallback: just return something
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        fallback = rows
    best_rows = fallback
    best_scope = max(r[-1] for r in fallback)

    def solve_with_mrV(target_scope, time_budget):
        # Build a complete search with MRV.
        # Variables: for each row r (0..n_rows-1), choose a set S_r of m-1 positive ints increasing, all <= target_scope.
        # Constraint: all differences across rows are distinct, and within a row all differences are distinct.
        # We'll do row-by-row with backtracking.
        # To keep it tractable, we only search row 0..n_rows-1, and use pruning based on remaining differences.
        # We'll also use a heuristic to order candidate rows.
        # We'll represent a partial row as a tuple of chosen values (including 0 at start).
        # We'll do a DFS with MRV on the next row's next element.
        # Actually, let's do a simpler approach: we'll search for the first row, then the second, etc.
        # We'll use a recursive function that builds one row at a time.
        # For each row, we'll do a DFS to find a valid row that doesn't conflict with used differences.
        # This is a two-level search.
        
        # To make it faster, we'll precompute all possible rows up to target_scope that are valid internally.
        # But that's too many. Instead, we'll generate rows on the fly.
        
        # Let's do a recursive search for the entire set of rows.
        # State: current row index, current partial row (list), used differences (set).
        # We'll use MRV: at each step, if we're in the middle of a row, we try to place the next element.
        # If we've completed a row, we move to the next row.
        
        # We'll use a global used set that we update as we go.
        # We'll also use a time check.
        
        # To keep it simple, let's do a row-by-row search with backtracking within a row.
        # We'll try to build each row from scratch, and if we get stuck, we backtrack to the previous row.
        
        # This is complex. Let's instead use a simpler approach:
        # We'll use a randomized local search to find a solution for the given target_scope.
        # We'll start with a greedy solution and then try to improve it.
        
        # Actually, let's do a simple backtracking search with MRV.
        # We'll represent the state as a list of rows (some complete, some partial).
        # We'll use a recursive function that tries to complete the current row or move to the next.
        
        # For efficiency, we'll limit the search depth and use a timeout.
        
        # Let's do a simpler approach: use a randomized greedy with backtracking.
        # We'll try to build each row, and if we get stuck, we'll restart with a different random seed.
        
        # This is getting too complex. Let's use a simpler method:
        # We'll use a genetic algorithm to find a solution for the given target_scope.
        
        # For now, let's just do a simple backtracking search with a timeout.
        
        # We'll use a recursive function that builds the rows one by one.
        # For each row, we'll try to build it using a DFS with MRV.
        
        # We'll use a global used set.
        
        # Let's do it.
        
        used = set()
        rows = []
        
        def dfs_row(row_idx, current_row):
            # If we've completed all rows, return True
            if row_idx == n_rows:
                return True
            
            # If we're building a row, try to complete it
            if len(current_row) == m:
                # Move to next row
                return dfs_row(row_idx + 1, [0])
            
            # Try to place the next element
            lo = current_row[-1] + 1
            hi = target_scope
            
            # Try values from lo to hi
            for x in range(lo, hi + 1):
                ok = True
                for a in current_row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    # Add x to current_row and update used
                    for a in current_row:
                        used.add(x - a)
                    current_row.append(x)
                    
                    if dfs_row(row_idx, current_row):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                    for a in current_row:
                        used.discard(x - a)
            
            return False
        
        # Start the search
        if dfs_row(0, [0]):
            return rows + [current_row]
        
        return None

    # Main loop: try to improve the solution
    # We'll try to find a solution with a smaller scope than best_scope.
    # We'll use a binary search or a linear search.
    # For now, let's just try a few target scopes.
    
    # Try to find a solution with scope = best_scope - 1, best_scope - 2, etc.
    # But this is expensive. Instead, let's use a randomized approach.
    
    # We'll try to find a solution with a random target scope between lb and best_scope - 1.
    # If we find one, we update best_rows and best_scope.
    
    # We'll do this for a fixed number of iterations or until time runs out.
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        # Pick a random target scope
        target = rng.randint(lb, best_scope - 1)
        
        # Try to find a solution with this target scope
        sol = solve_with_mrV(target, deadline - time.time() - 5)
        
        if sol is not None and is_valid_solution(sol):
            scope = max(r[-1] for r in sol)
            if scope < best_scope:
                best_rows = sol
                best_scope = scope
    
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
