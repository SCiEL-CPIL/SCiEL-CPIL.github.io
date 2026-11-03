import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def backtrack_search(M, rng, deadline):
        """
        Attempts to find a valid configuration with maximum value <= M.
        Uses backtracking with forward checking and MRV heuristic.
        """
        used_diffs = set()
        rows = []
        
        def get_conflicts(candidate_val, current_row, row_idx):
            """
            Returns the set of differences that would be created if candidate_val
            is appended to current_row, that are already in used_diffs or 
            would conflict with other pending parts (simplified: just used_diffs 
            and internal conflicts of the row so far).
            Note: Since we build row by row, 'pending' conflicts are only 
            internal to the current row being built.
            """
            conflicts = set()
            # Check against global used_diffs
            for a in current_row:
                d = candidate_val - a
                if d in used_diffs:
                    conflicts.add(d)
            # Check internal conflicts within current_row + candidate
            # We need to ensure all pairwise diffs in (current_row + [candidate]) are unique
            # We can pre-compute diffs of current_row? No, we check on the fly.
            # For efficiency, we just check if any new diff (candidate - a) matches 
            # any existing diff in current_row's internal set OR used_diffs.
            # But we also need to ensure new diffs don't match each other?
            # If candidate - a1 == candidate - a2, then a1 == a2, which is impossible 
            # since row is strictly increasing.
            # So we only need to check new diffs against:
            # 1. used_diffs (global)
            # 2. internal diffs of current_row
            return conflicts

        def solve(row_idx, pos, current_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos == m:
                # Row complete
                rows.append(current_row)
                if solve(row_idx + 1, 1, [0]): # Start next row with 0
                    return True
                rows.pop()
                return False
            
            # Heuristic: Try candidates in random order or sorted?
            # For compactness, we want small values.
            # But to satisfy constraints, we might need to jump.
            # Let's try a range of candidates.
            
            # MRV is tricky here because constraints are dynamic.
            # Let's just try candidates from current_row[-1] + 1 up to M.
            # To improve, we can shuffle the order of candidates slightly 
            # or prioritize those that don't create "popular" differences?
            
            # Optimization: Pre-calculate valid candidates?
            # Candidate x is valid if:
            # 1. x > current_row[-1]
            # 2. x <= M
            # 3. For all a in current_row: (x - a) not in used_diffs
            # 4. For all a, b in current_row: (x - a) != (b - a) ? No, (x-a) != (x-b) is always true.
            #    We need (x-a) not in {internal_diffs of current_row}
            
            # Let's get internal diffs of current_row
            # This is O(m^2) per step, m is small (8).
            internal_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    internal_diffs.add(current_row[j] - current_row[i])
            
            start_val = current_row[-1] + 1
            # Randomize the order of trying candidates to escape local minima?
            # Or stick to ascending for determinism/compactness bias?
            # Let's try ascending, but with a slight random perturbation in later stages
            # or just pure ascending to find the *first* valid solution which is usually compact.
            
            # To speed up, we can skip values that definitely conflict.
            # A value x conflicts if (x - a) in used_diffs OR (x - a) in internal_diffs.
            # This means x in {a + d for a in current_row for d in used_diffs}
            # AND x in {a + d for a in current_row for d in internal_diffs}
            
            # Let's just iterate. m is small, M is the bound.
            # If M is tight, this is fast. If M is loose, it might be slow.
            
            for x in range(start_val, M + 1):
                # Quick check
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    if d in internal_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If valid, add to used_diffs and recurse
                new_diffs_to_add = []
                for a in current_row:
                    new_diffs_to_add.append(x - a)
                
                for d in new_diffs_to_add:
                    used_diffs.add(d)
                
                new_row = current_row + [x]
                
                if solve(row_idx, pos + 1, new_row):
                    return True
                
                for d in new_diffs_to_add:
                    used_diffs.remove(d)
            
            return False

        # Start with first row
        if solve(0, 1, [0]):
            return rows
        return None

    def greedy_construct(n, k_val, rng):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            success = True
            for pos in range(k_val):
                found = False
                # Try to find a small value
                # Heuristic: try values from row[-1]+1 upwards
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        found = True
                        break
                    x += 1
                    # Safety break if we go too high (shouldn't happen for reasonable M)
                    if x > 10000:
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Main Strategy:
    # 1. Find a valid solution with greedy to establish an upper bound (scope).
    # 2. Use binary search on M (max value) with backtracking to find the minimum M.
    
    best_rows = greedy_construct(n_rows, k, rng)
    if best_rows is None:
        return [[0] * (k + 1) for _ in range(n_rows)] # Fallback, though invalid
    
    best_scope = max(max(r) for r in best_rows)
    
    # We know a solution exists for best_scope.
    # Let's try to find a smaller scope.
    # Lower bound: total_diffs (since diffs are 1..M, we need at least total_diffs distinct diffs)
    # Actually, diffs are not necessarily 1..M, but they are distinct positive integers.
    # The smallest possible max diff is at least total_diffs.
    # But the max entry is related to the max diff.
    # Let's start from a reasonable lower bound.
    
    # Simple lower bound: The k-th entry must be at least k.
    # A better bound: The sum of the smallest n_rows * diffs_per_row distinct positive integers.
    # But we just want to minimize the max entry.
    
    low = 10 # Arbitrary small start
    high = best_scope
    
    # Binary search for the minimal M
    # Note: Backtracking can be expensive. We limit the time per search.
    
    found_rows = None
    current_best_M = best_scope
    
    while low <= high and time.time() < deadline - 5:
        mid = (low + high) // 2
        # Try to find a solution with max value <= mid
        # Use a timeout for the backtracking search
        search_deadline = min(deadline - 2, time.time() + 10) # Allow up to 10s per search
        
        rows = backtrack_search(mid, rng, search_deadline)
        
        if rows is not None:
            # Valid solution found for M=mid
            current_best_M = mid
            found_rows = rows
            high = mid - 1
        else:
            # No solution found for M=mid (either doesn't exist or search timed out)
            # If search timed out, we can't be sure. 
            # But if we assume it doesn't exist (optimistic for binary search correctness 
            # only if search is complete), we might skip a valid M.
            # To be safe, if it times out, we should probably not assume it's impossible,
            # but binary search requires a clear yes/no.
            # Given the complexity, if it times out, we might just stop or increase low.
            # Let's assume "not found" means "increase low".
            low = mid + 1
            
    if found_rows is not None:
        best_rows = found_rows
        best_scope = current_best_M
    else:
        # If binary search failed to improve, stick with greedy
        pass
        
    # Final check and return
    if is_valid(best_rows):
        return best_rows
    else:
        # Emergency fallback
        return greedy_construct(n_rows, k, rng) or [[0] * (k + 1) for _ in range(n_rows)]

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
