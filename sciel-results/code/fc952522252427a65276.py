import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # k=8 means 9 marks per row (0 to k)
    # n_rows=4
    # We want to minimize the max entry (scope M)
    
    # Lower bound estimation:
    # Total diffs = n_rows * k * (k+1) / 2
    # Diffs are in 1..M
    # n_rows * k * (k+1) / 2 <= M * (M-1) / 2 is not quite right, 
    # it's that we need to pick distinct diffs from 1..M.
    # Actually, the diffs in each row are distinct from each other and from other rows.
    # So we need n_rows * k * (k+1) / 2 distinct differences from the range [1, M].
    # So M * (M-1) / 2 >= n_rows * k * (k+1) / 2
    # M * (M-1) >= n_rows * k * (k+1)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Solve M^2 - M - total_diffs_needed >= 0
    # M = (1 + sqrt(1 + 4 * total_diffs_needed)) / 2
    import math
    lb = math.ceil((1 + math.sqrt(1 + 4 * total_diffs_needed)) / 2)
    
    # We can start slightly higher if lb is too tight due to structure
    # But let's try lb.
    
    # For n=4, k=8: total_diffs = 4 * 8 * 9 / 2 = 144.
    # M^2 - M - 144 >= 0 -> M ~ 12.5? 
    # Wait. The differences are $a_j - a_i$.
    # The max difference is $M$.
    # The number of available differences in [1, M] is $M$.
    # We need 144 distinct differences.
    # So M >= 144.
    # My formula above was wrong. The differences are just integers from 1 to M.
    # They don't have to be distinct modulo anything.
    # So we just need M >= total_diffs_needed.
    # M >= 144.
    
    # However, the structure of the ruler imposes more constraints.
    # The counting bound mentioned in the prompt is 112? 
    # "simple counting bound is 112".
    # Let's re-read carefully.
    # "all differences between two entries of the same row are distinct over all rows"
    # This means the set of all differences $D = \bigcup_{r} \{ a_{r,j} - a_{r,i} \mid i < j \}$ has size $n\_rows \times k(k+1)/2$.
    # All elements in $D$ are distinct.
    # The elements are in range $[1, M]$.
    # So $M \ge |D| = 144$.
    # Why is the counting bound 112?
    # Ah, maybe the counting bound refers to something else or I am misinterpreting "differences".
    # If the bound is 112, it implies that not all differences need to be distinct? 
    # "all differences ... are distinct over all rows". This implies global distinctness.
    # If global distinctness is required, M must be at least the number of differences.
    # Unless... "differences" means something else?
    # No, "a_j - a_i".
    # Maybe the "counting bound" 112 is for a different variant?
    # Or maybe my calculation of total diffs is wrong?
    # k=8. Marks: 0, a1, ..., a8.
    # Pairs: (0,a1)...(0,a8) -> 8
    # (a1,a2)...(a1,a8) -> 7
    # ...
    # (a7,a8) -> 1
    # Sum = 8+7+...+1 = 36.
    # 4 rows -> 144 diffs.
    # If they must be distinct, M >= 144.
    # The prompt says "best known set has largest entry 134".
    # This is a contradiction. 134 < 144.
    # Therefore, the differences do NOT have to be globally distinct?
    # Re-read: "all differences between two entries of the same row are distinct over all rows"
    # This phrasing is ambiguous.
    # Interpretation A: For any two pairs (row1, i1, j1) and (row2, i2, j2), if (row1, i1, j1) != (row2, i2, j2), then diff1 != diff2. (Global distinctness).
    # Interpretation B: Within each row, differences are distinct. AND, the set of differences in row 1 is disjoint from set in row 2, etc.
    # Both imply global distinctness.
    
    # Let's check the source: "Difference triangle set".
    # Usually, a difference triangle set (DTS) or perfect difference set requires distinct differences.
    # If M=134 is possible, then the number of distinct differences required must be <= 134.
    # 144 > 134.
    # So either k is not 8? "Choose 4 rows of 8 increasing integers".
    # Does "8 increasing integers" mean 8 integers total (including 0)?
    # "starts at 0 and increases strictly".
    # "list of k + 1 integers".
    # If k=8, list has 9 integers.
    # If the list has 8 integers (k=7), diffs = 7*8/2 = 28. 4*28 = 112.
    # 112 matches the counting bound!
    # So, likely k in the prompt "rows of 8 increasing integers" implies k=7 in the formula `k+1`?
    # No, the interface is `construct(n_rows, k)`.
    # "Each row is a list of k + 1 integers".
    # If the task says "rows of 8 increasing integers", it probably means the list length is 8.
    # So `len(row) == 8`.
    # This implies `k+1 = 8 => k=7`.
    # Let's check the prompt again.
    # "Choose 4 rows of 8 increasing integers starting at 0"
    # "Interface: construct(n_rows, k) ... Each row is a list of k + 1 integers"
    # If the evaluator passes `k` such that `k+1=8`, then `k=7`.
    # If the evaluator passes `k=8`, then `len(row)=9`.
    # The prompt says "This task's n_rows and k".
    # And "Difference triangle set (4,7)".
    # The title says (4,7). This strongly suggests n=4, k=7.
    # So `len(row) = 8`.
    # Total diffs = 4 * (7*8/2) = 4 * 28 = 112.
    # Counting bound M >= 112.
    # Best known M = 134.
    # This makes sense.
    
    # So, assume k=7 is passed, or we should handle general k.
    # The code should use the passed `k`.
    # The prompt says "The evaluator passes this task's n_rows and k".
    # I will assume the evaluator passes k=7 for this specific task instance.
    # But the code should be general.
    
    # Recalculating LB for k=7:
    # total_diffs = n_rows * k * (k+1) / 2
    # M >= total_diffs
    
    # For k=7, n=4: total_diffs = 112.
    # We will search for M starting from 112.
    
    # Algorithm:
    # 1. Bitset for used diffs.
    # 2. Backtracking to fill rows.
    # 3. Symmetry breaking: rows sorted by their last element (or lexicographically).
    
    def solve_for_M(M):
        used = 0
        rows = []
        
        # We fill row by row.
        # To break symmetry, we enforce that the last element of row i <= last element of row i+1.
        # Actually, since we don't know the last element until the end, we can enforce that the tuple of marks is lexicographically <= the next row's tuple.
        # This is hard to check incrementally.
        # Simpler: Just generate rows in a canonical order.
        # Since rows are independent except for `used`, we can just generate them.
        # But to avoid generating permutations of the same set of rows, we can require that the rows are generated in non-decreasing order of their last element.
        
        def backtrack(row_idx, col_idx, current_row, last_mark):
            if row_idx == n_rows:
                return True
            
            if col_idx == 0:
                # Start new row
                # Symmetry break: if row_idx > 0, the last mark of current_row (which will be determined later) 
                # should be >= last mark of previous row?
                # We don't know the last mark yet.
                # Alternative: Sort rows at the end? No, we want to prune.
                # Let's just use a set of "seen" row tuples to avoid duplicates? 
                # With M~134 and 8 marks, the number of possible rows is huge.
                
                # Better symmetry break: 
                # The set of rows is unordered. 
                # We can enforce that the rows are generated in lexicographical order.
                # i.e., row[i] <= row[i+1] lexicographically.
                
                if row_idx > 0:
                    prev_row = rows[-1]
                    # We must ensure current_row (once complete) is >= prev_row lexicographically.
                    # We can check this incrementally.
                    # But we don't have current_row complete.
                    # We can check: if current_row[0..col_idx] is already > prev_row[0..col_idx], it's fine.
                    # If it is <, it's invalid.
                    # If it is equal so far, we must ensure the rest is >=.
                    
                    # Since col_idx is 0, current_row is [0].
                    # prev_row[0] is 0. So equal so far.
                    # We just need to ensure the final row is >= prev_row.
                    
                # Try to build the row
                # We need to choose k marks.
                # We do this recursively for each column.
                return build_row(row_idx, 1, [0], 0)
            
            # This structure is getting complex. Let's flatten.
            pass
        
        # Let's use a different structure.
        # State: (row_idx, col_idx, current_row_marks, used_diffs)
        # current_row_marks is a list.
        
        def dfs(row_idx, col_idx, current_row, used):
            if row_idx == n_rows:
                return True
            
            # If we are starting a new row (col_idx == 0)
            if col_idx == 0:
                # Symmetry check: 
                # If row_idx > 0, we require current_row (eventually) to be lexicographically >= rows[row_idx-1].
                # Since current_row starts with [0], and prev_row starts with [0], 
                # we just need to ensure that if we fail to be greater, we don't go down a path that can't recover.
                
                # We will build the row mark by mark.
                return build_row(row_idx, 1, [0], used)
            
            # This is not right. build_row needs to return the completed row to add to `rows`.
            # Let's make build_row return (success, new_used, completed_row)
            
        def build_row(r_idx, c_idx, row_marks, used):
            # row_marks has length c_idx
            # We need to add k - c_idx + 1 more marks?
            # No, total marks is k+1.
            # If c_idx == k+1, row is complete.
            
            if c_idx == k + 1:
                # Check symmetry with previous row
                if r_idx > 0:
                    prev = rows[r_idx-1]
                    # Check if row_marks >= prev lexicographically
                    if row_marks < prev:
                        return None
                # Add to rows
                rows.append(row_marks)
                # Recurse to next row
                return dfs(r_idx + 1, 0, [], used)
            
            # Try next mark
            # Last mark is row_marks[-1]
            last = row_marks[-1]
            # Next mark x must be > last
            # x <= M
            
            # Pruning: 
            # We have (k+1) - c_idx marks left to place.
            # The smallest possible next marks are last+1, last+2, ...
            # The largest possible mark is M.
            # We need to ensure we can fit the remaining marks.
            # Min sum of next marks?
            
            # Start from last + 1
            # To improve speed, we can iterate x.
            
            # Heuristic: Try x that are "spread out"? 
            # Or just increasing order.
            
            for x in range(last + 1, M + 1):
                # Calculate diffs
                new_diffs = 0
                valid = True
                for a in row_marks:
                    d = x - a
                    if d > M: # Should not happen if x<=M and a>=0
                        valid = False
                        break
                    bit = 1 << d
                    if used & bit:
                        valid = False
                        break
                    new_diffs |= bit
                
                if not valid:
                    continue
                
                # Check if we can complete the row with remaining marks
                # Remaining slots: k + 1 - (c_idx + 1)
                # If remaining slots > 0, we need space.
                # x + (remaining_slots) <= M is a necessary condition?
                # Not strictly, we can jump.
                
                res = build_row(r_idx, c_idx + 1, row_marks + [x], used | new_diffs)
                if res is not None:
                    return res
                
            return None
        
        def dfs(r_idx, c_idx, curr_row, used):
            if r_idx == n_rows:
                return True
            if c_idx == 0:
                return build_row(r_idx, 1, [0], used)
            # This dfs wrapper is redundant if build_row handles the row completion and calls dfs.
            # Let's just call build_row(0, 1, [0], 0)
            pass
        
        # Initialize
        rows.clear()
        # Call build_row for first row
        # But build_row appends to `rows` and calls dfs.
        # We need to handle the recursion properly.
        
        # Let's restructure to a single recursive function.
        
        def search(r_idx, c_idx, curr_row, used):
            if r_idx == n_rows:
                return True
            
            if c_idx == 0:
                # Start new row
                # Symmetry: if r_idx > 0, curr_row (once done) must be >= rows[r_idx-1]
                return search(r_idx, 1, [0], used)
            
            if c_idx == k + 1:
                # Row complete
                # Check symmetry
                if r_idx > 0:
                    if curr_row < rows[r_idx-1]:
                        return False
                rows.append(curr_row)
                return search(r_idx + 1, 0, [], used)
            
            # Add mark at c_idx
            last = curr_row[-1]
            # Optimization: 
            # If we are at the last position (c_idx == k), we just need one value.
            
            for x in range(last + 1, M + 1):
                new_diffs = 0
                ok = True
                for a in curr_row:
                    d = x - a
                    bit = 1 << d
                    if used & bit:
                        ok = False
                        break
                    new_diffs |= bit
                if not ok:
                    continue
                
                # Pruning: Check if remaining marks can fit
                # Remaining marks to place: k + 1 - c_idx
                # We are placing x now.
                # Next min mark is x+1.
                # We need to place (k + 1 - c_idx - 1) more marks.
                # Min value for last mark >= x + (k + 1 - c_idx - 1)
                # This must be <= M
                if x + (k + 1 - c_idx - 1) > M:
                    break # Since x is increasing, further x will also fail
                
                res = search(r_idx, c_idx + 1, curr_row + [x], used | new_diffs)
                if res:
                    return True
            return False
        
        rows.clear()
        return search(0, 0, [], 0)

    # Main loop
    best_rows = None
    # Start from LB
    # If k=7, LB=112.
    # If k=8, LB=144.
    
    # Add a small buffer to LB to skip impossible tight cases quickly?
    # No, we want the min M.
    
    current_M = lb
    while True:
        if time.time() > deadline:
            break
        
        if solve_for_M(current_M):
            best_rows = [list(r) for r in rows]
            break
        
        current_M += 1
        
        # If we go too high, we might time out.
        # But the solution is expected to be found.
        # If we reach M=200 and haven't found, something is wrong or time is up.
        
    if best_rows is None:
        # Fallback to greedy if search fails or times out
        # Re-implement greedy quickly
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
        return greedy_rows
    
    return best_rows

# Test with k=7
# construct(4, 7)

# One issue: `rows` is a list defined in `construct`. 
# `solve_for_M` uses `rows`. 
# `search` uses `rows`.
# This is fine as long as it's local to `construct`.

# Wait, `rows` is cleared in `solve_for_M`.
# `search` appends to `rows`.
# If `search` returns True, `rows` contains the solution.
# If `search` returns False, `rows` might be partially filled or empty.
# We clear it at the start of `solve_for_M`.

# Complexity:
# For M=112, k=7.
# The search space is large.
# Bitset operations are fast.
# Symmetry breaking helps.

# Potential issue: Python recursion depth.
# Depth is max n_rows * (k+1) = 4 * 8 = 32. Safe.

# Let's refine the pruning.
# In `search`, when choosing x:
# We check `if x + (k + 1 - c_idx - 1) > M: break`
# This ensures that even if we pick consecutive integers, we don't exceed M.
# This is a valid necessary condition.

# Another pruning:
# The number of available differences in [1, M] is M.
# We have used `used` differences.
# We need `remaining_diffs` more.
# `remaining_diffs` = total_diffs_needed - (number of diffs in completed rows) - (diffs in current row so far).
# If `M - popcount(used) < remaining_diffs`, then impossible.
# `popcount` for Python int is `bin(used).count('1')` or `used.bit_count()` (Python 3.10+).
# Let's use `used.bit_count()`.

# Update `search`:
# Calculate diffs used so far.
# This might be expensive to calculate every time?
# No, we can pass `used_count` as an argument.

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    import math
    lb = total_diffs_needed
    
    # If lb is very small, we might want to start a bit higher?
    # No, start at lb.
    
    def solve_for_M(M):
        rows = []
        
        def search(r_idx, c_idx, curr_row, used, used_count):
            if r_idx == n_rows:
                return True
            
            if c_idx == 0:
                return search(r_idx, 1, [0], used, used_count)
            
            if c_idx == k + 1:
                if r_idx > 0:
                    if curr_row < rows[r_idx-1]:
                        return False
                rows.append(curr_row)
                return search(r_idx + 1, 0, [], used, used_count)
            
            last = curr_row[-1]
            
            # Pruning: Check if enough diffs are left
            # Diffs used in current row so far:
            # We don't track this explicitly, but we can calculate remaining needed.
            # Total needed: total_diffs_needed
            # Used so far: used_count
            # Diffs in current row so far: 
            # Let's not track it separately, just check if `M - used_count < remaining_total_diffs`
            # remaining_total_diffs = total_diffs_needed - used_count
            # If M - used_count < remaining_total_diffs, return False.
            # This is a global check.
            
            if M - used_count < (total_diffs_needed - used_count):
                return False
            
            for x in range(last + 1, M + 1):
                # Check if we can fit remaining marks
                if x + (k + 1 - c_idx - 1) > M:
                    break
                
                new_diffs = 0
                ok = True
                for a in curr_row:
                    d = x - a
                    bit = 1 << d
                    if used & bit:
                        ok = False
                        break
                    new_diffs |= bit
                if not ok:
                    continue
                
                new_used = used | new_diffs
                new_used_count = used_count + new_diffs.bit_count()
                
                res = search(r_idx, c_idx + 1, curr_row + [x], new_used, new_used_count)
                if res:
                    return True
            return False
        
        return search(0, 0, [], 0, 0), rows

    best_rows = None
    current_M = lb
    
    while True:
        if time.time() > deadline:
            break
            
        found, rows = solve_for_M(current_M)
        if found:
            best_rows = rows
            break
        
        current_M += 1
        
        # If we are getting too close to deadline, break
        if current_M > lb + 100 and time.time() > deadline - 10:
            break

    if best_rows is None:
        # Greedy fallback
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
        return greedy_rows
    
    return best_rows
