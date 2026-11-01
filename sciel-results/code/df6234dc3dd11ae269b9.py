import time
import sys
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    num_marks = k + 1
    diffs_per_row = num_marks * (num_marks - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # 1. Greedy Construction for Initial Upper Bound
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_marks:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_solution = greedy_construct()
    if not best_solution:
        return []
    
    best_M = max(max(r) for r in best_solution)
    
    # 2. Precompute Valid Rows for a given M
    # A row is valid if all its internal differences are distinct.
    # We precompute all such rows for M up to best_M (or a bit higher if needed, but best_M is an upper bound).
    # Actually, we only need rows that fit in [0, M].
    
    def get_valid_rows(M):
        valid_rows = []
        # Generate all subsets of {1, ..., M} of size num_marks-1 (since 0 is fixed)
        # Then check if differences are distinct.
        # This is C(M, k) which for M=112, k=5 is ~ 112C5 = 112*111*110*109*108 / 120 ~ 11 million.
        # This might be too slow to precompute ALL valid rows for every M in a loop.
        # However, we can generate them on the fly or precompute once for the max M we expect to search.
        # Since we search M from LB to UB, and LB~105, UB~112, the range is small.
        # But C(112, 5) is large.
        
        # Alternative: Generate valid rows recursively.
        # A row is a list of length num_marks starting with 0.
        # Let's just generate all valid rows for the specific M we are testing in the solver,
        # or better, precompute a global list of ALL possible valid rows (with max element up to some limit)
        # and then filter by max element <= M.
        
        # Let's precompute all valid rows with max element <= best_M.
        # We can do this once.
        pass

    # Better approach: Precompute ALL valid rows (differences distinct) with max element <= best_M.
    # Then for a specific M, we just filter those with max element <= M.
    
    all_valid_rows = []
    
    def generate_rows(current, max_val, limit):
        if time.time() > deadline:
            return
        if len(current) == num_marks:
            # Check distinct differences
            diffs = set()
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    d = current[j] - current[i]
                    if d in diffs:
                        return
                    diffs.add(d)
            all_valid_rows.append((current, diffs, max_val))
            return
        
        # Next element must be > current[-1] and <= limit
        start = current[-1] + 1
        # Optimization: The last element must be <= limit.
        # Also, we need enough room for the remaining elements.
        # If we are at position p (0-indexed), we need num_marks - 1 - p more elements.
        # So the current max_val (which is current[-1]) must be such that we can reach <= limit.
        
        # Simple bound: current[-1] + (num_marks - len(current)) <= limit
        if current[-1] + (num_marks - len(current)) > limit:
            return

        for x in range(start, limit + 1):
            current.append(x)
            generate_rows(current, x, limit)
            current.pop()

    # Precompute valid rows up to best_M
    # This might be slow if best_M is large, but for 112 it's manageable?
    # 112C5 is ~ 11 million. Recursion overhead might be high.
    # Let's try to optimize or use a different generation strategy.
    # Actually, we don't need ALL valid rows. We just need to find a subset of n_rows rows
    # such that their difference sets are disjoint.
    
    # The "Inspiration" suggested ILP/Branch and Bound.
    # Let's implement a Backtracking Search over ROWS.
    # State:
    # - current_row_idx: which row we are building
    # - used_diffs: set of differences used so far
    # - M: current limit
    
    # We can iterate over M from LB to best_M.
    # For each M, we try to find n_rows rows from the set of valid rows with max <= M
    # such that their diff sets are disjoint.
    
    # To speed this up, we can precompute the list of valid rows for the MAX M we will ever test (which is best_M).
    # Then for each M, we filter this list.
    
    # Let's precompute valid rows up to best_M.
    # To avoid timeout in precomputation, we might limit the search or use a smarter generator.
    # Given the small gap between LB and UB, maybe we don't need to precompute everything.
    
    # Alternative Strategy:
    # Instead of precomputing all valid rows, we generate them on the fly during the search for a specific M.
    # But that means regenerating for each M.
    
    # Let's try precomputing once up to best_M.
    # If it takes too long, we fall back to the greedy solution.
    
    precomp_start = time.time()
    generate_rows([0], 0, best_M)
    precomp_end = time.time()
    
    # If precomputation took too long, just return greedy
    if precomp_end - precomp_start > 10:
        return best_solution
        
    # Sort valid rows by their max element to facilitate filtering?
    # Or just keep them in a list.
    
    # Now search for M from LB to best_M
    low = total_diffs
    high = best_M
    
    # Optimization: Sort valid rows by "density" or something?
    # For the row selection, we want rows that have "rare" differences?
    # Or just pick any valid set.
    
    def solve_for_M(M):
        # Filter valid rows with max element <= M
        candidate_rows = [r for r in all_valid_rows if r[2] <= M]
        
        # If not enough candidates, return None
        if len(candidate_rows) < n_rows:
            return None
        
        # Shuffle candidates to break symmetry? Or sort by max element?
        # Sorting by max element might help find solutions with smaller elements earlier?
        # Actually, we just need ANY solution.
        # Let's sort by max element ascending to prefer smaller rows?
        candidate_rows.sort(key=lambda x: x[2])
        
        used_diffs = set()
        chosen_rows = []
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n_rows:
                return True
            
            # Try each candidate row
            for i in range(len(candidate_rows)):
                row_data, diffs, max_val = candidate_rows[i]
                
                # Check if diffs are disjoint with used_diffs
                if diffs & used_diffs:
                    continue
                
                # Symmetry breaking:
                # If idx > 0, we can enforce that the current row is "greater" than the previous one in some sense.
                # Since rows are indistinguishable, we can enforce that the rows are chosen in increasing order of index in candidate_rows?
                # No, that restricts the set.
                # Better: Enforce that the lexicographic order of the chosen rows is increasing.
                # i.e., chosen_rows[idx-1] <= row_data lexicographically.
                if idx > 0 and row_data < chosen_rows[-1]:
                    continue
                
                # Choose this row
                chosen_rows.append(row_data)
                used_diffs.update(diffs)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                chosen_rows.pop()
                used_diffs.difference_update(diffs)
            
            return False
        
        if backtrack(0):
            return chosen_rows
        return None

    # Search M from low to high
    # We can jump if we want, but linear is safe.
    # To speed up, we can start from low and go up.
    
    found_solution = None
    for M in range(low, high + 1):
        if time.time() > deadline:
            break
            
        # Optimization: If M is small, the number of candidate rows is small, so backtrack is fast.
        # If M is large, it's slower.
        
        # We can also use a heuristic: try M values that are likely to have solutions.
        # But linear scan is fine for small range.
        
        solution = solve_for_M(M)
        if solution:
            found_solution = solution
            best_M = M
            break
    
    if found_solution:
        return found_solution
    else:
        return best_solution

# Test locally
if __name__ == "__main__":
    # n_rows=7, k=5
    res = construct(7, 5)
    if res:
        max_val = max(max(r) for r in res)
        print(f"Found solution with max value: {max_val}")
        # Validate
        used = set()
        valid = True
        for r in res:
            if r[0] != 0:
                valid = False
                break
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        valid = False
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No solution found")
