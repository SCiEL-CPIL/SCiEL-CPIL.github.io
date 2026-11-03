import time
import sys

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all differences are distinct across all rows, minimizing the max entry.
    """
    # Lower bound on M
    # Total differences needed = n_rows * (k * (k+1) // 2)
    # Max difference is M, so M >= total_diffs
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # We will search for the smallest M
    # Start from the counting bound
    start_m = total_diffs
    
    # Deadline
    deadline = time.time() + 110  # Leave 10s margin for 120s limit
    
    best_solution = None
    best_m = float('inf')
    
    # We'll try M from start_m upwards
    # If we find a solution for M, that's the optimal (since we search in increasing order)
    # But we need to be careful: is it guaranteed that if a solution exists for M, 
    # it exists for M+1? Yes. So the first M we find is the minimum.
    
    # However, the search might fail for M=105 due to search time, but a solution exists.
    # We need a robust search.
    
    # Let's use a backtracking search with symmetry breaking.
    # Rows are indistinguishable. We can enforce row_i <= row_{i+1} lexicographically.
    
    def solve_m(M):
        """
        Attempts to find a valid configuration with max entry M.
        Returns a list of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        
        # Helper to check if a row is valid (distinct diffs internally)
        # This is checked during construction.
        
        def backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_solution, best_m, deadline
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                # Found a solution
                # We need to copy the state
                best_solution = [list(r) for r in rows]
                best_m = M
                return True
            
            # Heuristic: try to fill current_row
            if len(current_row) < k + 1:
                # Need to add a mark
                # Try all possible next marks x
                # x must be > current_row[-1]
                # and x <= M
                last_mark = current_row[-1]
                for x in range(last_mark + 1, M + 1):
                    # Check differences
                    new_diffs = []
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Check if new_diffs have duplicates internally? 
                    # No, because a is distinct in current_row, so x-a are distinct.
                    
                    # Symmetry breaking for the first row? 
                    # Or for subsequent rows:
                    # If row_idx > 0, the new row must be lexicographically >= previous row.
                    # We can check this when the row is complete.
                    
                    # Add to state
                    current_row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                        
                    if backtrack(row_idx, current_row):
                        return True
                        
                    # Backtrack
                    for d in new_diffs:
                        used_diffs.remove(d)
                    current_row.pop()
                    
            else:
                # Row is complete
                # Check symmetry: if row_idx > 0, current_row must be >= rows[-1]
                if row_idx > 0 and current_row < rows[-1]:
                    return False
                
                # Add to rows
                rows.append(list(current_row))
                
                if backtrack(row_idx + 1, [0]):
                    return True
                
                # Backtrack
                rows.pop()
                
            return False
        
        # Start with first row starting at 0
        # To reduce symmetry, we can fix the first row's first non-zero element?
        # Or just let the search handle it.
        # However, for the very first row, we can try to make it "dense" or "sparse"?
        # Let's just start with [0].
        
        return backtrack(0, [0])

    # Search for M
    # We might not find a solution for small M quickly.
    # We'll iterate M.
    # If we run out of time, we return the best found.
    
    # Optimization: 
    # The search space is huge. 
    # For M=105, k=5, n=7.
    # Depth of recursion: 7 rows * 5 marks = 35 levels.
    # Branching factor: up to M.
    # This is too slow for naive backtracking.
    
    # We need a better heuristic or a different approach.
    # Let's try to generate candidate rows first?
    # No, M is small enough (105-120).
    
    # Let's add a time check inside the loop and break if too slow.
    # Also, we can use a randomized restart or different ordering.
    
    # Let's try a simple optimization: 
    # When adding a mark x, the differences x-a must be distinct and unused.
    # We can pre-compute valid extensions? No, state-dependent.
    
    # Let's try to limit the search depth or use a timeout per M.
    # If M=105 takes > 5s, skip to M=106?
    # But we want the best possible.
    
    # Alternative: Use a SAT solver? Not available.
    # Use a heuristic: 
    # Sort candidate marks by "constraint tightness".
    # Marks that create differences close to already used ones?
    
    # Let's stick to the backtracking but add a node limit.
    
    node_limit = 100000
    nodes_visited = 0
    
    def backtrack_limited(row_idx, current_row):
        nonlocal used_diffs, rows, best_solution, best_m, deadline, nodes_visited
        
        if time.time() > deadline:
            return False
        
        nodes_visited += 1
        if nodes_visited > node_limit:
            return False
            
        if row_idx == n_rows:
            best_solution = [list(r) for r in rows]
            best_m = M
            return True
            
        if len(current_row) < k + 1:
            last_mark = current_row[-1]
            # Heuristic: try marks that are "far" from last_mark first? 
            # Or "close"?
            # For Golomb rulers, spreading out is good.
            # Let's try from last_mark+1 to M.
            for x in range(last_mark + 1, M + 1):
                new_diffs = []
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                    
                if backtrack_limited(row_idx, current_row):
                    return True
                    
                for d in new_diffs:
                    used_diffs.remove(d)
                current_row.pop()
        else:
            if row_idx > 0 and current_row < rows[-1]:
                return False
                
            rows.append(list(current_row))
            if backtrack_limited(row_idx + 1, [0]):
                return True
            rows.pop()
            
        return False

    # We will try M from start_m
    # If we find a solution, we return it.
    # If we don't find one for a specific M within the node limit, we try M+1.
    # This is not strictly correct (might miss solution for M=105 and find for 106), 
    # but it's a heuristic.
    
    # To be safer, we can increase node limit for smaller M?
    # Or just run it.
    
    # Let's try M from start_m to start_m + 20
    for M in range(start_m, start_m + 50):
        if time.time() > deadline - 10:
            break
            
        nodes_visited = 0
        used_diffs = set()
        rows = []
        
        # Reset best_solution for this M attempt? 
        # No, we want the global best.
        # If we find a solution for M, it's better than any previous M (since we go up).
        # So if we find one, we can return immediately?
        # Yes, because we search M in increasing order.
        
        found = backtrack_limited(0, [0])
        
        if found:
            return best_solution
            
    # If no solution found with the limited search, 
    # fallback to a greedy construction to ensure we return *something* valid.
    # The seed program is a good fallback.
    
    # Let's implement the seed program's logic as a fallback.
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    return rows

# EVOLVE-BLOCK-END
