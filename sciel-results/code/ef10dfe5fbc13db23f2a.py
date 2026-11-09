"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    n_elem = k + 1
    diffs_per_row = n_elem * (n_elem - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: M must be at least total_diffs
    min_M = total_diffs
    
    # We will search for M starting from min_M
    # Using randomized backtracking with restarts
    
    def solve_for_M(M):
        """
        Attempt to find a valid configuration with max element <= M.
        Uses randomized backtracking with restarts.
        """
        max_restarts = 50
        max_nodes_per_restart = 20000
        
        for restart in range(max_restarts):
            node_count = [0]
            
            def _backtrack(row_idx, elem_idx, current_row, used_diffs, completed_rows):
                if node_count[0] > max_nodes_per_restart:
                    return None
                node_count[0] += 1
                
                if row_idx == n_rows:
                    return completed_rows
                
                if elem_idx == n_elem:
                    # Row is complete
                    return _backtrack(row_idx + 1, 0, [], used_diffs, completed_rows + [current_row])
                
                if elem_idx == 0:
                    # Value must be 0
                    return _backtrack(row_idx, 1, [0], used_diffs, completed_rows)
                
                start_val = current_row[-1] + 1
                end_val = M
                
                # Generate candidates
                candidates = list(range(start_val, end_val + 1))
                random.shuffle(candidates)
                
                for val in candidates:
                    # Check differences
                    valid = True
                    new_diffs = []
                    for prev_val in current_row:
                        d = val - prev_val
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # If valid, recurse
                    # Optimization: if len(new_diffs) is large, union is costly.
                    # But n_elem is small.
                    new_used = used_diffs.union(new_diffs)
                    result = _backtrack(row_idx, elem_idx + 1, current_row + [val], new_used, completed_rows)
                    if result is not None:
                        return result
                        
                return None
            
            res = _backtrack(0, 0, [], set(), [])
            if res is not None:
                return res
                
        return None

    # Search for M
    # Limit the search range to avoid infinite loops if no solution found in range
    # But the problem guarantees a solution exists (best known 112).
    for M in range(min_M, min_M + 20):
        result = solve_for_M(M)
        if result is not None:
            return result
            
    # Fallback: Greedy (may produce larger M)
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
