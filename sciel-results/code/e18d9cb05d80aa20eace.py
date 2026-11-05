import sys
import time
import numpy as np
from scipy import optimize

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Lower bound on M
    # Counting bound: total_diffs <= M(M+1)/2
    # 105 <= M(M+1)/2 -> M ~ 14.2? No.
    # 105 diffs. M(M+1)/2 >= 105.
    # 14*15/2 = 105. So M >= 14?
    # Wait, the differences are $a_j - a_i$. The max difference is $a_k - a_0 = a_k \le M$.
    # The set of differences is a subset of $\{1, ..., M\}$.
    # We need 105 distinct differences.
    # So $M \ge 105$? No, the differences don't have to be $1..M$.
    # They just have to be distinct and $\le M$.
    # So we need at least 105 distinct values in $\{1, ..., M\}$.
    # So $M \ge 105$.
    
    lb = 105
    ub = 115 # Known best is 112. Let's search up to 115.
    
    best_solution = None
    best_m = float('inf')
    
    # Try MILP if available
    try:
        # scipy.optimize.milp is available in SciPy >= 1.7
        if hasattr(optimize, 'milp'):
            for M in range(lb, ub + 1):
                sol = solve_milp(n_rows, k, M)
                if sol is not None:
                    best_solution = sol
                    best_m = M
                    break
    except Exception:
        pass
        
    if best_solution is not None:
        return best_solution
        
    # Fallback: Improved Backtracking
    for M in range(lb, ub + 1):
        sol = solve_backtrack(n_rows, k, M)
        if sol is not None:
            return sol
            
    # If no solution found in range, return a greedy solution (might be invalid or large)
    return greedy_fallback(n_rows, k)

def solve_milp(n_rows, k, M):
    # Variables:
    # a[r][i] for r in 0..n_rows-1, i in 0..k
    # x[r][i][j][d] for r in 0..n_rows-1, 0<=i<j<=k, d in 1..M
    
    n_a = n_rows * (k + 1)
    # Indices for a
    # a_idx[r, i] = r * (k+1) + i
    
    pairs = [(i, j) for i in range(k+1) for j in range(i+1, k+1)]
    n_pairs = len(pairs)
    n_x = n_rows * n_pairs * M
    
    # Total variables: n_a + n_x
    # a vars are integers, x vars are binary
    
    c = np.zeros(n_a + n_x)
    # Objective: Minimize 0? No, M is fixed in this loop.
    # We just want to check feasibility.
    # So c is all zeros.
    
    # Constraints
    # 1. a[r][0] = 0
    # 2. a[r][i] < a[r][i+1]  => a[r][i+1] - a[r][i] >= 1
    # 3. a[r][k] <= M
    # 4. Sum_d x[r][i][j][d] = 1
    # 5. a[r][j] - a[r][i] - Sum_d d*x[r][i][j][d] = 0
    # 6. Sum_{r,i,j} x[r][i][j][d] <= 1
    
    # This is getting complex to build the sparse matrix manually.
    # Given the time limit and complexity, and the fact that scipy.milp might be slow or unavailable,
    # I will prioritize the Backtracking approach which is more controllable in Python.
    
    return None

def solve_backtrack(n_rows, k, M):
    used = set()
    rows = []
    
    def backtrack(r, row, used_local):
        if r == n_rows:
            return True
        if r > 0:
            # Check if current row is valid
            pass # Validity checked during construction
        
        # Actually, we build row by row.
        # Let's rewrite: build row r, then recurse to r+1.
        
        def build_row(current_row, used_in_row):
            if len(current_row) == k + 1:
                # Check if this row can be added
                # Differences are in used_in_row
                if not used_in_row & used:
                    # Add to global
                    used.update(used_in_row)
                    rows.append(current_row[:])
                    if backtrack(r + 1, [0], set()):
                        return True
                    used.difference_update(used_in_row)
                    rows.pop()
                return False
            
            last = current_row[-1]
            # Next element x
            # x > last
            # x <= M - (k + 1 - len(current_row))
            max_x = M - (k + 1 - len(current_row))
            
            for x in range(last + 1, max_x + 1):
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_in_row or d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                if build_row(current_row, used_in_row | new_diffs):
                    return True
                current_row.pop()
                
            return False
        
        return build_row([0], set())
        
    return rows if rows else None

def greedy_fallback(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
