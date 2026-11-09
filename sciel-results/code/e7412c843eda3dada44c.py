"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n_rows, k):
    # k+1 elements per row
    # n_rows rows
    # Minimize max element
    
    # Total differences per row: C(k+1, 2)
    N_diffs_per_row = (k + 1) * k // 2
    N_diffs_total = n_rows * N_diffs_per_row
    
    # Lower bound for M
    M_lower = N_diffs_total
    
    start_time = time.time()
    time_limit = 110.0
    
    # We will iterate M from M_lower upwards
    # We assume a solution exists within M_lower + 20
    
    for M in range(M_lower, M_lower + 20):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution with max value <= M
        # We use a DFS with a node limit to avoid hanging on hard instances.
        
        used_diffs = set()
        rows = []
        current_row = [0]
        
        node_count = 0
        max_nodes = 2000000 # Limit per M to ensure progress
        
        def dfs(row_idx, col_idx):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return False
            
            if row_idx == n_rows:
                return True
                
            if col_idx == k + 1:
                # Current row complete
                rows.append(current_row[:])
                return dfs(row_idx + 1, 0)
            
            last_val = current_row[-1]
            
            # Try next values in increasing order
            for x in range(last_val + 1, M + 1):
                # Check validity
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    current_row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if dfs(row_idx, col_idx + 1):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
            
            return False
        
        # Start search
        if dfs(0, 1):
            return rows
            
    # Fallback: Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
