# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    n_elem = k + 1
    diffs_per_row = n_elem * (n_elem - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: We need total_diffs distinct differences in [1, M].
    # So M >= total_diffs.
    # However, differences are constrained by the row structure.
    # A tighter lower bound for DTS(n, k) is often close to total_diffs.
    # For n=7, k=5, total_diffs = 105.
    # Let's start from total_diffs.
    
    start_M = total_diffs
    
    def can_construct(M):
        used_diffs = set()
        rows = []
        
        def find_row(pos, current_row):
            """
            Recursively build a row.
            Returns a valid row (list) if found, else None.
            """
            if pos == n_elem:
                return current_row[:]
            
            start_val = current_row[-1] + 1 if current_row else 1
            
            # Optimization: Prune if not enough values left to fill the row
            # We need n_elem - pos more values, all > current_row[-1] and <= M
            if M - start_val < (n_elem - pos - 1):
                return None

            # Iterate possible values
            # Heuristic: Try values that create "large" differences first? 
            # Or small? 
            # Let's try increasing order.
            for val in range(start_val, M + 1):
                # Check differences with all previous elements
                valid = True
                new_diffs = set()
                for prev in current_row:
                    d = val - prev
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Prune: Check if we have enough remaining differences available
                # Total diffs needed for this row: diffs_per_row
                # Diffs used so far in this row: len(new_diffs) + (pos*(pos-1)//2) ? No, new_diffs are just for val.
                # Total diffs in current_row + val will be len(new_diffs) + (pos*(pos-1)//2) is not right.
                # Let's just check global capacity.
                # Used so far: len(used_diffs) + len(new_diffs)
                # Remaining rows: n_rows - len(rows) - 1 (if this succeeds)
                # This prune is hard to do precisely without knowing future rows.
                
                # Tentatively add new_diffs to used
                # To avoid copying sets, we can add, recurse, remove.
                for d in new_diffs:
                    used_diffs.add(d)
                
                current_row.append(val)
                res = find_row(pos + 1, current_row)
                if res is not None:
                    return res
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.remove(d)
            
            return None

        def backtrack_rows(row_idx):
            if row_idx == n_rows:
                return True
            
            # Heuristic: Try to find a row.
            # We can try to find a row that uses "rare" diffs?
            # For now, just find any valid row.
            row = find_row(0, [0])
            if row is None:
                return False
            
            # Collect diffs of this row
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            # Add to used
            for d in row_diffs:
                used_diffs.add(d)
            
            rows.append(row)
            
            if backtrack_rows(row_idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            for d in row_diffs:
                used_diffs.remove(d)
            
            return False

        return backtrack_rows(0), rows

    # Search for minimum M
    # Upper bound heuristic: total_diffs * 1.2 or something?
    # Best known for 7,5 is 112. Total diffs 105.
    # Let's try up to 150 to be safe, but break early.
    for M in range(start_M, start_M + 50):
        success, rows = can_construct(M)
        if success:
            return rows
    
    # Fallback: Return a valid configuration with a larger M
    # Greedy approach
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
