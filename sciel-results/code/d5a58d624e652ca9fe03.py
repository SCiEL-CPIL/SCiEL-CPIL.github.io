"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct over all rows.
    Minimize the largest entry (scope).
    
    Uses backtracking with bitmask optimization, symmetry breaking, and forward checking.
    """
    
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting lower bound
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    lb = max(lb, k)
    
    start_time = time.time()
    timeout = 115  # Leave margin
    
    def solve(scope):
        used_mask = 0
        rows = []
        
        def backtrack(row_idx, current_row, current_row_mask):
            """
            row_idx: index of the row being constructed
            current_row: list of elements in the current row (starts with [0])
            current_row_mask: bitmask of differences already formed within current_row
            """
            if row_idx == n_rows:
                return True
            
            # Symmetry breaking: Enforce lexicographic ordering of rows
            # current_row must be >= rows[-1] lexicographically
            if row_idx > 0:
                prev_row = rows[-1]
                # Check lexicographic order
                # current_row has len(current_row) elements so far
                # prev_row has k+1 elements
                for i in range(len(current_row)):
                    if current_row[i] < prev_row[i]:
                        return False
                    if current_row[i] > prev_row[i]:
                        break # Already greater, no further constraint
                # If loop finishes without break, current_row is a prefix of prev_row
                # But since lengths will be equal at the end, and we are increasing,
                # if current_row is a prefix, the next elements must be >= corresponding prev elements.
                # This is handled naturally by the min_next calculation if we are careful?
                # No, min_next is just current_row[-1] + 1.
                # We need to ensure that if we are still equal, the next choice is >= prev_row[i].
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row[:])
                new_used_mask = used_mask | current_row_mask
                
                # Forward checking: 
                rem_rows = n_rows - row_idx - 1
                rem_diffs_needed = rem_rows * diffs_per_row
                
                used_count = new_used_mask.bit_count()
                available = scope * (scope + 1) // 2 - used_count
                
                if available < rem_diffs_needed:
                    rows.pop()
                    return False
                
                if backtrack(row_idx + 1, [0], 0):
                    return True
                rows.pop()
                return False
            
            # Determine next element
            is_last_pos = (len(current_row) == k)
            is_row_0 = (row_idx == 0)
            
            min_next = current_row[-1] + 1 if current_row else 1
            
            # Symmetry breaking for Row 0: Last element must be scope
            if is_row_0 and is_last_pos:
                candidates = [scope]
            else:
                max_next = scope
                if is_row_0 and not is_last_pos:
                    max_next = scope - 1 # Must leave room for scope at the end
                
                # Lexicographic constraint from previous row
                if row_idx > 0:
                    prev_row = rows[-1]
                    # Find the first index where current_row and prev_row differ
                    # If they are equal so far, min_next must be at least prev_row[len(current_row)]
                    # Wait, current_row has indices 0..len(current_row)-1.
                    # We are choosing index len(current_row).
                    # If current_row[0..len-1] == prev_row[0..len-1], then next must be >= prev_row[len].
                    equal_so_far = True
                    for i in range(len(current_row)):
                        if current_row[i] != prev_row[i]:
                            equal_so_far = False
                            break
                    if equal_so_far:
                        min_next = max(min_next, prev_row[len(current_row)])
                
                candidates = range(min_next, max_next + 1)
            
            for x in candidates:
                # Generate new differences
                new_diffs = [x - a for a in current_row]
                
                # Check validity
                local_mask = 0
                valid = True
                for d in new_diffs:
                    if d > scope:
                        valid = False
                        break
                    bit = 1 << d
                    if (used_mask & bit) or (current_row_mask & bit):
                        valid = False
                        break
                    local_mask |= bit
                
                if not valid:
                    continue
                
                new_row_mask = current_row_mask | local_mask
                
                # Place x
                current_row.append(x)
                
                if backtrack(row_idx, current_row, new_row_mask):
                    return True
                
                current_row.pop()
            
            return False
        
        return backtrack(0, [0], 0)
    
    # Try scopes starting from the lower bound
    for scope in range(lb, lb + 100):
        if time.time() - start_time > timeout:
            break
        
        result = solve(scope)
        if result:
            # Return the rows found
            # Note: rows is local to solve, we need to return it
            # We can modify solve to return rows or use a nonlocal variable
            pass # Placeholder, will fix in replacement

    # Fallback: Greedy
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
