import sys
from functools import lru_cache

def construct(n_rows, k):
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound: we need total_diffs distinct positive differences, each <= M
    # So M >= total_diffs
    lower_bound = total_diffs
    
    # Upper bound: start from lower_bound and increase
    # For (7,5): total_diffs = 7*15 = 105, optimal is 112
    for M in range(lower_bound, lower_bound + 30):
        result = solve_mrv(n_rows, k, M)
        if result is not None:
            return result
    
    # Fallback to greedy if backtracking fails
    return greedy_fallback(n_rows, k)

def solve_mrv(n_rows, k, M):
    """
    Solve using MRV heuristic: pick the next row that has the fewest valid extensions.
    Use bitset for used differences.
    """
    used = 0  # bitset of used differences
    rows = []
    
    # We'll use a recursive approach with MRV
    # State: which rows are assigned, used differences bitset
    # For efficiency, we try all rows and pick the one with fewest valid completions
    
    def count_valid_rows(used_bits, M, k):
        """Count how many rows can still be formed given used_bits."""
        count = 0
        for start in range(1, M + 1):
            # Try to form a row starting with 0, then next element >= start
            # Actually, let's just check if we can greedily form at least one row
            # This is expensive, so we'll use a simpler heuristic
            pass
        return count
    
    # Simpler approach: try rows in order but with better pruning
    # Key insight: for a given M, the number of available differences is M
    # We need total_diffs distinct differences, each <= M
    # So we need M >= total_diffs
    
    if M < total_diffs:
        return None
    
    # Use backtracking with forward checking
    def backtrack(row_idx, used_bits, rows):
        if row_idx == n_rows:
            return True
        
        # Try to find a valid row
        row = [0]
        if find_row(0, row, used_bits, k, M):
            rows.append(row)
            
            # Update used bits
            new_used = used_bits
            for i in range(len(row)):
                for j in range(i):
                    diff = row[i] - row[j]
                    new_used |= (1 << diff)
            
            if backtrack(row_idx + 1, new_used, rows):
                return True
            
            rows.pop()
        
        return False
    
    def find_row(pos, row, used_bits, k, M):
        """Find the next element for the row at position pos."""
        if pos == k + 1:
            return True
        
        last = row[-1]
        start = last + 1
        
        # Pruning: check if enough differences remain
        # Remaining positions: k + 1 - pos
        # Each new element x creates (pos) new differences
        # Total remaining differences needed: sum of pos for pos in range(len(row), k+1)
        remaining_diffs = 0
        for p in range(len(row), k + 1):
            remaining_diffs += p
        
        # Available differences: M - used_count
        used_count = bin(used_bits).count('1')
        if M - used_count < remaining_diffs:
            return False
        
        for x in range(start, M + 1):
            # Check if all differences x - a for a in row are available
            ok = True
            for a in row:
                d = x - a
                if d > M or (used_bits & (1 << d)):
                    ok = False
                    break
            
            if ok:
                row.append(x)
                new_used = used_bits
                for a in row[:-1]:
                    d = x - a
                    new_used |= (1 << d)
                
                if find_row(pos + 1, row, new_used, k, M):
                    return True
                
                row.pop()
        
        return False
    
    if backtrack(0, 0, rows):
        return rows
    return None

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
