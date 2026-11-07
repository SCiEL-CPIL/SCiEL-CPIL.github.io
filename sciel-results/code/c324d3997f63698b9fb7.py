"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _verify_squares(squares):
    """
    Verify that the given list of squares are mutually orthogonal Latin squares.
    Returns True if valid, False otherwise.
    """
    if not squares:
        return False
    
    n = len(squares[0])
    if n != 10:
        return False
        
    # Check if each square is a Latin square
    for sq in squares:
        if not isinstance(sq, (list, tuple)):
            return False
        for i in range(n):
            if not isinstance(sq[i], (list, tuple)):
                return False
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
                
    # Check orthogonality for all pairs
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pair = (squares[i][r][c], squares[j][r][c])
                    if pair in pairs:
                        return False
                    pairs.add(pair)
            if len(pairs) != n * n:
                return False
    return True

def _find_orthogonal_mate(L1, n=10):
    """
    Find a Latin square L2 orthogonal to L1 using backtracking with MRV heuristic.
    Returns None if not found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Initialize L2 with None
    L2 = [[None for _ in range(n)] for _ in range(n)]
    
    # Track constraints
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    used_pairs = set()
    
    # List of all cells (r, c)
    cells = [(r, c) for r in range(n) for c in range(n)]
    
    # We will use a recursive backtracking function
    # To avoid recursion depth issues and allow time checking, we might use iterative or just careful recursion.
    # n=10 is small, recursion depth 100 is fine.
    
    def backtrack(num_placed):
        if num_placed == n * n:
            return True
            
        # Time check
        if time.time() > deadline:
            return False
        
        # MRV Heuristic: Find the unassigned cell with the fewest valid candidates
        min_candidates = None
        min_count = float('inf')
        target_cell = None
        
        for r in range(n):
            for c in range(n):
                if L2[r][c] is None:
                    # Calculate valid symbols
                    valid_syms = []
                    for s in range(n):
                        if s not in row_used[r] and s not in col_used[c]:
                            # Check orthogonality constraint
                            pair = (L1[r][c], s)
                            if pair not in used_pairs:
                                valid_syms.append(s)
                    
                    if len(valid_syms) < min_count:
                        min_count = len(valid_syms)
                        min_candidates = valid_syms
                        target_cell = (r, c)
                        
                        if min_count == 0:
                            return False # Dead end
                        if min_count == 1:
                            break # Can't get better than 1
                    
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        # Break out of inner loop, but need to break outer too or check flag
                        pass
                        
        if target_cell is None:
            return True # Should have returned earlier if num_placed == n*n
            
        r, c = target_cell
        
        # Try each candidate symbol
        # Shuffle candidates to avoid bias? Or just try in order.
        # For MOLS, order might matter for speed, but any solution works.
        for s in min_candidates:
            # Place symbol
            L2[r][c] = s
            row_used[r].add(s)
            col_used[c].add(s)
            used_pairs.add((L1[r][c], s))
            
            if backtrack(num_placed + 1):
                return True
            
            # Unplace symbol
            L2[r][c] = None
            row_used[r].remove(s)
            col_used[c].remove(s)
            used_pairs.remove((L1[r][c], s))
            
        return False

    # To help the search, we can fix the first row to be [0, 1, ..., 9].
    # This is valid because if L2 is a solution, permuting its symbols gives another solution.
    # By fixing the first row, we reduce the search space significantly.
    # However, we must ensure that the fixed row is consistent with L1.
    # L1[0][c] = c. So pair is (c, s).
    # If we fix L2[0][c] = c, then pairs are (c, c). These are all distinct.
    # So this is a valid constraint.
    
    for c in range(n):
        L2[0][c] = c
        row_used[0].add(c)
        col_used[c].add(c)
        used_pairs.add((L1[0][c], c))
        
    if not backtrack(10): # 10 cells placed in first row
        return None
        
    return L2

def construct():
    start_time = time.time()
    
    # Square 1: Cyclic Latin Square L1[i][j] = (i+j) % 10
    l1 = []
    for i in range(10):
        row = []
        for j in range(10):
            row.append((i + j) % 10)
        l1.append(row)
        
    # Ensure types are Python int
    l1 = [[int(v) for v in row] for row in l1]
    
    # Try to find an orthogonal mate
    l2 = _find_orthogonal_mate(l1)
    
    if l2 is not None:
        # Ensure types are Python int
        l2 = [[int(v) for v in row] for row in l2]
        
        # Verify before returning to be absolutely sure
        if _verify_squares([l1, l2]):
            return [l1, l2]
            
    # Fallback: Return just l1
    return [l1]

# EVOLVE-BLOCK-END
