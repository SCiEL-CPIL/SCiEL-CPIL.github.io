import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 10
    start_time = time.time()
    deadline = 75.0  # Safety margin before 90s kill
    
    # Construct two orthogonal Latin squares of order 10
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 2*j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1, L2]
    
    # Verification functions
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True
        
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n
    
    # Check validity of initial squares
    if not is_latin(L1) or not is_latin(L2) or not are_orthogonal(L1, L2):
        return [L1]
    
    # Try to find a third square orthogonal to both L1 and L2
    # Using backtracking with constraint propagation
    # For each cell (i,j), the value in L3 must satisfy:
    # 1. Each row of L3 is a permutation of 0..9
    # 2. Each column of L3 is a permutation of 0..9
    # 3. For each (i,j), (L1[i][j], L3[i][j]) must be unique across all cells
    # 4. For each (i,j), (L2[i][j], L3[i][j]) must be unique across all cells
    
    # Precompute constraints
    # For each cell (i,j), what values are forbidden?
    # We'll use backtracking with forward checking
    
    # State: L3 as a 10x10 array, initially all -1
    L3 = [[-1] * n for _ in range(n)]
    
    # Track what values are used in each row and column
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    # Track which (L1_val, L3_val) pairs are used
    l1_l3_pairs = set()
    # Track which (L2_val, L3_val) pairs are used
    l2_l3_pairs = set()
    
    # Order cells by most constrained first (MRV heuristic)
    # Initially all cells have 10 possible values, but we can order by row then col
    # Better: process in an order that allows early pruning
    
    # Let's try a simple row-by-row, col-by-col approach with backtracking
    # and forward checking
    
    def get_available_values(i, j):
        """Get available values for cell (i,j) in L3"""
        available = set(range(n))
        available -= row_used[i]
        available -= col_used[j]
        
        # Check L1-L3 pair constraint
        l1_val = L1[i][j]
        for v in list(available):
            if (l1_val, v) in l1_l3_pairs:
                available.discard(v)
        
        # Check L2-L3 pair constraint
        l2_val = L2[i][j]
        for v in list(available):
            if (l2_val, v) in l2_l3_pairs:
                available.discard(v)
        
        return available
    
    def backtrack(i, j):
        """Try to fill L3 starting from cell (i,j)"""
        # Check time
        if time.time() > deadline:
            return None
        
        # If we've filled all cells, we're done
        if i == n - 1 and j == n - 1:
            # Verify the complete square
            if is_latin(L3) and are_orthogonal(L1, L3) and are_orthogonal(L2, L3):
                return L3[:]
            return None
        
        # Get next cell
        next_i, next_j = i, j + 1
        if next_j >= n:
            next_i, next_j = i + 1, 0
        
        # If current cell is already filled, move to next
        if L3[i][j] != -1:
            return backtrack(next_i, next_j)
        
        # Get available values
        available = get_available_values(i, j)
        
        # Try each available value
        for v in sorted(available):
            # Place value
            L3[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            l1_l3_pairs.add((L1[i][j], v))
            l2_l3_pairs.add((L2[i][j], v))
            
            # Recurse
            result = backtrack(next_i, next_j)
            if result is not None:
                return result
            
            # Undo
            L3[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            l1_l3_pairs.discard((L1[i][j], v))
            l2_l3_pairs.discard((L2[i][j], v))
        
        return None
    
    # Try to find a third square
    result = backtrack(0, 0)
    if result is not None:
        squares.append(result)
    
    return squares
# EVOLVE-BLOCK-END
