import time
import random

def construct():
    n = 10
    # Start time to guard against infinite loops or slow paths
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    # Square 1: Standard cyclic Latin square
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We need to find S2 such that it is a Latin square and orthogonal to S1.
    # Constraints for S2:
    # 1. Each row is a permutation of 0..9
    # 2. Each col is a permutation of 0..9
    # 3. For each (i,j), the pair (S1[i][j], S2[i][j]) is unique.
    
    # Data structures for backtracking
    # row_masks[i]: bitmask of used symbols in row i
    # col_masks[j]: bitmask of used symbols in col j
    # pair_masks[a]: bitmask of used symbols b such that pair (a,b) has been seen.
    # Since S1[i][j] = a, we need to track which b's are used for each a.
    # There are 10 possible 'a's.
    
    row_masks = [0] * n
    col_masks = [0] * n
    pair_masks = [0] * n  # pair_masks[a] tracks which b's are paired with a
    
    S2 = [[-1] * n for _ in range(n)]
    
    def is_valid(i, j, b):
        # Check row
        if row_masks[i] & (1 << b):
            return False
        # Check col
        if col_masks[j] & (1 << b):
            return False
        # Check orthogonality
        a = S1[i][j]
        if pair_masks[a] & (1 << b):
            return False
        return True

    def backtrack(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        
        ni = i
        nj = j + 1
        if nj == n:
            ni = i + 1
            nj = 0
            
        if ni == n:
            return True

        # Try all symbols b for position (ni, nj)
        # To improve performance, we can shuffle the symbols
        symbols = list(range(n))
        random.shuffle(symbols)
        
        for b in symbols:
            if is_valid(ni, nj, b):
                # Place b
                S2[ni][nj] = b
                row_masks[ni] |= (1 << b)
                col_masks[nj] |= (1 << b)
                a = S1[ni][nj]
                pair_masks[a] |= (1 << b)
                
                if backtrack(ni, nj):
                    return True
                
                # Undo
                S2[ni][nj] = -1
                row_masks[ni] ^= (1 << b)
                col_masks[nj] ^= (1 << b)
                pair_masks[a] ^= (1 << b)
                
        return False

    # Run backtracking
    # We might need multiple restarts if the first random path fails, 
    # but for n=10, a single run with randomization usually succeeds immediately.
    # However, to be safe, we loop a few times if it fails quickly.
    found = False
    for _ in range(100):
        if time.time() > deadline:
            break
        # Reset state
        row_masks = [0] * n
        col_masks = [0] * n
        pair_masks = [0] * n
        S2 = [[-1] * n for _ in range(n)]
        
        if backtrack(0, 0):
            found = True
            break
    
    if not found:
        # Fallback: Return just S1 to be safe, though we expect to find 2.
        # Actually, if we can't find it, returning 1 is better than invalid.
        return [S1]
        
    # Verify S2 is valid (sanity check)
    # Check Latin
    for i in range(n):
        if sorted(S2[i]) != list(range(n)):
            return [S1]
    for j in range(n):
        if sorted([S2[i][j] for i in range(n)]) != list(range(n)):
            return [S1]
            
    # Check Orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (S1[i][j], S2[i][j])
            if p in pairs:
                return [S1]
            pairs.add(p)
            
    if len(pairs) != n*n:
        return [S1]

    return [S1, S2]
