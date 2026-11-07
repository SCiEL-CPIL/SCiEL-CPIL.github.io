import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    n = 10
    # Square 1
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find Square 2 orthogonal to Square 1
    # Use backtracking
    sq2 = [[-1 for _ in range(n)] for _ in range(n)]
    pairs_used = set()
    
    # Precompute allowed values for each cell based on sq1 and Latin constraints
    # This is complex to set up perfectly quickly.
    # Simpler: Randomized restarts with backtracking.
    
    import random
    
    def backtrack(i, j):
        if time.time() > deadline:
            return False
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        # Try all symbols 0..9
        # Shuffle for randomness
        symbols = list(range(n))
        random.shuffle(symbols)
        
        for sym in symbols:
            # Check Latin constraints for sq2
            if sym in sq2[i]:
                continue
            if any(sq2[r][j] == sym for r in range(n)):
                continue
            
            # Check Orthogonality constraint
            pair = (sq1[i][j], sym)
            if pair in pairs_used:
                continue
            
            # Place
            sq2[i][j] = sym
            pairs_used.add(pair)
            
            if backtrack(i, j + 1):
                return True
            
            # Undo
            sq2[i][j] = -1
            pairs_used.remove(pair)
            
        return False

    # Run search
    found = False
    for _ in range(100): # Limit restarts
        if time.time() > deadline:
            break
        random.seed()
        sq2 = [[-1 for _ in range(n)] for _ in range(n)]
        pairs_used = set()
        if backtrack(0, 0):
            found = True
            break
            
    if found:
        return [sq1, sq2]
    else:
        return [sq1]
