import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe within 90s
    
    # S0: (i+j)%10
    S0 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # We try to find S1 orthogonal to S0
    # S1 is 10x10, initialized to -1
    S1 = [[-1 for _ in range(10)] for _ in range(10)]
    
    row_used = [0] * 10  # bitmask for symbols used in each row
    col_used = [0] * 10  # bitmask for symbols used in each col
    pairs_used = set()   # set of (s0_val, s1_val)
    
    # Precompute S0 values for quick access
    # s0_val[i][j] = S0[i][j]
    
    def is_valid(i, j, s):
        # Check row
        if row_used[i] & (1 << s):
            return False
        # Check col
        if col_used[j] & (1 << s):
            return False
        # Check orthogonality
        pair = (S0[i][j], s)
        if pair in pairs_used:
            return False
        return True
        
    def place(i, j, s):
        S1[i][j] = s
        row_used[i] |= (1 << s)
        col_used[j] |= (1 << s)
        pairs_used.add((S0[i][j], s))
        
    def unplace(i, j, s):
        S1[i][j] = -1
        row_used[i] &= ~(1 << s)
        col_used[j] &= ~(1 << s)
        pairs_used.remove((S0[i][j], s))
        
    # Backtracking solver
    # We fill cell by cell.
    # To speed up, we can choose the next cell with the fewest available candidates.
    # However, for simplicity and speed in Python, a fixed order with good pruning might suffice.
    # Let's try row-major order first, but with forward checking.
    
    # Better: Use a queue of empty cells, pick the one with min candidates.
    # This is expensive to compute every time.
    # Let's stick to a simple backtracking with a time check.
    
    # We'll use a recursive function.
    # To avoid recursion depth issues, we can use an iterative stack, but 100 depth is fine for Python (default limit 1000).
    
    # Optimization: Precompute candidates for each cell?
    # No, candidates change as we place symbols.
    
    # Let's use a simple recursive backtracker.
    
    def backtrack(count):
        if time.time() > deadline:
            return False
            
        if count == 100:
            return True
            
        # Find the next empty cell
        # To improve performance, we can look for the cell with the fewest valid options.
        # Let's implement a simple heuristic: pick the first empty cell.
        # This might be slow. Let's pick the cell with min candidates.
        
        min_candidates = 11
        best_i = -1
        best_j = -1
        best_candidates = []
        
        for i in range(10):
            for j in range(10):
                if S1[i][j] == -1:
                    candidates = []
                    for s in range(10):
                        if is_valid(i, j, s):
                            candidates.append(s)
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_i = i
                        best_j = j
                        best_candidates = candidates
                        if min_candidates == 0:
                            return False
                        if min_candidates == 1:
                            # Can't get better than 1, break early?
                            # But we need to check if there are other cells with 0?
                            # No, if one has 1, others might have 0.
                            # So we must scan all.
                            pass
        
        if best_i == -1:
            # No empty cells? Should not happen if count < 100
            return count == 100
            
        for s in best_candidates:
            place(best_i, best_j, s)
            if backtrack(count + 1):
                return True
            unplace(best_i, best_j, s)
            
        return False

    found = False
    try:
        # Use a non-recursive approach or ensure recursion limit is high enough
        # 100 depth is fine.
        found = backtrack(0)
    except RecursionError:
        pass

    if found:
        # Verify the solution
        # Check Latin
        for i in range(10):
            if sorted(S1[i]) != list(range(10)):
                found = False
                break
        if found:
            for j in range(10):
                if sorted([S1[i][j] for i in range(10)]) != list(range(10)):
                    found = False
                    break
        if found:
            # Check Orthogonality
            pairs = set()
            for i in range(10):
                for j in range(10):
                    pairs.add((S0[i][j], S1[i][j]))
            if len(pairs) != 100:
                found = False
        
    if found:
        return [S0, S1]
    else:
        return [S0]

