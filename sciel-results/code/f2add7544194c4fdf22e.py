import time

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    L2 = [[None for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    deadline = time.time() + 50
    
    def backtrack(i, j):
        if time.time() > deadline:
            return False
            
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
            
        if L2[i][j] is not None:
            return backtrack(i, j + 1)
            
        v1 = L1[i][j]
        # Try values in a specific order?
        # Randomization might help avoid getting stuck in a bad branch?
        # But deterministic is better for reproducibility.
        for v in range(n):
            if v in row_used[i] or v in col_used[j]:
                continue
            if (v1, v) in pair_used:
                continue
            
            L2[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pair_used.add((v1, v))
            
            if backtrack(i, j + 1):
                return True
            
            L2[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            pair_used.remove((v1, v))
            
        return False

    found = backtrack(0, 0)
    
    if found:
        # Convert to list of lists of ints
        L2_int = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
        L1_int = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        
        # Final sanity check
        # Check Latin
        for sq in [L1_int, L2_int]:
            for i in range(n):
                if sorted(sq[i]) != list(range(n)):
                    return [L1_int]
            for j in range(n):
                col = [sq[i][j] for i in range(n)]
                if sorted(col) != list(range(n)):
                    return [L1_int]
            
        # Check Orthogonality
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (L1_int[i][j], L2_int[i][j])
                if p in pairs:
                    return [L1_int]
                pairs.add(p)
                
        return [L1_int, L2_int]
    else:
        L1_int = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        return [L1_int]
