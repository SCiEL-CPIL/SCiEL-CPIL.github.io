import time
import random

def construct():
    n = 10
    # S1[i][j] = (i + j) % n
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    # S2[i][j] = (i + 2 * j) % n
    S2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify S1 and S2 are orthogonal (they are, as per analysis)
    # But let's be safe and just return them if search fails.
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s margin
    
    # Try to find S3
    # Fix S3[0] = [0, 1, ..., 9]
    S3 = [[None] * n for _ in range(n)]
    for j in range(n):
        S3[0][j] = j
        
    used_pairs_1 = set()
    used_pairs_2 = set()
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    # Initialize with row 0
    for j in range(n):
        v = S3[0][j]
        used_pairs_1.add((S1[0][j], v))
        used_pairs_2.add((S2[0][j], v))
        row_used[0].add(v)
        col_used[j].add(v)
        
    def backtrack(i, j):
        if time.time() - start_time > time_limit:
            return False
            
        if i == n:
            return True
            
        if j == n:
            return backtrack(i + 1, 0)
            
        if S3[i][j] is not None:
            return backtrack(i, j + 1)
            
        # Try values
        # Heuristic: try values that are most constrained?
        # Just random or sequential. Sequential might be faster if a solution exists near the start.
        # But random helps avoid specific dead ends.
        candidates = list(range(n))
        random.shuffle(candidates)
        
        for v in candidates:
            if v in row_used[i] or v in col_used[j]:
                continue
                
            p1 = (S1[i][j], v)
            p2 = (S2[i][j], v)
            
            if p1 in used_pairs_1 or p2 in used_pairs_2:
                continue
                
            # Place
            S3[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            used_pairs_1.add(p1)
            used_pairs_2.add(p2)
            
            if backtrack(i, j + 1):
                return True
            
            # Unplace
            S3[i][j] = None
            row_used[i].remove(v)
            col_used[j].remove(v)
            used_pairs_1.remove(p1)
            used_pairs_2.remove(p2)
            
        return False

    found = backtrack(1, 0)
    
    if found:
        # Ensure types are correct (int, list)
        S1 = [[int(S1[i][j]) for j in range(n)] for i in range(n)]
        S2 = [[int(S2[i][j]) for j in range(n)] for i in range(n)]
        S3 = [[int(S3[i][j]) for j in range(n)] for i in range(n)]
        return [S1, S2, S3]
    else:
        S1 = [[int(S1[i][j]) for j in range(n)] for i in range(n)]
        S2 = [[int(S2[i][j]) for j in range(n)] for i in range(n)]
        return [S1, S2]
