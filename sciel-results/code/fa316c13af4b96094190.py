import time
import random

def construct():
    n = 10
    # S1: Standard cyclic Latin square
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second square S2 orthogonal to S1
    # S2 must be a Latin square such that pairs (S1[i][j], S2[i][j]) are unique.
    # We can construct S2 using a backtracking algorithm.
    # To save time, we use a heuristic: random restarts.
    
    deadline = time.time() + 50  # Safety margin
    
    def is_orthogonal(S1, S2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (S1[i][j], S2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    def is_latin(S):
        for i in range(n):
            if len(set(S[i])) != n:
                return False
            col = [S[k][i] for k in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def backtrack(S2, row, col, used_rows, used_cols, used_pairs):
        if row == n - 1 and col == n - 1:
            return True
        next_row = row + 1 if col == n - 1 else row
        next_col = 0 if col == n - 1 else col + 1
        
        for val in range(n):
            if val in used_rows[row]:
                continue
            if val in used_cols[col]:
                continue
            if (S1[row][col], val) in used_pairs:
                continue
            
            S2[row][col] = val
            used_rows[row].add(val)
            used_cols[col].add(val)
            used_pairs.add((S1[row][col], val))
            
            if backtrack(S2, next_row, next_col, used_rows, used_cols, used_pairs):
                return True
            
            used_rows[row].remove(val)
            used_cols[col].remove(val)
            used_pairs.remove((S1[row][col], val))
            S2[row][col] = -1
            
        return False

    # Try to find one mate
    found = False
    attempts = 0
    while time.time() < deadline and not found:
        attempts += 1
        # Randomize the order of values to escape local minima?
        # For now, simple backtracking might be fast enough for 1 mate.
        S2 = [[-1 for _ in range(n)] for _ in range(n)]
        used_rows = [set() for _ in range(n)]
        used_cols = [set() for _ in range(n)]
        used_pairs = set()
        
        # Heuristic: Fill the first row with a permutation
        perm = list(range(n))
        random.shuffle(perm)
        for j in range(n):
            S2[0][j] = perm[j]
            used_rows[0].add(perm[j])
            used_cols[j].add(perm[j])
            used_pairs.add((S1[0][j], perm[j]))
            
        if backtrack(S2, 1, 0, used_rows, used_cols, used_pairs):
            found = True
            break
            
        if attempts > 100: # Give up if it's taking too long
            break

    if found:
        return [S1, S2]
    else:
        return [S1]
