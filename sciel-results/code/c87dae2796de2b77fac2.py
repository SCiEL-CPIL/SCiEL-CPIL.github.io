import time

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    used_pairs = set()
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    
    def backtrack(i, j):
        ni = i
        nj = j + 1
        if nj == n:
            ni += 1
            nj = 0
        if ni == n:
            return True
            
        for val in range(n):
            if val in row_used[ni] or val in col_used[nj]:
                continue
            pair = (L1[ni][nj], val)
            if pair in used_pairs:
                continue
                
            L2[ni][nj] = val
            row_used[ni].add(val)
            col_used[nj].add(val)
            used_pairs.add(pair)
            
            if backtrack(ni, nj):
                return True
                
            L2[ni][nj] = -1
            row_used[ni].remove(val)
            col_used[nj].remove(val)
            used_pairs.remove(pair)
            
        return False

    start_time = time.time()
    # Run backtracking
    if backtrack(0, 0):
        L1 = [[int(x) for x in row] for row in L1]
        L2 = [[int(x) for x in row] for row in L2]
        return [L1, L2]
    else:
        return [L1]
