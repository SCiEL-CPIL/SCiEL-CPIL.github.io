import time

def _is_latin(square, n):
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(a, b, n):
    seen = set()
    for i in range(n):
        for j in range(n):
            pair = (a[i][j], b[i][j])
            if pair in seen:
                return False
            seen.add(pair)
    return True

def _is_pairwise_orthogonal(squares, n):
    k = len(squares)
    for i in range(k):
        for j in range(i + 1, k):
            if not _are_orthogonal(squares[i], squares[j], n):
                return False
    return True

def _try_find_orthogonal_square(base_square, n, time_limit):
    """
    Attempts to find a Latin square orthogonal to base_square using backtracking.
    Returns None if not found within time_limit.
    """
    start_time = time.time()
    
    square = [[None for _ in range(n)] for _ in range(n)]
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    pair_used = set()
    
    def backtrack(pos):
        if time.time() - start_time > time_limit:
            return False
        
        if pos == n * n:
            return True
            
        i = pos // n
        j = pos % n
        
        base_val = base_square[i][j]
        
        candidates = []
        for val in range(n):
            if val in row_used[i]:
                continue
            if val in col_used[j]:
                continue
            if (base_val, val) in pair_used:
                continue
            candidates.append(val)
            
        for val in candidates:
            square[i][j] = val
            row_used[i].add(val)
            col_used[j].add(val)
            pair_used.add((base_val, val))
            
            if backtrack(pos + 1):
                return True
                
            square[i][j] = None
            row_used[i].remove(val)
            col_used[j].remove(val)
            pair_used.remove((base_val, val))
            
        return False

    if backtrack(0):
        if _is_latin(square, n) and _are_orthogonal(base_square, square, n):
            return [[int(square[i][j]) for j in range(n)] for i in range(n)]
    return None

def construct():
    n = 10
    deadline = time.time() + 55.0

    L1 = [[int((i + j) % n) for j in range(n)] for i in range(n)]

    if not _is_latin(L1, n):
        L1 = [[int((i + j) % n) for j in range(n)] for i in range(n)]

    result = [L1]

    if time.time() < deadline - 5.0:
        L2 = _try_find_orthogonal_square(L1, n, 2.0)
        if L2 is not None:
            result.append(L2)
            if not _is_pairwise_orthogonal(result, n):
                result = [L1]
        else:
            result = [L1]

    for sq in result:
        for r in range(n):
            for c in range(n):
                sq[r][c] = int(sq[r][c])

    return result
