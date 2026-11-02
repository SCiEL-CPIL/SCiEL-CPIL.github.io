<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But tensor rank can be lower than this in some cases
    # For matrix multiplication, known bounds suggest rank around n*m*p/2 to n*m*p
    
    best_result = None
    
    # First, get schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank
    for r in range(10, rank_school):
        if time.time() > deadline:
            break
        
        result = _try_rank_search(n, m, p, r, timeout=min(30, deadline - time.time()))
        if result:
            best_result = result
            break  # Found a valid decomposition, return it
    
    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {'U': U_school, 'V': V_school, 'W': W_school}
    
    lower_bound = max(n*m, m*p, p*n)
    for r in range(lower_bound, rank_school):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        result = _try_rank_numerical(n, m, p, r, timeout=min(40, remaining))
        if result:
            best_result = result
            break
    
    return best_result
>>>>>>> REPLACE
