import time

def _verify_latin(square):
    n = len(square)
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
        if sorted(square[j][i] for j in range(n)) != list(range(n)):
            return False
    return True

def _verify_orthogonal(s1, s2):
    n = len(s1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (s1[i][j], s2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n

def _solve_mols_search(n, deadline):
    s0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [s0]
    
    def try_add_square(existing_squares):
        S = [[-1 for _ in range(n)] for _ in range(n)]
        for j in range(n):
            S[0][j] = j
            
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = [set() for _ in range(len(existing_squares))]
        
        for j in range(n):
            val = S[0][j]
            row_used[0].add(val)
            col_used[j].add(val)
            for k, E in enumerate(existing_squares):
                pair_used[k].add((E[0][j], val))
                
        def backtrack(i, j):
            if time.time() > deadline:
                return False
            if i == n:
                return True
            if j == n:
                return backtrack(i + 1, 0)
            if S[i][j] != -1:
                return backtrack(i, j + 1)
                
            for val in range(n):
                if val in row_used[i] or val in col_used[j]:
                    continue
                
                valid = True
                for k, E in enumerate(existing_squares):
                    if (E[i][j], val) in pair_used[k]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                S[i][j] = val
                row_used[i].add(val)
                col_used[j].add(val)
                for k in range(len(existing_squares)):
                    pair_used[k].add((existing_squares[k][i][j], val))
                    
                if backtrack(i, j + 1):
                    return True
                    
                S[i][j] = -1
                row_used[i].remove(val)
                col_used[j].remove(val)
                for k in range(len(existing_squares)):
                    pair_used[k].remove((existing_squares[k][i][j], val))
                    
            return False
            
        if backtrack(1, 0):
            return S
        return None

    while len(squares) < n:
        if time.time() > deadline:
            break
        new_sq = try_add_square(squares)
        if new_sq is None:
            break
        squares.append(new_sq)
        
    return squares

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    try:
        squares = _solve_mols_search(10, deadline)
        if not squares:
            return [[(i+j) % 10 for j in range(10)] for i in range(10)]
        result = []
        for sq in squares:
            new_sq = [[int(sq[i][j]) for j in range(10)] for i in range(10)]
            result.append(new_sq)
        return result
    except Exception:
        return [[(i+j) % 10 for j in range(10)] for i in range(10)]
