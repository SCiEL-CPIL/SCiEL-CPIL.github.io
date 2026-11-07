import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Construct two orthogonal Latin squares of order 10
    # Using linear forms over Z_10
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 2*j) % 10
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1, L2]
    
    # Verification functions
    def is_latin(sq):
        for row in sq:
            if sorted(row) != list(range(n)):
                return False
        for col in zip(*sq):
            if sorted(col) != list(range(n)):
                return False
        return True
        
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == n * n

    # Check validity of initial pair
    if not is_latin(L1) or not is_latin(L2) or not are_orthogonal(L1, L2):
        return [L1]

    # Strategy: Try to construct additional squares using linear forms
    # L_k(i,j) = (a_k * i + b_k * j) % 10
    # For orthogonality, we need the coefficient vectors to be independent
    
    # Actually, for order 10, it's known that no 3 mutually orthogonal Latin squares exist
    # This is a well-known result in combinatorial design theory
    
    # However, let's verify this by attempting to find a third square
    # through a systematic search with time limit
    
    # For each position (i,j), we need to assign a symbol such that:
    # 1. Each row has all symbols 0-9 exactly once
    # 2. Each column has all symbols 0-9 exactly once
    # 3. For each existing square, the pair (existing[i][j], new[i][j]) is unique
    
    # This is a constraint satisfaction problem. Let's use backtracking with pruning
    
    def find_third_square(existing_squares):
        # Precompute constraints
        # For each (i,j), track which symbols are forbidden by orthogonality
        forbidden = [[set() for _ in range(n)] for _ in range(n)]

        for sq in existing_squares:
            seen_pairs = set()
            for i in range(n):
                for j in range(n):
                    val = sq[i][j]
                    # For each symbol s, if (val, s) was seen before, s is forbidden at (i,j)
                    # But we don't know s yet, so we track differently

        # Better approach: track for each (i,j) which (existing_val, new_val) pairs are used
        # Actually, for orthogonality with a single square, we need that for each symbol v in that square,
        # the new square takes each symbol exactly once in the positions where that square has v

        # Let's use a different approach: backtracking with constraint propagation

        grid = [[-1 for _ in range(n)] for _ in range(n)]

        # Precompute for each row, which symbols are still available
        row_avail = [set(range(n)) for _ in range(n)]
        col_avail = [set(range(n)) for _ in range(n)]

        # For orthogonality constraints, for each existing square, track which (existing_val, new_val)
        # pairs have been used
        ortho_used = [set() for _ in existing_squares]

        # For each existing square, for each value v, track which new values have been used
        # at positions where the existing square has value v
        ortho_val_used = [[set() for _ in range(n)] for _ in existing_squares]

        def can_place(i, j, val):
            if val not in row_avail[i]:
                return False
            if val not in col_avail[j]:
                return False
            for k, sq in enumerate(existing_squares):
                ex_val = sq[i][j]
                if (ex_val, val) in ortho_used[k]:
                    return False
            return True

        def place(i, j, val):
            grid[i][j] = val
            row_avail[i].remove(val)
            col_avail[j].remove(val)
            for k, sq in enumerate(existing_squares):
                ex_val = sq[i][j]
                ortho_used[k].add((ex_val, val))
                ortho_val_used[k][ex_val].add(val)

        def unplace(i, j, val):
            grid[i][j] = -1
            row_avail[i].add(val)
            col_avail[j].add(val)
            for k, sq in enumerate(existing_squares):
                ex_val = sq[i][j]
                ortho_used[k].remove((ex_val, val))
                ortho_val_used[k][ex_val].remove(val)

        def backtrack(i, j):
            if time.time() - start_time > deadline:
                return False

            if i == n:
                return True

            if j == n:
                return backtrack(i + 1, 0)

            # Try to place a value at (i, j)
            for val in sorted(row_avail[i] & col_avail[j]):
                if can_place(i, j, val):
                    place(i, j, val)
                    if backtrack(i, j + 1):
                        return True
                    unplace(i, j, val)

            return False

        # Start backtracking
        if backtrack(0, 0):
            return [row[:] for row in grid]
        return None

    # Try to find a third square
    if time.time() - start_time < deadline:
        third_sq = find_third_square(squares)
        if third_sq is not None:
            # Verify it's valid
            if is_latin(third_sq) and all(are_orthogonal(third_sq, sq) for sq in squares):
                squares.append(third_sq)

    # Try to find a fourth square if we have time
    if len(squares) >= 3 and time.time() - start_time < deadline:
        fourth_sq = find_third_square(squares)
        if fourth_sq is not None:
            if is_latin(fourth_sq) and all(are_orthogonal(fourth_sq, sq) for sq in squares):
                squares.append(fourth_sq)

    # Return the squares found
    return squares
