def construct():
    n = 10
    squares = []
    
    # Generate 4 squares based on Z5 slopes
    # v = 5 * v2 + v5
    # v2 = (r2 + c2) % 2
    # v5 = (r5 + k * c5) % 5
    for k in [1, 2, 3, 4]:
        L = [[0] * n for _ in range(n)]
        for r in range(n):
            r5 = r % 5
            r2 = r // 5
            for c in range(n):
                c5 = c % 5
                c2 = c // 5
                v5 = (r5 + k * c5) % 5
                v2 = (r2 + c2) % 2
                L[r][c] = 5 * v2 + v5
        squares.append(L)
        
    # Verify Latin property for all
    for L in squares:
        for r in range(n):
            if sorted(L[r]) != list(range(n)):
                return squares[:1] # Fallback to at least 1 if broken, but they should be fine
        for c in range(n):
            col = [L[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return squares[:1]

    # Verify mutual orthogonality
    valid_squares = [squares[0]]
    for i in range(1, len(squares)):
        candidate = squares[i]
        is_orth = True
        for j in range(len(valid_squares)):
            other = valid_squares[j]
            pairs = set()
            for r in range(n):
                for c in range(n):
                    p = (other[r][c], candidate[r][c])
                    if p in pairs:
                        is_orth = False
                        break
                    pairs.add(p)
            if not is_orth:
                break
        if is_orth:
            valid_squares.append(candidate)
            
    # Attempt to add 5 more squares based on cross terms
    # v5 = (r5 + c5 + j * c2) % 5
    # v2 = (r2 + c2) % 2
    for j in [1, 2, 3, 4]:
        L = [[0] * n for _ in range(n)]
        for r in range(n):
            r5 = r % 5
            r2 = r // 5
            for c in range(n):
                c5 = c % 5
                c2 = c // 5
                v5 = (r5 + c5 + j * c2) % 5
                v2 = (r2 + c2) % 2
                L[r][c] = 5 * v2 + v5
        squares.append(L)
        
    # And one more with v2 variation
    L = [[0] * n for _ in range(n)]
    for r in range(n):
        r5 = r % 5
        r2 = r // 5
        for c in range(n):
            c5 = c % 5
            c2 = c // 5
            v5 = (r5 + c5) % 5
            v2 = (r2 + c2 + c5) % 2
            L[r][c] = 5 * v2 + v5
    squares.append(L)

    # Re-verify and add if orthogonal to all existing
    for candidate in squares[4:]:
        is_orth = True
        for j in range(len(valid_squares)):
            other = valid_squares[j]
            pairs = set()
            for r in range(n):
                for c in range(n):
                    p = (other[r][c], candidate[r][c])
                    if p in pairs:
                        is_orth = False
                        break
                    pairs.add(p)
            if not is_orth:
                break
        if is_orth:
            valid_squares.append(candidate)
            
    return valid_squares
