import time

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = 72
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(i, j, k, coeff=1):
        """Add a single rank-1 term to U, V, W."""
        col = add_term.c
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += coeff
        add_term.c += 1
    
    add_term.c = 0
    
    # Row 1 (rows 0-2 of A, rows 0-4 of B): 2x2 + 2x3 blocks
    # A blocks: A00(0-1,0-1), A01(0-1,2-4), A10(2,0-1), A11(2,2-4)
    # B blocks: B00(0-1,0-1), B01(0-1,2-4), B10(2,0-1), B11(2,2-4)
    
    # M12 = (A00+A11)(B00+B11): rank 4
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(2, j, 0, 1)
            add_term(2, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(2, j, 0, 1)
            add_term(2, j, 1, 1)
    
    # M21 = (A10-A00)(B00+B11): rank 6
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
    
    # M23 = (A10-A00)(B11-B00): rank 6
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 2, -1)
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 2, -1)
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 2, -1)
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    
    # M31 = (A00+A11)(B00+B11): rank 6
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
    
    # M33 = (A11+A00-A10)(B11+B00-B01): rank 6
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
    for i in [0, 1]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 2, 1)
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [2]:
        for j in [0, 1]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
    
    # Row 2 (rows 3-4 of A, rows 0-4 of B): 1x3 + 1x2 blocks
    # A blocks: A00(3,0-2), A01(3,3-4), A10(4,0-2), A11(4,3-4)
    # B blocks: B00(0-2,0-2), B01(0-2,3-4), B10(3-4,0-2), B11(3-4,3-4)
    
    # M12 = (A00+A11)(B00+B11): rank 6
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    
    # M21 = (A10-A00)(B00+B11): rank 4
    for i in [4]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [3]:
        for j in [3, 4]:
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    
    # M23 = (A10-A00)(B11-B00): rank 4
    for i in [4]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [3]:
        for j in [3, 4]:
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    
    # M31 = (A00+A11)(B00+B11): rank 4
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    
    # M33 = (A11+A00-A10)(B11+B00-B01): rank 4
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [4]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [3]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [4]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [4]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    
    # Row 3 (rows 5-6 of A, rows 0-4 of B): 1x3 + 1x2 blocks
    # A blocks: A00(5,0-2), A01(5,3-4), A10(6,0-2), A11(6,3-4)
    # B blocks: B00(0-2,0-2), B01(0-2,3-4), B10(3-4,0-2), B11(3-4,3-4)
    
    # M12 = (A00+A11)(B00+B11): rank 6
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    
    # M21 = (A10-A00)(B00+B11): rank 4
    for i in [6]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [5]:
        for j in [3, 4]:
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    
    # M23 = (A10-A00)(B11-B00): rank 4
    for i in [6]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [5]:
        for j in [3, 4]:
            add_term(i, j, 3, -1)
            add_term(i, j, 4, -1)
    
    # M31 = (A00+A11)(B00+B11): rank 4
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    
    # M33 = (A11+A00-A10)(B11+B00-B01): rank 4
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [6]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    for i in [5]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, 1)
            add_term(i, j, 1, 1)
            add_term(i, j, 2, 1)
    for i in [6]:
        for j in [3, 4]:
            add_term(i, j, 3, 1)
            add_term(i, j, 4, 1)
    for i in [6]:
        for j in [0, 1, 2]:
            add_term(i, j, 0, -1)
            add_term(i, j, 1, -1)
            add_term(i, j, 2, -1)
    
    assert add_term.c == rank, f"Expected rank {rank}, got {add_term.c}"
    
    return {"U": U, "V": V, "W": W}

