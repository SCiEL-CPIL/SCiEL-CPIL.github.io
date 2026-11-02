import time

DIMENSIONS = [2, 2, 9]

def construct():
    """
    Construct a rank-32 exact tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    The rank-32 decomposition uses:
    - 4 blocks of Strassen's algorithm for M_{2,2,2} (rank 7 each) for column pairs (0,1), (2,3), (4,5), (6,7)
    - 1 trivial block for M_{2,2,1} (rank 4) for column 8
    Total rank = 4*7 + 4 = 32
    
    This is the known minimum rank for M_{2,2,9}.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    col = 0
    
    def add_strassen(k0, k1):
        """Add 7 Strassen factors for a 2x2 * 2x2 block with columns k0, k1."""
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2: (A10+A11)B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3: A00(B10-B11)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4: A11(B00-B01)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5: (A00+A01)B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6: (A10-A00)(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (A01-A11)(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        """Add 4 trivial factors for a 2x2 * 2x1 block with column k."""
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Column pairs (0,1), (2,3), (4,5), (6,7) use Strassen
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    # Column 8 uses trivial decomposition
    add_trivial(8)
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
