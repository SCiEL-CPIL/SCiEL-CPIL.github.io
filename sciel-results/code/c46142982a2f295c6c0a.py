import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    """
    Construct a rank-32 tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy:
    - Partition the 9 output columns into 4 pairs: (0,1), (2,3), (4,5), (6,7) and 1 single: (8).
    - For each pair (k1, k2), use Strassen's rank-7 algorithm for the 2x2 * 2x2 sub-problem.
    - For the single column k=8, use the rank-4 schoolbook algorithm for 2x2 * 2x1.
    - Total rank: 4 * 7 + 4 = 32.
    
    Index conventions:
    - U has shape (n*m, rank) = (4, rank). Index u[i*m + j].
    - V has shape (m*p, rank) = (18, rank). Index v[j*p + k].
    - W has shape (p*n, rank) = (18, rank). Index w[k*n + i].
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32
    
    current_col = 0

    def add_strassen_block(k1, k2, start_col):
        """
        Adds 7 Strassen products for the sub-problem A(2x2) * B(2x2) where B has columns k1, k2.
        Returns the next available column index.
        """
        # U indices (fixed for 2x2 A):
        # A = [[a00, a01], [a10, a11]]
        # u00 = 0, u01 = 1, u10 = 2, u11 = 3
        
        # V indices (depend on k1, k2):
        # B block = [[b0k1, b0k2], [b1k1, b1k2]]
        # v00 = 0*p + k1 = k1
        # v01 = 0*p + k2 = k2
        # v10 = 1*p + k1 = p + k1
        # v11 = 1*p + k2 = p + k2
        v00 = k1
        v01 = k2
        v10 = p + k1
        v11 = p + k2
        
        # W indices (depend on k1, k2):
        # C block = [[c0k1, c0k2], [c1k1, c1k2]]
        # w00 = k1*n + 0 = 2*k1
        # w01 = k1*n + 1 = 2*k1 + 1
        # w02 = k2*n + 0 = 2*k2
        # w03 = k2*n + 1 = 2*k2 + 1
        w00 = 2 * k1
        w01 = 2 * k1 + 1
        w02 = 2 * k2
        w03 = 2 * k2 + 1
        
        # Strassen Products:
        # M1 = (a00 + a11) * (b00 + b11)
        c = start_col
        U[0][c] += 1; U[3][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w03][c] += 1
        
        # M2 = (a10 + a11) * b00
        c = start_col + 1
        U[2][c] += 1; U[3][c] += 1
        V[v00][c] += 1
        W[w01][c] += 1; W[w03][c] -= 1
        
        # M3 = a00 * (b01 - b11)
        c = start_col + 2
        U[0][c] += 1
        V[v01][c] += 1; V[v11][c] -= 1
        W[w02][c] += 1; W[w03][c] += 1
        
        # M4 = a11 * (b10 - b00)
        c = start_col + 3
        U[3][c] += 1
        V[v10][c] += 1; V[v00][c] -= 1
        W[w00][c] += 1; W[w01][c] += 1
        
        # M5 = (a00 + a01) * b11
        c = start_col + 4
        U[0][c] += 1; U[1][c] += 1
        V[v11][c] += 1
        W[w00][c] -= 1; W[w02][c] += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        c = start_col + 5
        U[2][c] += 1; U[0][c] -= 1
        V[v00][c] += 1; V[v01][c] += 1
        W[w03][c] += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        c = start_col + 6
        U[1][c] += 1; U[3][c] -= 1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] += 1
        
        return start_col + 7

    def add_schoolbook_single(k, start_col):
        """
        Adds 4 schoolbook products for the sub-problem A(2x2) * B(2x1) where B has column k.
        Returns the next available column index.
        """
        # V indices:
        v00 = k       # b0k
        v10 = p + k   # b1k
        
        # W indices:
        w00 = 2 * k     # c0k
        w10 = 2 * k + 1 # c1k
        
        # N1 = a00 * b0k
        c = start_col
        U[0][c] += 1
        V[v00][c] += 1
        W[w00][c] += 1
        
        # N2 = a01 * b1k
        c = start_col + 1
        U[1][c] += 1
        V[v10][c] += 1
        W[w00][c] += 1
        
        # N3 = a10 * b0k
        c = start_col + 2
        U[2][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1
        
        # N4 = a11 * b1k
        c = start_col + 3
        U[3][c] += 1
        V[v10][c] += 1
        W[w10][c] += 1
        
        return start_col + 4

    # Process 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        current_col = add_strassen_block(k1, k2, current_col)
        
    # Process single column: 8
    current_col = add_schoolbook_single(8, current_col)
    
    # Convert to numpy arrays for consistency, though list of lists is often accepted.
    # The interface expects {'U', 'V', 'W'}.
    return {"U": U, "V": V, "W": W}
