import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition by splitting:
    # n=3 -> [2, 1]
    # m=4 -> [2, 2]  
    # p=7 -> [4, 3] -> [2,2] + [2,1]
    
    # Sub-blocks and their ranks:
    # M_{2,2,2}: Strassen rank 7
    # M_{2,2,1}: schoolbook rank 4
    # M_{1,2,2}: schoolbook rank 4
    # M_{1,2,1}: schoolbook rank 2
    
    # Total rank = 2*M_{2,2,4} + 2*M_{2,2,3} + 2*M_{1,2,4} + 2*M_{1,2,3}
    #           = 2*(2*7) + 2*(7+4) + 2*(4*2) + 2*(3*2)
    #           = 28 + 22 + 16 + 12 = 78
    
    rank_total = 78
    
    U = [[0] * rank_total for _ in range(n * m)]  # (12, 78)
    V = [[0] * rank_total for _ in range(m * p)]  # (28, 78)
    W = [[0] * rank_total for _ in range(p * n)]  # (21, 78)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, start_col):
        """Add Strassen decomposition for M_{2,2,2} at offset (i0,j0,k0).
        Returns the next column index."""
        # Strassen: 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U rows: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # V rows: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # W rows: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        
        u_rows = [
            i0 * m + j0,       # A11
            i0 * m + j0 + 1,   # A12
            (i0+1) * m + j0,   # A21
            (i0+1) * m + j0 + 1  # A22
        ]
        v_rows = [
            j0 * p + k0,       # B11
            j0 * p + k0 + 1,   # B12
            (j0+1) * p + k0,   # B21
            (j0+1) * p + k0 + 1  # B22
        ]
        w_rows = [
            k0 * n + i0,       # C11
            k0 * n + i0 + 1,   # C12
            (k0+1) * n + i0,   # C21
            (k0+1) * n + i0 + 1  # C22
        ]
        
        A11, A12, A21, A22 = u_rows
        B11, B12, B21, B22 = v_rows
        C11, C12, C21, C22 = w_rows
        
        # M1 = (A11+A22)(B11+B22)
        c = start_col
        U[A11][c] = 1; U[A22][c] = 1
        V[B11][c] = 1; V[B22][c] = 1
        W[C11][c] = 1; W[C22][c] = 1
        c += 1
        
        # M2 = (A21+A22)B11
        U[A21][c] = 1; U[A22][c] = 1
        V[B11][c] = 1
        W[C21][c] = 1; W[C22][c] = -1
        c += 1
        
        # M3 = A11(B21-B22)
        U[A11][c] = 1
        V[B21][c] = 1; V[B22][c] = -1
        W[C12][c] = 1; W[C22][c] = 1
        c += 1
        
        # M4 = A22(B12-B11)
        U[A22][c] = 1
        V[B12][c] = 1; V[B11][c] = -1
        W[C11][c] = 1; W[C21][c] = 1
        c += 1
        
        # M5 = (A11+A12)B22
        U[A11][c] = 1; U[A12][c] = 1
        V[B22][c] = 1
        W[C11][c] = -1; W[C12][c] = 1
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[A21][c] = 1; U[A11][c] = -1
        V[B11][c] = 1; V[B12][c] = 1
        W[C22][c] = 1
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[A12][c] = 1; U[A22][c] = -1
        V[B21][c] = 1; V[B22][c] = 1
        W[C11][c] = 1; W[C22][c] = -1
        c += 1
        
        return c
    
    def add_schoolbook_221(i0, j0, k0, start_col):
        """Add schoolbook for M_{2,2,1} at offset (i0,j0,k0). Rank 4."""
        for di in range(2):
            for dj in range(2):
                c = start_col + di * 2 + dj
                u_idx = (i0 + di) * m + (j0 + dj)
                v_idx = (j0 + dj) * p + k0
                w_idx = k0 * n + (i0 + di)
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
        return start_col + 4
    
    def add_schoolbook_122(i0, j0, k0, start_col):
        """Add schoolbook for M_{1,2,2} at offset (i0,j0,k0). Rank 4."""
        for dj in range(2):
            for dk in range(2):
                c = start_col + dj * 2 + dk
                u_idx = i0 * m + (j0 + dj)
                v_idx = (j0 + dj) * p + (k0 + dk)
                w_idx = (k0 + dk) * n + i0
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
        return start_col + 4
    
    def add_schoolbook_121(i0, j0, k0, start_col):
        """Add schoolbook for M_{1,2,1} at offset (i0,j0,k0). Rank 2."""
        for dj in range(2):
            c = start_col + dj
            u_idx = i0 * m + (j0 + dj)
            v_idx = (j0 + dj) * p + k0
            w_idx = k0 * n + i0
            U[u_idx][c] = 1
            V[v_idx][c] = 1
            W[w_idx][c] = 1
        return start_col + 2
    
    # Now fill in the blocks:
    # i in {0,1} (top 2 rows), j in {0,1} (first 2 cols), k in {0,1} (first 2 cols of B)
    col = add_strassen_222(0, 0, 0, col)
    
    # i in {0,1}, j in {0,1}, k in {2,3}
    col = add_strassen_222(0, 0, 2, col)
    
    # i in {0,1}, j in {0,1}, k in {4,5}
    col = add_strassen_222(0, 0, 4, col)
    
    # i in {0,1}, j in {0,1}, k in {6}
    col = add_schoolbook_221(0, 0, 6, col)
    
    # i in {0,1}, j in {2,3}, k in {0,1}
    col = add_strassen_222(0, 2, 0, col)
    
    # i in {0,1}, j in {2,3}, k in {2,3}
    col = add_strassen_222(0, 2, 2, col)
    
    # i in {0,1}, j in {2,3}, k in {4,5}
    col = add_strassen_222(0, 2, 4, col)
    
    # i in {0,1}, j in {2,3}, k in {6}
    col = add_schoolbook_221(0, 2, 6, col)
    
    # i in {2}, j in {0,1}, k in {0,1}
    col = add_schoolbook_122(2, 0, 0, col)
    
    # i in {2}, j in {0,1}, k in {2,3}
    col = add_schoolbook_122(2, 0, 2, col)
    
    # i in {2}, j in {0,1}, k in {4,5}
    col = add_schoolbook_122(2, 0, 4, col)
    
    # i in {2}, j in {0,1}, k in {6}
    col = add_schoolbook_121(2, 0, 6, col)
    
    # i in {2}, j in {2,3}, k in {0,1}
    col = add_schoolbook_122(2, 2, 0, col)
    
    # i in {2}, j in {2,3}, k in {2,3}
    col = add_schoolbook_122(2, 2, 2, col)
    
    # i in {2}, j in {2,3}, k in {4,5}
    col = add_schoolbook_122(2, 2, 4, col)
    
    # i in {2}, j in {2,3}, k in {6}
    col = add_schoolbook_121(2, 2, 6, col)
    
    # col should be 78
    assert col == rank_total, f"Expected col={rank_total}, got {col}"
    
    if time.time() > deadline:
        pass  # Shouldn't happen, but safety
    
    return {"U": U, "V": V, "W": W}
