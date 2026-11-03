"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    # Try rank 52 with Strassen for 2x2x2 blocks
    # If Strassen verification fails, fall back to schoolbook rank 56
    
    def try_strassen_decomposition():
        rank = 52
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        def add_strassen(a_col_base, b_row_base, c_col_base):
            nonlocal col
            A00 = a_col_base
            A01 = a_col_base + 1
            A10 = 4 + a_col_base
            A11 = 4 + a_col_base + 1
            B00 = b_row_base * 7 + c_col_base
            B01 = b_row_base * 7 + c_col_base + 1
            B10 = (b_row_base + 1) * 7 + c_col_base
            B11 = (b_row_base + 1) * 7 + c_col_base + 1
            C00 = c_col_base * 2
            C01 = (c_col_base + 1) * 2
            C10 = c_col_base * 2 + 1
            C11 = (c_col_base + 1) * 2 + 1
            
            # Using verified Strassen:
            # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B11-B01),
            # M4=A11(B00-B10), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
            # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B11][col] += 1; V[B01][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B00][col] += 1; V[B10][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_schoolbook_2x2x3(a_col_base, b_row_base, c_col_base):
            nonlocal col
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        a_idx = i * 4 + a_col_base + j
                        b_idx = (b_row_base + j) * 7 + c_col_base + k
                        c_idx = (c_col_base + k) * 2 + i
                        U[a_idx][col] += 1
                        V[b_idx][col] += 1
                        W[c_idx][col] += 1
                        col += 1
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook_2x2x3(0, 0, 4)
        add_strassen(2, 2, 0)
        add_strassen(2, 2, 2)
        add_schoolbook_2x2x3(2, 2, 4)
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if val != 1:
                        return None
        return U, V, W
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    result = try_strassen_decomposition()
    if result is None:
        U, V, W = schoolbook_decomposition()
    else:
        U, V, W = result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
