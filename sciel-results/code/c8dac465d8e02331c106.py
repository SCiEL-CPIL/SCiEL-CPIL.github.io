# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=tol):
        return False
    return True

def _strassen_3x3_factors():
    """
    Returns (U, V, W) for 3x3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Index convention: row index = i*3+j for U, j*3+k for V, k*3+i for W
    """
    # Build the 23 Strassen terms for 3x3x3
    # Each term c has: U[:,c] (coefficients on A), V[:,c] (coefficients on B), W[:,c] (coefficients on C)
    # A[i,j] is at U[i*3+j, c]
    # B[j,k] is at V[j*3+k, c]
    # C[k,i] is at W[k*3+i, c]
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper to add coefficient
    def add_U(i, j, c, val): U[i*3+j, c] = val
    def add_V(j, k, c, val): V[j*3+k, c] = val
    def add_W(k, i, c, val): W[k*3+i, c] = val
    
    # Term 0: M0 = (A00+A22)*(B00+B22)
    # C00 += M0, C02 += M0, C20 += M0, C22 += M0
    add_U(0,0,0,1); add_U(2,2,0,1)
    add_V(0,0,0,1); add_V(2,2,0,1)
    add_W(0,0,0,1); add_W(2,0,0,1); add_W(0,2,0,1); add_W(2,2,0,1)
    
    # Term 1: M1 = (A20+A22)*B00
    # C00 -= M1, C20 += M1, C22 -= M1
    add_U(2,0,1,1); add_U(2,2,1,1)
    add_V(0,0,1,1)
    add_W(0,0,1,-1); add_W(2,0,1,1); add_W(2,2,1,-1)
    
    # Term 2: M2 = A00*(B01-B21)
    # C01 += M2, C21 -= M2
    add_U(0,0,2,1)
    add_V(0,1,2,1); add_V(2,1,2,-1)
    add_W(1,0,2,1); add_W(1,2,2,-1)
    
    # Term 3: M3 = A22*(B20-B00)
    # C20 -= M3, C22 += M3
    add_U(2,2,3,1)
    add_V(2,0,3,1); add_V(0,0,3,-1)
    add_W(0,2,3,-1); add_W(2,2,3,1)
    
    # Term 4: M4 = (A00+A01)*(B00+B01)
    # C00 -= M4, C01 += M4
    add_U(0,0,4,1); add_U(0,1,4,1)
    add_V(0,0,4,1); add_V(0,1,4,1)
    add_W(0,0,4,-1); add_W(1,0,4,1)
    
    # Term 5: M5 = (A20+A21)*(B20+B21)
    # C20 += M5, C21 += M5
    add_U(2,0,5,1); add_U(2,1,5,1)
    add_V(2,0,5,1); add_V(2,1,5,1)
    add_W(0,2,5,1); add_W(1,2,5,1)
    
    # Term 6: M6 = (A01-A21)*(B11+B22)
    # C01 -= M6, C02 += M6, C21 += M6, C22 -= M6
    add_U(0,1,6,1); add_U(2,1,6,-1)
    add_V(1,1,6,1); add_V(2,2,6,1)
    add_W(1,0,6,-1); add_W(2,0,6,1); add_W(1,2,6,1); add_W(2,2,6,-1)
    
    # Term 7: M7 = (A10-A22)*(B00+B11)
    # C00 += M7, C11 -= M7, C22 += M7
    add_U(1,0,7,1); add_U(2,2,7,-1)
    add_V(0,0,7,1); add_V(1,1,7,1)
    add_W(0,0,7,1); add_W(1,1,7,-1); add_W(2,2,7,1)
    
    # Term 8: M8 = (A00+A11)*(B11+B22)
    # C02 -= M8, C11 += M8
    add_U(0,0,8,1); add_U(1,1,8,1)
    add_V(1,1,8,1); add_V(2,2,8,1)
    add_W(2,0,8,-1); add_W(1,1,8,1)
    
    # Term 9: M9 = (A11+A22)*(B00+B11)
    # C00 += M9, C11 += M9
    add_U(1,1,9,1); add_U(2,2,9,1)
    add_V(0,0,9,1); add_V(1,1,9,1)
    add_W(0,0,9,1); add_W(1,1,9,1)
    
    # Term 10: M10 = (A10+A11)*(B10+B11)
    # C10 += M10, C11 += M10
    add_U(1,0,10,1); add_U(1,1,10,1)
    add_V(1,0,10,1); add_V(1,1,10,1)
    add_W(0,1,10,1); add_W(1,1,10,1)
    
    # Term 11: M11 = (A11+A21)*(B21+B22)
    # C12 += M11, C21 += M11
    add_U(1,1,11,1); add_U(2,1,11,1)
    add_V(2,1,11,1); add_V(2,2,11,1)
    add_W(2,1,11,1); add_W(1,2,11,1)
    
    # Term 12: M12 = (A00+A02)*(B02+B22)
    # C02 += M12
    add_U(0,0,12,1); add_U(0,2,12,1)
    add_V(0,2,12,1); add_V(2,2,12,1)
    add_W(2,0,12,1)
    
    # Term 13: M13 = (A20+A22)*(B02+B22)
    # C02 += M13, C22 += M13
    add_U(2,0,13,1); add_U(2,2,13,1)
    add_V(0,2,13,1); add_V(2,2,13,1)
    add_W(2,0,13,1); add_W(2,2,13,1)
    
    # Term 14: M14 = (A02-A22)*(B00+B22)
    # C00 += M14, C02 -= M14, C22 += M14
    add_U(0,2,14,1); add_U(2,2,14,-1)
    add_V(0,0,14,1); add_V(2,2,14,1)
    add_W(0,0,14,1); add_W(2,0,14,-1); add_W(2,2,14,1)
    
    # Term 15: M15 = (A02+A01)*(B20+B21)
    # C01 += M15, C02 += M15
    add_U(0,2,15,1); add_U(0,1,15,1)
    add_V(2,0,15,1); add_V(2,1,15,1)
    add_W(1,0,15,1); add_W(2,0,15,1)
    
    # Term 16: M16 = (A10+A20)*(B12+B22)
    # C12 += M16, C22 += M16
    add_U(1,0,16,1); add_U(2,0,16,1)
    add_V(1,2,16,1); add_V(2,2,16,1)
    add_W(2,1,16,1); add_W(2,2,16,1)
    
    # Term 17: M17 = (A12+A22)*(B00+B10)
    # C00 += M17, C10 += M17, C12 -= M17
    add_U(1,2,17,1); add_U(2,2,17,1)
    add_V(0,0,17,1); add_V(1,0,17,1)
    add_W(0,0,17,1); add_W(0,1,17,1); add_W(2,1,17,-1)
    
    # Term 18: M18 = (A12-A10)*(B11+B21)
    # C11 += M18, C12 += M18
    add_U(1,2,18,1); add_U(1,0,18,-1)
    add_V(1,1,18,1); add_V(2,1,18,1)
    add_W(1,1,18,1); add_W(2,1,18,1)
    
    # Term 19: M19 = (A11+A12)*(B10+B20)
    # C10 += M19, C12 += M19
    add_U(1,1,19,1); add_U(1,2,19,1)
    add_V(1,0,19,1); add_V(2,0,19,1)
    add_W(0,1,19,1); add_W(2,1,19,1)
    
    # Term 20: M20 = (A01+A11)*(B02+B12)
    # C02 += M20, C12 += M20
    add_U(0,1,20,1); add_U(1,1,20,1)
    add_V(0,2,20,1); add_V(1,2,20,1)
    add_W(2,0,20,1); add_W(2,1,20,1)
    
    # Term 21: M21 = (A10+A12)*(B01+B11)
    # C01 += M21, C11 += M21
    add_U(1,0,21,1); add_U(1,2,21,1)
    add_V(0,1,21,1); add_V(1,1,21,1)
    add_W(1,0,21,1); add_W(1,1,21,1)
    
    # Term 22: M22 = (A01+A21)*(B02+B12)
    # C02 += M22, C12 += M22
    add_U(0,1,22,1); add_U(2,1,22,1)
    add_V(0,2,22,1); add_V(1,2,22,1)
    add_W(2,0,22,1); add_W(2,1,22,1)
    
    # Convert to list of lists for interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return U_list, V_list, W_list

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 6
    
    # Try Strassen-based decomposition: rank 46
    # Split B (3x6) into B1 (3x3) and B2 (3x3)
    # C = A*B1 + A*B2
    # Use independent 3x3 Strassen for each half
    
    try:
        U3, V3, W3 = _strassen_3x3_factors()
        
        # For 3x3x6:
        # U: (9, 46) - A coefficients, same for both halves
        # V: (18, 46) - B coefficients, first 9 rows for B1, next 9 for B2
        # W: (18, 46) - C coefficients, first 9 rows for C1 (cols 0-2), next 9 for C2 (cols 3-5)
        
        rank = 46
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        
        for c in range(23):
            # First half: columns 0-2 of B and C
            for i in range(9):
                U[i][c] = U3[i][c]
            for j in range(9):
                V[j][c] = V3[j][c]  # B1 rows
                W[j][c] = W3[j][c]  # C1 rows (k*3+i where k is 0,1,2 -> cols 0,1,2)
            
            # Second half: columns 3-5 of B and C
            c2 = c + 23
            for i in range(9):
                U[i][c2] = U3[i][c]
            for j in range(9):
                V[9+j][c2] = V3[j][c]  # B2 rows
                W[9+j][c2] = W3[j][c]  # C2 rows (k*3+i where k is 0,1,2 -> cols 3,4,5)
        
        result = {"U": U, "V": V, "W": W}
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return result
        
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
